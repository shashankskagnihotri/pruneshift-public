""" Implements losses and criterions."""
import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy

from distiller_zoo import DistillKL
from crd.criterion import CRDLoss
from pruneshift.teachers import Teacher
from pruneshift.teachers import DatabaseNetwork
from torch.utils.data import DataLoader


    
class StandardLoss(nn.Module):
    def __init__(self, network: nn.Module, **kwargs):
        super(StandardLoss, self).__init__()
        self.network = network

    def forward(self, batch):
        idx, x, y = batch
        logits = self.network(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(torch.argmax(logits, 1), y)
        return loss, {"acc": acc}


class KnowledgeDistill(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        teacher: Teacher,
        kd_T: float = 4.0,
        gamma: float = 0.1,
        charlie: float = 0.9,
        **kwargs,
    ):
        super(KnowledgeDistill, self).__init__()
        self.network = network
        self.teacher = teacher
        self.kd_T = kd_T  # temperature for KD
        self.gamma = gamma  # scaling for the classification loss
        self.charlie = charlie  # scaling for the KD loss

    def forward(self, batch):
        idx, x, y = batch
        logits = network(x)
        # print("\n\n\n\n")
        # print(logits)
        criterion_dv = DistillKL(self.kd_T)
        self.teacher.eval()
        with torch.no_grad():
            teacher_logits = self.teacher(idx, x)
        loss = F.cross_entropy(logits, y) * self.gamma
        loss_kd = criterion_dv(logits, teacher_logits) * self.charlie
        acc = accuracy(torch.argmax(logits, 1), y)
        teacher_acc = accuracy(torch.argmax(teacher_logits, 1), y)
        stats = {
            "teacher_acc": teacher_acc,
            "acc": acc,
            "kl_loss": loss,
            "KD_loss": loss_kd,
        }
        return loss + loss_kd, stats


class AttentionDistill(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        teacher: Teacher,
        kd_T: float = 4.0,
        p: float = 1.0,
        mixture: float = 1.0,
        **kwargs,
    ):
        super(AttentionDistill, self).__init__()
        self.network = network
        self.teacher_activations = {}
        self.student_activations = {}
        self.teacher = teacher
        self.mixture = mixture
        self.p = p

    def attention_map(self, features):
        """ Adopted from the RepDistiller repository."""
        f = features
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))

    def attention_distance(self, student_features, teacher_features):
        student_height = student_features.shape[2]
        teacher_height = teacher_features.shape[2]

        if student_height > teacher_height:
            target_shape = (teacher_height, teacher_height)
            student_features = F.adaptive_avg_pool2d(student_features, target_shape)
        elif student_height < teacher_height:
            target_shape = (student_height, student_height)
            teacher_features = F.adaptive_avg_pool2d(student_features, target_shape)

        student_attention = self.attention_map(student_features)
        teacher_attention = self.attention_map(teacher_features)

        return torch.norm(student_attention - teacher_attention, 2) / 2

    def ensure_hook(self, module_name: str, network: nn.Module, teacher: bool):
        """ Ensures that there are hooks in a module."""

        def save_activation(module, inputs, outputs):
            if teacher:
                self.teacher_activations[module_name] = outputs
            else:
                self.student_activations[module_name] = outputs

        module = getattr(network, module_name)

        # Check wether the module was already registered.
        if hasattr(module, "__activation_hook"):
            return

        module.register_forward_hook(save_activation)
        module.__activation_hook = None

    def target_modules(self):
        return ["layer1", "layer2", "layer3"]

    def prepare_activations(self, student: nn.Module, teacher: nn.Module):
        for module_name in self.target_modules:
            ensure_hook(module_name, student, False)
            if isinstance(teacher, DatabaseNetwork):
                raise NotImplementedError
            else:
                ensure_hook(module_name, teacher.network, True)

    def forward(self, batch):
        idx, x, y = batch
        self.prepare_activations(self.student, self.teacher)
        stats = {}

        # Calculate normal loss
        logits = self.network(x)
        loss = F.cross_entropy(logits, y)
        stats["acc"] = accuracy(torch.argmax(logits, 1), y)

        # Calculate attention map losses.
        at_losses = []
        for module_name in self.student_activations:
            at_loss = self.attention_distance(
                self.student_activations[module_name],
                self.teacher_activations[module_name],
            )
            stats["at_loss_" + module_name] = at_loss
            at_losses.append(at_loss)

        at_loss = sum(at_losses)
        stats["loss"] = at_loss

        return loss + self.beta * at_loss, stats


class AugmixKnowledgeDistill(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        teacher: Teacher,
        kd_T: float = 4.0,
        charlie: float = 0.5,
        alpha: float = 12.0,
        beta: float = 0.5,
        **kwargs,
    ):
        """Implements the AugmixLoss from the augmix paper.

        Args:
            alpha: Multiplitave factor for the jensen-shannon divergence.
        """
        super(AugmixKnowledgeDistill, self).__init__()
        self.network = network
        self.alpha = alpha  # scaling for the augmix loss
        self.beta = beta  # scaling for the classification loss
        self.teacher = teacher
        self.kd_T = kd_T  # temperature for KD
        self.charlie = charlie  # scaling for the KD loss

    def forward(self, batch):
        idx, x, y = batch

        comb_x = torch.cat(x)

        self.teacher.eval()
        criterion_dv = DistillKL(self.kd_T)

        kd_logits = self.network(comb_x)
        with torch.no_grad():
            teacher_logits = self.teacher(idx, comb_x)

        loss_kd = criterion_dv(kd_logits, teacher_logits) * self.charlie
        logits = torch.split(kd_logits, kd_logits.shape[0] // 3)

        p_clean, p_aug1, p_aug2 = (
            F.softmax(logits[0], dim=1),
            F.softmax(logits[1], dim=1),
            F.softmax(logits[2], dim=1),
        )

        # Clamp mixture distribution to avoid exploding KL divergence
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, 1e-7, 1).log()

        loss_js = (
            F.kl_div(p_mixture, p_clean, reduction="batchmean")
            + F.kl_div(p_mixture, p_aug1, reduction="batchmean")
            + F.kl_div(p_mixture, p_aug2, reduction="batchmean")
        ) / 3.0
        loss_js *= self.alpha

        loss = F.cross_entropy(logits[0], y)
        acc = accuracy(torch.argmax(logits[0], 1), y)
        loss_cls = (loss + loss_js) * self.beta

        stats = {
            "acc": acc,
            "kl_loss": loss,
            "augmix_loss": loss_js,
            "KD_loss": loss_kd,
        }

        return loss_cls + loss_kd, stats


class AugmixLoss(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        alpha: float = 12.0,
        beta: float = 1.0,
        soft_target: bool = False,
        **kwargs,
    ):
        """Implements the AugmixLoss from the augmix paper.

        Args:
            alpha: Multiplitave factor for the jensen-shannon divergence.
        """
        super(AugmixLoss, self).__init__()
        self.network = network
        self.alpha = alpha
        self.beta = beta

        self.soft_target = soft_target
        if soft_target:
            raise NotImplementedError

    def forward(self, batch):
        idx, x, y = batch

        logits = self.network(torch.cat(x))
        logits = torch.split(logits, logits.shape[0] // 3)

        p_clean, p_aug1, p_aug2 = (
            F.softmax(logits[0], dim=1),
            F.softmax(logits[1], dim=1),
            F.softmax(logits[2], dim=1),
        )

        # Clamp mixture distribution to avoid exploding KL divergence
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, 1e-7, 1).log()

        loss_js = (
            F.kl_div(p_mixture, p_clean, reduction="batchmean")
            + F.kl_div(p_mixture, p_aug1, reduction="batchmean")
            + F.kl_div(p_mixture, p_aug2, reduction="batchmean")
        ) / 3.0
        loss_js *= self.alpha

        loss = self.beta * F.cross_entropy(logits[0], y)
        acc = accuracy(torch.argmax(logits[0], 1), y)

        stats = {"acc": acc, "kl_loss": loss, "augmix_loss": loss_js}

        return loss + loss_js, stats


class CRD_Loss(nn.Module):
    def __init__(self, network: nn.Module, datamodule, teacher:Teacher, teacher_path,
        teacher_model_id, kd_T: float = 4.,
        gamma:float = 0.1, charlie:float = 0.1,
        delta: float = 0.8, feat_dim: int=128,
        nce_k:int= 16384, nce_t:int=0.07, nce_m:int= 0.5,
        percent:float=1.0, mode:str='exact', **kwargs):

        super(CRD_Loss, self).__init__()
        self.network = network
        self.datamodule = datamodule
        # You can access the dataset by using:
        # self.datamodule.train_dataset[0] -> idx, x, y    or for augmix idx, (x, x1, x2), y
        self.teacher = teacher
        #self.teacher_network = create_network(teacher_model_id, ckpt_path=teacher_path)
        self.kd_T = kd_T 	 	#temperature for KD
        self.gamma = gamma 	 	#scaling for the classification loss
        self.charlie = charlie  	#scaling for the KD loss
        self.delta = delta  		#scaling for the CRD loss
        self.feat_dim = feat_dim	#the dimension of the projection space
        self.nce_k = nce_k		#number of negatives paired with each positive
        self.nce_t = nce_t		#the temperature
        self.nce_m = nce_m		#the momentum for updating the memory buffer
        self.feat_dim = feat_dim
        self.percent = percent
        self.mode = mode

    def forward(self, network: nn.Module, batch):
        preact = False
        idx, x, y, contrast_idx = batch

        percent = self.percent
        label = y
        num_samples = len(x)
        feat_dim = self.feat_dim

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

        if self.mode == 'exact':
            pos_idx = idx
        elif self.mode == 'relax':
            pos_idx = np.random.choice(self.cls_positive[target], 1)
            pos_idx = pos_idx[0]

        replace = True if self.k > len(self.cls_negative[target]) else False
        neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
        constrast_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

        criterion_dv = DistillKL(self.kd_T)

        feat_s, logit_s = self.network(x, is_feat=True, preact=preact)
        #self.teacher_network.eval()
        self.teacher.eval()
        with torch.no_grad():
            #feat_t, logit_t = self.teacher_network(idx, x, is_feat=True, preact=preact)
            feat_t, logit_t = self.teacher(idx, x, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]
        f_s = feat_s[-1]
        f_t = feat_t[-1]
        s_dim = feat_s[-1].shape[1]
        t_dim = feat_t[-1].shape[1]
        n_data = len(x.cpu().detach().numpy())
        criterion_kd = CRDLoss([s_dim, t_dim, n_data, feat_dim, self.nce_k, self.nce_t, self.nce_m])
        loss_crd = criterion_kd(f_s, f_t, idx, constrast_idx) * self.delta
        loss = F.cross_entropy(logits, y) * self.gamma
        loss_kd = criterion_dv(logits, teacher_logits) * self.charlie
        acc = accuracy(torch.argmax(logits, 1), y)
        stats = {"acc": acc, "kl_loss": loss, "KD_loss": loss_kd, "CRD_loss": loss_crd}

        return loss+loss_kd+loss_crd , stats

class Augmix_CRD_Loss(nn.Module):
    def __init__(self, network: nn.Module, datamodule, teacher:Teacher, teacher_path, teacher_model_id,
        kd_T: float = 4., alpha:float=12., gamma:float = 0.1,
        charlie:float = 0.1, delta: float = 0.8,
        feat_dim: int=128, nce_k:int= 16384,
        nce_t:int=0.07, nce_m:int= 0.5,
        percent:float=1.0, mode:str='exact', k=4096,
        s_dim=None, t_dim=None, n_data=None, **kwargs):

        super(Augmix_CRD_Loss, self).__init__()
        #self.teacher_network = create_network(teacher_model_id, ckpt_path=teacher_path)
        self.network = network
        self.teacher=teacher
        self.kd_T = kd_T 	 	#temperature for KD
        self.alpha = alpha		#scaling for the augmix loss
        self.gamma = gamma 	 	#scaling for the classification loss
        self.charlie = charlie  	#scaling for the KD loss
        self.delta = delta  		#scaling for the CRD loss
        self.feat_dim = feat_dim	#the dimension of the projection space
        self.nce_k = nce_k		#number of negatives paired with each positive
        self.nce_t = nce_t		#the temperature
        self.nce_m = nce_m		#the momentum for updating the memory buffer
        self.percent = percent
        self.mode = mode
        self.k=k
        self.s_dim = s_dim
        self.t_dim = t_dim
        self.n_data = n_data
        self.criterion_kd = CRDLoss([ s_dim, t_dim, n_data, feat_dim, nce_k, nce_t, nce_m])

    def mimic_crd(x, y, idx, sample_idx):
        return x, y, idx, sample_idx

    def forward(self, batch):
        preact = False
        num_classes = 100
        idx, x, y = batch

        percent = self.percent
        label = y
        target = y
        target = target.cpu().detach().numpy()
        num_samples = len(x)
        feat_dim = self.feat_dim

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

        if self.mode == 'exact':
            pos_idx = idx
        elif self.mode == 'relax':
            pos_idx = np.random.choice(self.cls_positive[target], 1)
            pos_idx = pos_idx[0]

        replace = True if self.k > len(self.cls_negative[target]) else False
        neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))


        comb_x = torch.cat(x)
        contrast_idx = sample_idx[0]
        #contrast_idx.to(comb_x.device())

        criterion_dv = DistillKL(self.kd_T)


        self.network.train()
        feat_s, _ = self.network(comb_x)
        _, logit_s = self.network(comb_x)
        with torch.no_grad():
            #feat_t, logit_t = self.teacher_network(idx, comb_x, is_feat=True, preact=preact)
            self.teacher.train()
            feat_t, _ = self.teacher(idx, comb_x)
            self.teacher.eval()
            logits_t = self.teacher(idx, comb_x)
            device = feat_t[0].device
            feat_t = [f.detach() for f in feat_t]
        f_s = feat_s[-1]
        f_t = feat_t[-1]

        s_dim = feat_s[-1].shape[1]
        t_dim = feat_t[-1].shape[1]
        n_data = len(comb_x.cpu().detach().numpy())

        loss_crd = self.criterion_kd(f_s, f_t, idx, contrast_idx) * self.delta

        loss_kd = criterion_dv(logit_s, logit_t) * self.charlie

        logits = torch.split(logit_s, logit_s.shape[0] // 3)

        p_clean, p_aug1, p_aug2 = (
            F.softmax(logits[0], dim=1),
            F.softmax(logits[1], dim=1),
            F.softmax(logits[2], dim=1),
        )

        # Clamp mixture distribution to avoid exploding KL divergence
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, 1e-7, 1).log()

        loss_js = (
            F.kl_div(p_mixture, p_clean, reduction="batchmean")
            + F.kl_div(p_mixture, p_aug1, reduction="batchmean")
            + F.kl_div(p_mixture, p_aug2, reduction="batchmean")
        ) / 3.0
        loss_js *= self.alpha

        loss = F.cross_entropy(logits[0], y)
        acc = accuracy(torch.argmax(logits[0], 1), y)
        loss_cls = (loss + loss_js) * self.beta

        stats = {
            "acc": acc,
            "kl_loss": loss,
            "augmix_loss": loss_js,
            "KD_loss": loss_kd,
            "CRD_loss": loss_crd,
        }

        return loss_cls + loss_kd + loss_crd, stats
