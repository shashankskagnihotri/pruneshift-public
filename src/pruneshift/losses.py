""" Implements losses and criterions."""
import logging
from typing import Dict
from collections import UserDict

import pandas
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy
import torchvision.models
import timm.models
import scalable_resnet
import models
from SupCon.losses import SupConLoss
from crd.criterion import CRDLoss
from pruneshift.teachers import Teacher
from pruneshift.network_markers import at_entry_points
from pruneshift.network_markers import classifier
from .utils import ImagenetSubsetWrapper

logger = logging.getLogger(__name__)



def js_divergence(logits0, logits1, logits2):
    """ Calculates the Jensen-Shannon divergence.

    Adopted from ImageNet-R: https://github.com/hendrycks/imagenet-r
    """
    p_clean, p_aug1, p_aug2 = (
        F.softmax(logits0, dim=1),
        F.softmax(logits1, dim=1),
        F.softmax(logits2, dim=1),
    )

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, 1e-7, 1).log()

    return (
                   F.kl_div(p_mixture, p_clean, reduction="batchmean")
                   + F.kl_div(p_mixture, p_aug1, reduction="batchmean")
                   + F.kl_div(p_mixture, p_aug2, reduction="batchmean")
           ) / 3.0


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):        
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


class StandardLoss(nn.Module):
    def __init__(self, network: nn.Module, datamodule=None, supCon: bool= False, multiheaded: bool= False):
        super(StandardLoss, self).__init__()
        self.network = network
        self.supCon = supCon
        self.multiheaded= True
        #self.multiheaded= False 
        total_mask=0
        non_zero=0
        for a, p in self.network.named_buffers():
            if 'weight_mask' in a:
                a_copy=p.detach().cpu().numpy()
                total_mask+=len(a_copy.flatten())
                non_zero+=numpy.count_nonzero(a_copy)
        print("\n\nNumber of parameters: ", 
                (sum(p.numel() for p in self.network.parameters() 
                    if p.requires_grad)-total_mask+non_zero))


    def forward(self, batch):
        _, x, y = batch
        if self.supCon:
            self.network.encoder.eval()
            with torch.no_grad():
                features=self.network.encoder(x)
            logits=self.network.classifier(features)
        elif self.multiheaded:
            logits1, logits2, logits3, logits = self.network(x)
            loss1 = F.cross_entropy(logits1, y)
            loss2 = F.cross_entropy(logits2, y)
            loss3 = F.cross_entropy(logits3, y)
            if self.network.training:
                loss1.backward(retain_graph=True)
                loss2.backward(retain_graph=True)
                loss3.backward(retain_graph=True)

        else:
            logits=self.network(x)
        loss=F.cross_entropy(logits,y)
        acc=accuracy(torch.argmax(logits,1),y)
        return loss, {"acc":acc}
		#if type(x) is tuple:
		#	logits = self.network(torch.cat(x))
		#	logits, _, _ = torch.split(logits, len(logits) // 3)


class AugmixLoss(nn.Module):
    def __init__(
            self,
            network: nn.Module,
            datamodule,
            augmix_alpha: float = 12.0,
            supCon: bool = False,
            multiheaded: bool = False
    ):
        """Implements the AugmixLoss from the augmix paper.

        Args:
            alpha: Multiplitave factor for the jensen-shannon divergence.
        """
        super(AugmixLoss, self).__init__()
        self.network = network
        self.augmix_alpha = augmix_alpha
        self.supCon = supCon
        self.multiheaded = True 
        #self.multiheaded = False

    def forward(self, batch):
        idx, x, y = batch

        # Split the augmix samples.
        if self.supCon:
            self.network.encoder.eval()
            self.network.classifier.train()
            with torch.no_grad():
                features = self.network.encoder(torch.cat(x))
            logits=self.network.classifier(features)
        elif self.multiheaded:
            logits1, logits2, logits3, logits = self.network(torch.cat(x))
            loss1 = F.cross_entropy(torch.split(logits1, len(logits1)//3)[0], y)
            loss2 = F.cross_entropy(torch.split(logits2, len(logits2)//3)[0], y)
            loss3 = F.cross_entropy(torch.split(logits3, len(logits3)//3)[0], y)
            if self.network.training:
                loss1.backward(retain_graph=True)
                loss2.backward(retain_graph=True)
                loss3.backward(retain_graph=True)
        else:
            logits = self.network(torch.cat(x))
        logits = torch.split(logits, len(logits) // 3)

        loss_js = self.augmix_alpha * js_divergence(*logits)
        loss = F.cross_entropy(logits[0], y)
        acc = accuracy(torch.argmax(logits[0], 1), y)

        stats = {"acc": acc, "kl_loss": loss, "augmix_loss": loss_js}

        return loss + loss_js, stats


class KnowledgeDistill(nn.Module):
    def __init__(
            self,
            network: nn.Module,
            datamodule,
            teacher: Teacher,
            augmix: bool = False,
            augmix_alpha: float = 12.,
            augmix_jensen: bool = False,
            beta: float = 1,
            kd_T: float = 4.0,
            kd_mixture: float = 0.9,
            only_smooth: bool = False,
    ):
        super(KnowledgeDistill, self).__init__()
        self.network = network
        self.teacher = teacher
        self.kd_T = kd_T  # temperature for KD
        self.kd_mixture = kd_mixture  # scaling for the KD loss
        self.only_smooth = only_smooth
        self.augmix = augmix
        self.augmix_alpha = augmix_alpha
        self.augmix_jensen = augmix_jensen 
        self.beta = beta

    def kd_loss(self, student_logits, teacher_logits, y):
        batch_size = teacher_logits.shape[0]
        num_classes = teacher_logits.shape[1]

        p_t = F.softmax(teacher_logits / self.kd_T, dim=1)

        if self.augmix:
            y = torch.cat([y] * 3)

        if self.only_smooth:
            # Just use the smoothing effect.
            # This implementation could probably be more efficient.
            class_p = p_t[range(batch_size), y]
            non_class_p = (1 - class_p) / (num_classes - 1)
            p_t = torch.zeros_like(p_t) + non_class_p[:, None]
            p_t[range(batch_size), y] = class_p

        p_s = F.log_softmax(student_logits / self.kd_T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.kd_T ** 2) / batch_size

        return loss

    def forward(self, batch):
        idx, x, y = batch

        if self.augmix:
            x = torch.cat(x)

        logits = self.network(x)

        with torch.no_grad():
            teacher_logits = self.teacher(idx, x)

        loss_kd = self.kd_mixture * self.kd_loss(logits, teacher_logits, y)

        stats = {}

        if self.augmix:
            logits, logits_aug0, logits_aug1 = torch.split(logits, len(logits) // 3)

            if self.augmix_jensen:
                _, _, logits_aug1 = torch.split(teacher_logits, len(teacher_logits) // 3)
    
            loss_js = js_divergence(logits, logits_aug0, logits_aug1) * self.augmix_alpha * self.beta
            stats["loss_augmix"] = loss_js
            loss_cls = (F.cross_entropy(logits, y)+ loss_js) * (1 - self.kd_mixture)

        else:
            loss_cls = F.cross_entropy(logits, y) * (1 - self.kd_mixture)
        stats["cross_entropy_loss"] = loss_cls
        stats["acc"] = accuracy(torch.argmax(logits, 1), y)
        stats["loss_kd"] = loss_kd

        return loss_cls + loss_kd, stats


class AugmixKnowledgeDistill(KnowledgeDistill):
    def __init__(
            self,
            network: nn.Module,
            teacher: Teacher,
            datamodule,
            kd_T: float = 4.0,
            kd_mixture: float = 0.5,
            only_smooth: bool = 0.5,
            augmix_alpha: float = 6.0,
    ):
        super(AugmixKnowledgeDistill, self).__init__(network, teacher, kd_T, kd_mixture, only_smooth)
        self.augmix_alpha = augmix_alpha  # scaling for the augmix loss

    def forward(self, batch):
        idx, x, y = batch

        comb_x = torch.cat(x)

        student_logits = self.network(comb_x)
        with torch.no_grad():
            teacher_logits = self.teacher(idx, comb_x)

        loss_kd = self.kd_mixture * self.kd_loss(student_logits, teacher_logits, y)
        logits = torch.split(student_logits, student_logits.shape[0] // 3)

        loss_js = js_divergence(*logits) * self.augmix_alpha

        loss = (1 - self.kd_mixture) * F.cross_entropy(logits[0], y)
        acc = accuracy(torch.argmax(logits[0], 1), y)
        loss_cls = loss * (1 - self.kd_mixture) + loss_js

        stats = {
            "acc": acc,
            "cross_entropy_loss": loss,
            "augmix_loss": loss_js,
            "KD_loss": loss_kd,
        }

        return loss_cls + loss_kd, stats


class ActivationCollector(UserDict):
    """ Collects activations from forward passes."""

    def __init__(self, modules: Dict, mode="out"):
        super(ActivationCollector, self).__init__()
        assert mode in ["out", "in"]
        self.modules = modules
        self.mode = mode
        self.prepare()

    def ensure_hook(self, name: str, module: nn.Module):
        """ Ensures that there are hooks in a module."""

        def save_activation(module, inputs, outputs):
            if self.mode == "in":
                self.data[name] = inputs[0]
                return
            if isinstance(outputs, tuple):
                # This can be removed when we only
                # use the hooks to get activations from models.
                # if we delete the is_feat part.
                outputs = outputs[0]
            self.data[name] = outputs

        # Check whether the module was already registered.
        if hasattr(module, "__activation_hook"):
            return

        module.register_forward_hook(save_activation)
        module.__activation_hook = None

    def prepare(self):
        for name, module in self.modules.items():
            self.ensure_hook(name, module)

    def reset(self):
        self.data = {}


class AttentionDistill(nn.Module):
    def __init__(
            self,
            network: nn.Module,
            teacher: Teacher,
            datamodule,
            p: float = 1.0,
            beta: float = 2.0,
            charlie: float = 0.,
            kd_T: float = 4.,
            augmix: bool = False,
            augmix_alpha: float = 12,
            augmix_average: bool = False,
            **kwargs,
    ):
        super(AttentionDistill, self).__init__()
        self.network = network
        self.teacher = teacher
        # at_entry_points returns the modules that should be used for
        # Attention distillation.
        self.student_collector = ActivationCollector(at_entry_points(network))
        self.teacher_collector = ActivationCollector(at_entry_points(teacher))
        self.beta = beta
        self.p = p
        self.augmix = augmix
        self.augmix_alpha = augmix_alpha
        self.augmix_average = augmix_average
        self.charlie = charlie
        self.criterion_kd = DistillKL(kd_T)

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

        # Vectorized attention maps.
        student_attention = self.attention_map(student_features)
        teacher_attention = self.attention_map(teacher_features)

        return torch.norm(student_attention - teacher_attention, 2, dim=1).mean() / 2

    def forward(self, batch):
        idx, x, y = batch
        stats = {}

        if self.augmix:
            x = torch.cat(x)

        # Teacher forward pass.
        with torch.no_grad():
            teacher_logits = self.teacher(idx, x)

        # Studend forward pass.
        if self.augmix:
            logits = self.network(x)
            logits, logits_aug1, logits_aug2 = torch.split(logits, len(logits) // 3)
        else:
            logits = self.network(x)

        # Calculate the cross entropy loss.
        loss = (1 - self.charlie) * F.cross_entropy(logits, y)
        stats["cross_entropy_loss"] = loss
        stats["acc"] = accuracy(torch.argmax(logits, 1), y)

        # Calculate attention map losses.
        at_losses = []

        assert len(self.student_collector)
        for module_name in self.student_collector:
            activation_student = self.student_collector[module_name]
            activation_teacher = self.teacher_collector[module_name]

            if self.augmix and self.augmix_average:
                # Average the activations.
                n_s = len(activation_student) // 3
                activation_student = sum(torch.split(activation_student, n_s)) / 3
                n_t = len(activation_teacher) // 3
                activation_teacher = sum(torch.split(activation_teacher, n_t)) / 3

            at_loss = self.attention_distance(activation_student, activation_teacher)
            stats["loss_AT_" + module_name] = at_loss
            at_losses.append(at_loss)

        at_loss = self.beta * sum(at_losses)
        stats["loss_AT"] = at_loss

        if self.charlie > 0:
            loss_kd = self.criterion_kd(logits, teacher_logits) * self.charlie
            stats["loss_KD"] = loss_kd
        else:
            loss_kd = 0

        # Reset everything to delete the graphs (student).
        self.teacher_collector.reset()
        self.student_collector.reset()

        if self.augmix:
            augmix_loss = self.augmix_alpha * js_divergence(logits, logits_aug1, logits_aug2)
            stats["loss_augmix"] = augmix_loss
            return loss + at_loss + augmix_loss + loss_kd, stats

        return loss + at_loss + loss_kd, stats


class ContrastiveDistill(nn.Module):
    def __init__(self, network: nn.Module, datamodule, teacher: Teacher,
                 kd_T: float = 4., augmix: bool = False, augmix_alpha: float = 12.,
                 beta: float = 0.1,
                 gamma: float = 0.1,
                 charlie: float = 0.1, delta: float = 0.8,
                 feat_dim: int = 128, nce_k: int = 16384,
                 nce_t: int = 0.07, nce_m: int = 0.5, **kwargs):
        super(ContrastiveDistill, self).__init__()
        self.network = network
        self.datamodule = datamodule
        self.teacher = teacher
        self.kd_T = kd_T  # temperature for KD
        self.augmix = augmix
        self.augmix_alpha = augmix_alpha  # scaling for the augmix loss
        self.gamma = gamma  # scaling for the classification loss
        self.beta = beta
        self.charlie = charlie  # scaling for the KD loss
        self.delta = delta  # scaling for the CRD loss
        self.feat_dim = feat_dim  # the dimension of the projection space

        s_dim = classifier(network).weight.shape[1]
        t_dim = classifier(teacher).weight.shape[1]
        n_data = datamodule.train_length()
        self.criterion_kd = CRDLoss([s_dim, t_dim, n_data, feat_dim, nce_k, nce_t, nce_m])
        self.student_collector = ActivationCollector({"classifier": classifier(network)}, mode="in")
        self.teacher_collector = ActivationCollector({"classifier": classifier(teacher)}, mode="in")

    def forward(self, batch):
        idx, x, y, contrast_idx = batch

        if self.augmix:
            x = torch.cat(x)

        criterion_dv = DistillKL(self.kd_T)

        self.network.train()
        logits_student = self.network(x)

        with torch.no_grad():
            # Is this correct?
            self.teacher.eval()
            logits_teacher = self.teacher(idx, x)

        # Changed that
        f_s = self.student_collector["classifier"]
        f_t = self.teacher_collector["classifier"]

        # TODO (Shashank): Is this the best way? Or should we average.
        if self.augmix:
            f_s = f_s[0: len(f_s) // 3]
            f_t = f_t[0: len(f_t) // 3]

        stats = {}

        loss_crd = self.criterion_kd(f_s, f_t, idx, contrast_idx) * self.delta
        stats["loss_crd"] = loss_crd

        loss_kd = criterion_dv(logits_student, logits_teacher) * self.charlie
        stats["loss_kd"] = loss_kd

        # TODO (Shashank): Remove beta later on.
        if self.augmix:
            logits, logits_aug1, logits_aug2 = torch.split(logits_student, logits_student.shape[0] // 3)
            loss_js = self.augmix_alpha * js_divergence(logits, logits_aug1, logits_aug2) * self.beta
            stats["loss_augmix"] = loss_js
        else:
            logits = logits_student

        loss_cls = F.cross_entropy(logits, y)
        stats["cross_entropy_loss"] = loss_cls
        stats["acc"] = accuracy(torch.argmax(logits, 1), y)

        if self.augmix:
            return loss_cls + loss_kd + loss_crd + loss_js, stats

        return loss_cls + loss_kd + loss_crd, stats



class SupCon(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, network: nn.Module, temperature=0.1, contrast_mode='all', augmix:bool=False,
                 base_temperature=0.07, **kwargs):
        super(SupCon, self).__init__()
        self.network=network
        self.temperature = temperature
        self.augmix=augmix
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.criterion = SupConLoss(temperature=base_temperature)
        #in_feats=self.network.projection.out_features
        #self.classification = nn.Linear(in_feats,100)
      

    def forward(self, batch):
        idx, x, labels = batch
        #print('\n\n\n\n\nAugmix:', self.augmix)

        if self.augmix:
            x = torch.cat(x)
        #x=torch.cat(x)
        bsz = labels.shape[0]
        features = self.network(x)
        #logits=self.classification(features)
        if self.augmix:
            #logits_clean, logits_aug1, logits_aug2= torch.split(logits, logits.shape[0] //3)
            f1, f2, f3 = torch.split(features, [bsz, bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
        else:
            #logits_clean = logits
            features = torch.cat([features.unsqueeze(1)], dim=1)

        loss = self.criterion(features, labels)
        #acc = accuracy(torch.argmax(logits_clean,1), labels)
        #acc= 0
        stats = {          
            "SupConLoss": loss,
        }
        return loss, stats


class KD_SupCon(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    """
    def __init__(self, network: nn.Module, teacher: Teacher, temperature=0.1, contrast_mode='all', augmix:bool=False,
            base_temperature=0.07, feat_dim:int=128, **kwargs):
        super(KD_SupCon, self).__init__()
        self.network=network
        with torch.no_grad():
            self.teacher=teacher
        self.temperature = temperature
        self.augmix=augmix
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.criterion = SupConLoss(temperature=base_temperature)
        #in_feats=self.network.projection.out_features
        #self.classification=nn.Linear(in_feats,100)
        with torch.no_grad():
            self.teacher_collector = ActivationCollector({"classifier": classifier(teacher)}, mode="in")
            RESNET_CLASSES = (torchvision.models.ResNet, models.ResNet, timm.models.ResNet, scalable_resnet.ResNet)
            if isinstance(self.teacher.network, RESNET_CLASSES):
                dim_in=self.teacher.network.fc.in_features
            else:
                dim_in=self.teacher.network.network.fc.in_features
            feat_dim=feat_dim
            self.flatten=nn.Flatten()
            self.contrast=nn.Linear(dim_in, dim_in)
            self.relu=nn.ReLU(inplace=True)
            self.projection=nn.Linear(dim_in, feat_dim)        


    def forward(self, batch):
        idx, x, labels = batch
        if self.augmix:
            x=torch.cat(x)

        bsz = labels.shape[0]
        features=self.network(x)
        with torch.no_grad():
            self.teacher(idx,x)

            teacher_features=self.teacher_collector["classifier"]
        with torch.no_grad():
            teacher_features=F.normalize(self.projection(self.relu((self.contrast(self.flatten(teacher_features))))))
            #logits=self.classification(features)
        if self.augmix:
            with torch.no_grad():
                #logits_clean, logits_aug1, logits_aug2= torch.split(logits, logits.shape[0] //3)
                t_f1, t_f2, t_f3 = torch.split(teacher_features, [bsz, bsz, bsz], dim=0)
            f1, f2, f3 = torch.split(features, [bsz, bsz, bsz], dim=0)

            t_f1, t_f2, t_f3 = torch.split(teacher_features, [bsz, bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1), t_f1.unsqueeze(1), t_f2.unsqueeze(1), t_f3.unsqueeze(1)], dim=1)
        else:
            with torch.no_grad():
                #logits_clean=logits
                features = torch.cat([features.unsqueen(1), teacher_features.unsqueeze(1)], dim=1)

        loss = self.criterion(features, labels)
        #acc = accuracy(torch.argmax(logits_clean,1), labels)
        #acc=0
        stats = {"SupConLoss": loss,}
                                                
        return loss, stats

