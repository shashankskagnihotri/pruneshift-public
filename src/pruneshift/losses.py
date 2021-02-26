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
from torch.utils.data import DataLoader


    
class StandardLoss(nn.Module):
    def forward(self, network: nn.Module, batch):
        idx, x, y = batch
        if network.training:
            _, logits = network(x)
        else:
            logits = network(x)
        #import ipdb;ipdb.set_trace()
        #print(y.__class__)
        #print(logits.__class__)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(torch.argmax(logits, 1), y)
        return loss, {"acc": acc}


class KnowledgeDistill(nn.Module):
    def __init__(
        self,
        teacher: Teacher,
        kd_T: float = 4.0,
        gamma: float = 0.1,
        charlie: float = 0.9,
    ):
        super(KnowledgeDistill, self).__init__()
        self.teacher = teacher
        self.kd_T = kd_T  # temperature for KD
        self.gamma = gamma  # scaling for the classification loss
        self.charlie = charlie  # scaling for the KD loss

    def forward(self, network: nn.Module, batch):
        idx, x, y = batch
        _, logits = network(x)
        # print("\n\n\n\n")
        # print(logits)
        criterion_dv = DistillKL(self.kd_T)
        self.teacher.eval()
        with torch.no_grad():
            _ ,teacher_logits = self.teacher(idx, x)
        loss = F.cross_entropy(logits, y) * self.gamma
        loss_kd = criterion_dv(logits, teacher_logits) * self.charlie
        acc = accuracy(torch.argmax(logits, 1), y)
        stats = {"acc": acc, "kl_loss": loss, "KD_loss": loss_kd}
        return loss + loss_kd, stats


class AugmixKnowledgeDistill(nn.Module):
    def __init__(
        self,
        teacher: Teacher,
        kd_T: float = 4.0,
        charlie: float = 0.5,
        alpha: float = 12.0,
        beta: float = 0.5,
    ):
        """Implements the AugmixLoss from the augmix paper.

        Args:
            alpha: Multiplitave factor for the jensen-shannon divergence.
        """
        super(AugmixKnowledgeDistill, self).__init__()
        self.alpha = alpha  # scaling for the augmix loss
        self.beta = beta  # scaling for the classification loss
        self.teacher = teacher
        self.kd_T = kd_T  # temperature for KD
        self.charlie = charlie  # scaling for the KD loss

    def forward(self, network: nn.Module, batch):
        idx, x, y = batch

        comb_x = torch.cat(x)

        self.teacher.eval()
        criterion_dv = DistillKL(self.kd_T)

        _, kd_logits = network(comb_x)
        with torch.no_grad():
            if self.teacher.training:
                _, teacher_logits = self.teacher(idx, comb_x)
            else:
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
        self, alpha: float = 12.0, beta: float = 1.0, soft_target: bool = False
    ):
        """Implements the AugmixLoss from the augmix paper.

        Args:
            alpha: Multiplitave factor for the jensen-shannon divergence.
        """
        super(AugmixLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.soft_target = soft_target
        if soft_target:
            raise NotImplementedError

    def forward(self, network: nn.Module, batch):
        idx, x, y = batch

        _, logits = network(torch.cat(x))
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
    def __init__(self, teacher:Teacher, teacher_path,
        teacher_model_id, kd_T: float = 4.,
        gamma:float = 0.1, charlie:float = 0.1,
        delta: float = 0.8, feat_dim: int=128, 
        nce_k:int= 16384, nce_t:int=0.07, nce_m:int= 0.5,
        percent:float=1.0, mode:str='exact'):
        
        super(CRD_Loss, self).__init__()
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
        self.feat_dim=feat_dim
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

        feat_s, logit_s = network(x, is_feat=True, preact=preact)
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
    def __init__(self, teacher:Teacher, teacher_path, teacher_model_id,
        kd_T: float = 4., alpha:float=12., gamma:float = 0.1,
        charlie:float = 0.1, delta: float = 0.8,
        feat_dim: int=128, nce_k:int= 16384,
        nce_t:int=0.07, nce_m:int= 0.5,
        percent:float=1.0, mode:str='exact', k=4096,
        s_dim=None, t_dim=None, n_data=None):
        
        super(Augmix_CRD_Loss, self).__init__()
        #self.teacher_network = create_network(teacher_model_id, ckpt_path=teacher_path)
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
    
    def forward(self, network: nn.Module, batch):
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

        
        network.train()
        feat_s, _ = network(comb_x)
        _, logit_s = network(comb_x)
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
