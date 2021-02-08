""" Implements losses and criterions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy

from ..distiller_zoo import DistillKL
from ..crd.criterion import CRDLoss


class StandardLoss(nn.Module):

    def forward(self, network: nn.Module, batch):
        x, y = batch
        logits = network(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(torch.argmax(logits, 1), y)
        return loss, {"acc": acc} 

class KnowledgeDistill(nn.Module):

    def __init__(self, teacher_path, teacher_model_id, kd_T: float = 4., gamma:float = 0.1, charlie:float = 0.9):
        self.teacher_network = create_network(teacher_model_id, ckpt_path=teacher_path)
        self.kd_T = kd_T 	 	#temperature for KD
        self.gamma = gamma 	 	#scaling for the classification loss
        self.charlie = charlie  	#scaling for the KD loss

    def forward(self, network: nn.Module, batch):
        x,y = batch
        criterion_dv = DistillKL(self.kd_T)
        logits = network(x)
        self.teacher_network.eval()
        with torch.no_grad():
            teacher_logits = self.teacher_network(x)
        loss = F.cross_entropy(logits, y) * self.gamma
        loss_kd = criterion_dv(logits, teacher_logits) * self.charlie
        acc = accuracy(torch.argmax(logits, 1), y)
        stats = {"acc": acc, "kl_loss": loss, "KD_loss": loss_kd}
        return loss+loss_kd , stats

class Augmix_KnowledgeDistill(nn.Module):

    def __init__(self, teacher_path, teacher_model_id, kd_T: float = 4., charlie:float = 0.9, self, alpha: float = 12., beta: float = 1.):
        """ Implements the AugmixLoss from the augmix paper.

        Args:
            alpha: Multiplitave factor for the jensen-shannon divergence.
        """ 
        super(AugmixLoss, self).__init__()
        self.alpha = alpha		#scaling for the augmix loss
        self.beta = beta 		#scaling for the classification loss
        self.teacher_network = create_network(teacher_model_id, ckpt_path=teacher_path)
        self.kd_T = kd_T 	 	#temperature for KD
        self.charlie = charlie  	#scaling for the KD loss


    def forward(self, network: nn.Module, batch):
        x, y = batch
        self.teacher_network.eval()
        criterion_dv = DistillKL(self.kd_T)
        
        logits = torch.split(network(torch.cat(x)), x[0].shape[0])
        with torch.no_grad():
            teacher_logits = torch.split(self.teacher_network(torch.cat(x)), x[0].shape[0])
        
        loss_kd = criterion_dv(logits, teacher_logits) * self.charlie
    
        p_clean, p_aug1, p_aug2 = F.softmax(
            logits[0], dim=1), F.softmax(
            logits[1], dim=1), F.softmax(
            logits[2], dim=1)
    
        # Clamp mixture distribution to avoid exploding KL divergence
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    
        loss_js = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                   F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                   F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
        loss_js *= self.alpha

        loss = self.beta * F.cross_entropy(logits[0], y)
        acc = accuracy(torch.argmax(logits[0], 1), y)
    
        stats = {"acc": acc, "kl_loss": loss, "augmix_loss": loss_js, "KD_loss": loss_kd}
    
        return loss + loss_js + loss_kd, stats 

class AugmixLoss(nn.Module):

    def __init__(self, alpha: float = 12., beta: float = 1.):
        """ Implements the AugmixLoss from the augmix paper.

        Args:
            alpha: Multiplitave factor for the jensen-shannon divergence.
        """ 
        super(AugmixLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta 


    def forward(self, network: nn.Module, batch):
        x, y = batch
        logits = torch.split(network(torch.cat(x)), x[0].shape[0])
    
        p_clean, p_aug1, p_aug2 = F.softmax(
            logits[0], dim=1), F.softmax(
            logits[1], dim=1), F.softmax(
            logits[2], dim=1)
    
        # Clamp mixture distribution to avoid exploding KL divergence
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    
        loss_js = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                   F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                   F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
        loss_js *= self.alpha

        loss = self.beta * F.cross_entropy(logits[0], y)
        acc = accuracy(torch.argmax(logits[0], 1), y)
    
        stats = {"acc": acc, "kl_loss": loss, "augmix_loss": loss_js}
    
        return loss + loss_js, stats 


