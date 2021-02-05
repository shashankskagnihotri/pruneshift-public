""" Implements losses and criterions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy



class StandardLoss(nn.Module):

    def forward(self, network: nn.Module, batch):
        x, y = batch
        logits = network(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(torch.argmax(logits, 1), y)
        return loss, {"acc": acc} 


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


