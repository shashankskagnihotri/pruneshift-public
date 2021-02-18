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


    
class StandardLoss(nn.Module):
    def forward(self, network: nn.Module, batch):
        idx, x, y = batch
        logits = network(x)
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

        kd_logits = network(comb_x)
        with torch.no_grad():
            teacher_logits = self.teacher(idx, comb_x)

        loss_kd = criterion_dv(kd_logits, teacher_logits) * self.charlie
        logits = torch.split(kd_logits, logits.shape[0] // 3)

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

        logits = network(torch.cat(x))
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
