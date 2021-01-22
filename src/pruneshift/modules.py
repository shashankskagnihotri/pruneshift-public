from bisect import bisect_right
from typing import Sequence
from functools import partial

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.metrics import functional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F


class MultiStepWarmUpLr(LambdaLR):
    def __init__(self, optimizer, milestones: Sequence[int], warmup_end: int = 0, gamma: float = 0.1):
        """Learning rate scheduler that supports a warmup period with milestones."""

        def lr_schedule(epoch):
            if epoch < warmup_end:
                return (epoch + 1) / warmup_end
            return gamma ** bisect_right(milestones, epoch)

        super(MultiStepWarmUpLr, self).__init__(optimizer, lr_schedule)


def standard_loss(network: nn.Module, batch):
    x, y = batch
    logits = network(x)
    loss = F.cross_entropy(logits, y)
    acc = functional.accuracy(torch.argmax(logits, 1), y)
    return loss, acc


def augmix_loss(network: nn.Module, batch, alpha: float = 0.0001):
    """ Loss for the augmix datasets. Adopted from the augmix repository.

    Args:
        network: The network.
        batch: The input should consist out of three parts [orig, modified, modified]
        alpha: Balancing the loss function.

    Returns:
        The combined loss, the accuracy.
    """
    x, y = batch
    logits = torch.split(network(torch.cat(x)), x[0].shape[0])

    p_clean, p_aug1, p_aug2 = F.softmax(
        logits[0], dim=1), F.softmax(
        logits[1], dim=1), F.softmax(
        logits[2], dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()

    loss_js = alpha * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                       F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                       F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
    loss = F.cross_entropy(logits[0], y)
    acc = functional.accuracy(torch.argmax(logits[0], 1), y)

    return loss + alpha * loss_js, acc


class VisionModule(pl.LightningModule):
    """Module for training and testing computer vision models."""

    def __init__(
        self,
        network: nn.Module,
        test_labels: Sequence[str] = None,
        learning_rate: float = 0.0001,
        augmix_loss_alpha: float = 0.0001,
        optimizer_fn=optim.Adam,
        scheduler_fn=None,
        monitor: str = None,
    ):

        super(VisionModule, self).__init__()
        self.network = network
        self.learning_rate = learning_rate
        self.augmix_loss_alpha = augmix_loss_alpha
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn
        self.monitor = monitor

        if test_labels is None:
            self.test_labels = ["acc"]
        else:
            self.test_labels = [f"acc_{l}" for l in test_labels]
        self.test_acc = nn.ModuleDict({l: Accuracy() for l in self.test_labels})

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        if not isinstance(batch[0], torch.Tensor):
            loss_fn = partial(augmix_loss, alpha=self.augmix_loss_alpha)
        else:
            loss_fn = standard_loss

        loss, acc = loss_fn(self, batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = standard_loss(self, batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx, dataset_idx=0):
        x, y = batch
        self.test_acc[self.test_labels[dataset_idx]](y, torch.argmax(self(x), -1))

    def test_epoch_end(self, outputs):
        for label, metric in self.test_acc.items():
            self.log(f"test_{label}", metric.compute())

    def configure_optimizers(self):
        if self.optimizer_fn is None:
            return

        optimizer = self.optimizer_fn(self.parameters(), lr=self.learning_rate)

        if self.scheduler_fn is None:
            return optimizer

        scheduler = {"scheduler": self.scheduler_fn(optimizer)}

        if self.monitor is not None:
            scheduler["monitor"] = self.monitor

        return [optimizer], [scheduler]
