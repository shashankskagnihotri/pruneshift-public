from bisect import bisect_right
from functools import partial
import logging
from typing import Sequence
from typing import Callable


import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.metrics import functional
import torch
import torch.nn as nn
from torch.nn.utils.prune import is_pruned
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F

from pruneshift.hydra import hydrate
from pruneshift.hydra import dehydrate


logger = logging.getLogger(__name__)



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
    return loss, {"acc": acc} 


def augmix_loss(network: nn.Module, batch, alpha: float = 12.):
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

    stats = {"acc": acc, "kl_loss": loss, "augmix_loss": loss_js}

    return loss + loss_js, stats 


class VisionModule(pl.LightningModule):
    """Module for training and testing computer vision models."""

    def __init__(
        self,
        network: nn.Module,
        test_labels: Sequence[str] = None,
        learning_rate: float = 0.1,
        augmix_loss_alpha: float = 12.,
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
        # TODO: Make this extensible for Shashank
        if not isinstance(batch[0], torch.Tensor):
            loss_fn = partial(augmix_loss, alpha=self.augmix_loss_alpha)
        else:
            loss_fn = standard_loss

        loss, stats = loss_fn(self, batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        for n, v in stats.items():
            n = "train_" + n
            self.log(n, v, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, stats = standard_loss(self, batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        for n, v in stats.items():
            n = "val_" + n
            prog_bar = True if n == "val_acc" else False
            self.log(n, v, on_step=False, on_epoch=True, prog_bar=prog_bar, sync_dist=True)

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


class PrunedModule(VisionModule):
    def __init__(
        self,
        network: nn.Module,
        prune_fn: Callable,
        test_labels: Sequence[str] = None,
        learning_rate: float = 0.1,
        augmix_loss_alpha: float = 12.,
        optimizer_fn=optim.Adam,
        scheduler_fn=None,
        monitor: str = None,
    ):
        super(PrunedModule, self).__init__(network=network,
                                           test_labels=test_labels,
                                           learning_rate=learning_rate,
                                           augmix_loss_alpha=augmix_loss_alpha,
                                           optimizer_fn=optimizer_fn,
                                           scheduler_fn=scheduler_fn,
                                           monitor=monitor)

        self.prune_fn = prune_fn 

    def setup(self, stage: str):
        if is_pruned(self.network):
            return

        info = self.prune_fn(self.network)
        self.print(f"\n {info.summary()}")


class HydraModule(VisionModule):
    def __init__(
        self,
        network: nn.Module,
        ratio: float,
        T_max: int = 200,
        T_prune_end: int = 30,
        test_labels: Sequence[str] = None,
        learning_rate: float = 0.1,
        augmix_loss_alpha: float = 12.,
        optimizer_fn=optim.Adam,
        scheduler_fn=None,
    ):
        # TODO: We currently only support training with schedulers.
        assert scheduler_fn is not None
        super(HydraModule, self).__init__(network=network,
                                          test_labels=test_labels,
                                          learning_rate=learning_rate,
                                          augmix_loss_alpha=augmix_loss_alpha,
                                          optimizer_fn=None,
                                          scheduler_fn=None,
                                          monitor=None)

        self.ratio = ratio
        self.T_prune_end = T_prune_end

    def setup(self, stage: str):
        if stage != "fit":
            return

        hydrate(self.network, self.ratio)
        

    def on_train_epoch_start(self):
        if self.trainer.current_epoch == self.T_prune_end:
            dehydrate(self.network)
           
            optimizers, lr_schedulers = self.configure_optimizers()
            self.trainer.optimizers = optimizers
            self.trainer.lr_schedulers = lr_schedulers

    def configure_optimizers(self, step_shift=None):
        pass 

