from bisect import bisect_right
from functools import partial
import logging
from typing import Sequence
from typing import Callable


import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.metrics import functional
import torch
import torch.nn as nn
from torch.nn.utils.prune import is_pruned
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F

from pruneshift.prune_hydra import hydrate
from pruneshift.prune_hydra import dehydrate
from pruneshift.losses import StandardLoss


logger = logging.getLogger(__name__)


class MultiStepWarmUpLr(LambdaLR):
    def __init__(
        self,
        optimizer,
        milestones: Sequence[int],
        warmup_end: int = 0,
        gamma: float = 0.1,
    ):
        """Learning rate scheduler that supports a warmup period with milestones."""

        def lr_schedule(epoch):
            if epoch < warmup_end:
                return (epoch + 1) / warmup_end
            return gamma ** bisect_right(milestones, epoch)

        super(MultiStepWarmUpLr, self).__init__(optimizer, lr_schedule)


class VisionModule(pl.LightningModule):
    """Module for training and testing computer vision models."""

    def __init__(
        self,
        network: nn.Module,
        test_labels: Sequence[str] = None,
        train_loss: nn.Module = None,
        optimizer_fn=optim.Adam,
        scheduler_fn=None,
    ):

        super(VisionModule, self).__init__()
        self.network = network
        self.train_loss = StandardLoss() if train_loss is None else train_loss
        self.val_loss = StandardLoss()
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn

        if test_labels is None:
            self.test_labels = ["acc"]
        else:
            self.test_labels = [f"acc_{l}" for l in test_labels]
        self.test_acc = nn.ModuleDict({l: Accuracy() for l in self.test_labels})

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        loss, stats = self.train_loss(self, batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        for n, v in stats.items():
            n = "train_" + n
            self.log(n, v, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, stats = self.val_loss(self, batch)
        acc = stats["acc"]

        self.log(
            "val_acc", acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
        )
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx, dataset_idx=0):
        x, y = batch
        self.test_acc[self.test_labels[dataset_idx]](y, torch.argmax(self(x), -1))

    def test_epoch_end(self, outputs):
        for label, metric in self.test_acc.items():
            self.log(f"test_{label}", metric.compute())

    def configure_optimizers(self):
        if self.optimizer_fn is None:
            return

        optimizer = self.optimizer_fn(self.parameters())

        if self.scheduler_fn is None:
            return optimizer

        scheduler = {"scheduler": self.scheduler_fn(optimizer)}

        # if self.monitor is not None:
        #     scheduler["monitor"] = self.monitor

        return [optimizer], [scheduler]


class PrunedModule(VisionModule):
    def __init__(
        self,
        network: nn.Module,
        prune_fn: Callable,
        test_labels: Sequence[str] = None,
        train_loss: nn.Module = None,
        optimizer_fn=optim.Adam,
        scheduler_fn=None,
    ):
        super(PrunedModule, self).__init__(
            network=network,
            test_labels=test_labels,
            train_loss=train_loss,
            optimizer_fn=optimizer_fn,
            scheduler_fn=scheduler_fn,
        )

        self.prune_fn = prune_fn

    def setup(self, stage: str):
        if is_pruned(self.network):
            return

        info = self.prune_fn(self.network)
        self.print(f"\n {info.summary()}")

    # def on_train_epoch_start(self):
    #     if self.trainer.current_epoch == self.T_prune_end:
    #         dehydrate(self.network)

    #         optimizers, lr_schedulers = self.configure_optimizers()
    #         self.trainer.optimizers = optimizers
    #         self.trainer.lr_schedulers = lr_schedulers

