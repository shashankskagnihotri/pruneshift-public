from bisect import bisect_right
from functools import partial
from typing import Sequence
from typing import Optional
from typing import Type

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.metrics import functional
import torch
import torch.nn as nn
import torch.optim as optim
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


class VisionModule(pl.LightningModule):
    """Module for training and testing computer vision models."""

    def __init__(
        self,
        network: nn.Module,
        test_labels: Sequence[str] = None,
        learning_rate: float = 0.0001,
        optimizer_fn=optim.Adam,
        scheduler_fn=None,
        monitor: str = None,
    ):

        super(VisionModule, self).__init__()
        self.network = network
        self.learning_rate = learning_rate
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

    def _predict(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = functional.accuracy(torch.argmax(logits, 1), y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._predict(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._predict(batch)
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

        config = {}
        config["optimizer"] = self.optimizer_fn(
            self.parameters(), lr=self.learning_rate
        )

        if self.scheduler_fn is not None:
            config["scheduler"] = self.scheduler_fn(optimizer=config["optimizer"])

        if self.monitor is not None:
            config["monitor"] = self.monitor

        return config
