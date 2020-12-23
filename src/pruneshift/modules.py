# TODO: Add support for multiple optimizer and shedulers.
from typing import Sequence
from typing import Dict

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.metrics import functional
import torch
import torch.nn as nn
from torch.nn import functional as F
import gin


@gin.configurable
class VisionModule(pl.LightningModule):
    """Module for training computer vision models."""

    def __init__(self,
                 network: nn.Module,
                 test_labels: Sequence[str] = None,
                 lr: float = 0.0001,
                 hparams: Dict = None):
        super(VisionModule, self).__init__()
        self.network = network
        self.lr = lr
        self.hparams = hparams

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
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._predict(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx, dataset_idx=0):
        x, y = batch
        self.test_acc[self.test_labels[dataset_idx]](y, torch.argmax(self(x), -1))

    def test_epoch_end(self, outputs):
        for label, metric in self.test_acc.items():
            self.log(f"test_{label}", metric.compute())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

