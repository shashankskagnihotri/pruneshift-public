# TODO: Add support for multiple optimizer and shedulers.
from typing import Sequence

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.metrics import functional
import torch
import torch.nn as nn
from torch.nn import functional as F


class VisionModule(pl.LightningModule):
    """Module for training computer vision models."""

    def __init__(self,
                 network: nn.Module,
                 test_labels: Sequence[str] = ("", ),
                 val_labels: Sequence[str] = ("", ),
                 lr: float = 0.0001):
        super(VisionModule, self).__init__()
        self.network = network
        self.val_labels = [f"acc_{l}" for l in val_labels]
        self.test_labels = [f"acc_{l}" for l in test_labels]
        self.val_acc = nn.ModuleDict({l: Accuracy() for l in self.val_labels})
        self.test_acc = nn.ModuleDict({l: Accuracy() for l in self.test_labels})
        self.lr = lr
        self.test_statistics = None

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = functional.accuracy(torch.argmax(logits, 1), y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss

    def _log_plenty(self, module_dict, mode):
        for label, metric in module_dict.items():
            self.log(f"{mode}_{label}", metric.compute())

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        x, y = batch
        self.val_acc[self.val_labels[dataset_idx]](y, torch.argmax(self(x), -1))

    def validation_epoch_end(self, outputs):
        self._log_plenty(self.val_acc, "val")

    def test_step(self, batch, batch_idx, dataset_idx):
        x, y = batch
        self.test_acc[self.test_labels[dataset_idx]](y, torch.argmax(self(x), -1))

    def test_epoch_end(self, outputs):
        self._log_plenty(self.test_acc, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

