# TODO: We need a way to create a resnet50 model.
# TODO: We need a way to create the Cifar10 data module.
#       I just want to have a function that returns me the Cifar10

import os

import hydra
from hydra.utils import instantiate
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
import torch
import torch.nn as nn
from torch.nn import functional as F


class TrainingModule(pl.LightningModule):
    """Module for training and finetuning the models."""
    def __init__(self,
                 network: nn.Module,
                 lr: float = 0.001):
        super(TrainingModule, self).__init__()
        self.network = network
        self.lr = lr

    def forward(self, x):
        return self.network(x)

    def _predict(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(torch.argmax(logits, 1), y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._predict(batch)
        self.log("Training/Loss", loss)
        self.log("Training/Accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._predict(batch)
        self.log("Validation/Loss", loss)
        self.log("Validation/Accuracy", acc)


@hydra.main(config_name="config")
def run(cfg):
    pass


if __name__ == "__main__":
    run()
