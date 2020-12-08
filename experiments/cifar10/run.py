# TODO: We need a way to create a resnet50 model,
#       it should support pre_trained and custom models.
# TODO: Train a resnet50 model.


import os

import hydra
from hydra.utils import instantiate
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

from pruneshift import datamodule


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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)



@hydra.main(config_name="config")
def run(cfg):
    data = instantiate(cfg.DataModule)
    trainer = instantiate(cfg.Trainer)
    model = models.resnet18(pretrained=True)
    module = TrainingModule(model)
    trainer.fit(module, datamodule=data)


if __name__ == "__main__":
    run()
