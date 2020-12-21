# TODO: Save the stats into a csv file, after each test run.
import time
import os
import pathlib
from typing import Sequence, NamedTuple
from copy import deepcopy

import click
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy 
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

from pruneshift import datamodule
from pruneshift import topology
from pruneshift.prune.utils import simple_prune
from pruneshift import strategy
from .cli import cli


class TestModule(pl.LightningModule):
    """Module for training models."""
    def __init__(self,
                 network: nn.Module,
                 labels: Sequence[str],
                 lr: float = 0.0001):
        super(TestModule, self).__init__()
        self.network = network
        self.labels = labels
        self.accuracy = nn.ModuleDict({l: Accuracy() for l in labels})
        self.lr = lr
        self.test_statistics = None

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

    def forward(self, x):
        return self.network(x)
        
    def test_step(self, batch, batch_idx, dataset_idx):
        x, y = batch
        self.accuracy[self.labels[dataset_idx]](y, torch.argmax(self(x), -1))
        
    def test_epoch_end(self, output):
        self.test_statistics = {l: a.compute().item() for l, a in self.accuracy.items()}
        

@cli.command()
@click.argument("ratios", nargs=-1, default=(1, 2, 4, 8, 16, 32))
@click.option("--network", type=str, default="cifar10_resnet50")
@click.option("--strategy", type=str, default="l1")
@click.option("--train-data", type=str, default="cifar10")
@click.option("--test-data", type=str, default="cifar10_corrupted")
@click.option("--learning-rate", type=float, default=0.1)
@click.option("--save-every", type=int, default=10)
@click.pass_obj
def oneshot(obj, **kw):
    """ Does oneshot pruning."""

    original_network = topology(kw["network"], pretrained=True)
    trainer = obj["trainer_factory"]()
    train_data = obj["datamodule_factory"](kw["train_data"])
    test_data = obj["test_module"](kw["test_data"])

    for ratio in kw["ratio"]:
        network = deepcopy(original_network)
        simple_prune(network, strategy, amount=1 - 1 / ratio)
        module = TestModule(network, test_data.labels)
        trainer.fit(module, datamodule=train_data)
        trainer.test(module, datamodule=test_data)

