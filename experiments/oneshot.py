import time
import os
from typing import Sequence
from copy import deepcopy

import hydra
from hydra import instantiate
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
from pruneshift import simple_prune


def list_checkpoints(chpt: ChptPath):
    pass



def load_network(chpt: ChptPath):
    print("Recreating network for {}.".format(chpt))
    start = time.time()
    network = topology(chpt.model_name, num_classes=10)
    state = torch.load(chpt.epoch_path)
    # Remove the structure of the network 
    conv_state = {}
    for name in state["state_dict"]:
        conv_state[name[8:]] = state["state_dict"][name]
    network.load_state_dict(conv_state)
    print("Finished recreating network in {:.1f}s.".format(time.time() - start))
    return network


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
        

def prune_shift(chpt, strategy, train_data, test_data, module_map):
    compression_ratios = [1, 2, 4, 8, 16]
    original_network = load_network(chpt)

    
    early_stop_callback = EarlyStopping(
        monitor="Validation/Accuracy")
    trainer = pl.Trainer(callbacks=[early_stop_callback],
                         logger=pl_loggers.CSVLogger(save_dir="/tmp/test_pruneshift"),
                         checkpoint_callback=False,
                         gpus=1)
    statistics = []

    for ratio in compression_ratios:
        network = deepcopy(original_network)
        # Prune the network
        simple_prune(network, strategy, module_map=module_map, amount=1 - 1 / ratio)
        module = TestModule(network, test_data.labels)
        trainer.fit(module, datamodule=train_data)
        trainer.test(module, datamodule=test_data)
        statistics.append({"network": chpt.model_name,
                           "dataset": chpt.datamodule_name,
                           "epoch": chpt.epoch + 1,
                           "ratio": ratio,
                           **module.test_statistics})
    return statistics


class TestModule(pl.LightningModule):
    """Module for training models."""
    def __init__(self,
                 network: nn.Module,
                 labels: Sequence[str]):
        super(TestModule, self).__init__()
        self.network = network
        self.labels = labels

    def forward(self, x):
        return self.network(x)

    def test_step(self, batch, batch_idx, dataset_idx):
        x, y = batch
        return y, torch.argmax(self(x), 1)

    def test_epoch_end(self, outputs):
        data = {}
        for label, dataset_outputs in zip(label, outputs):
            acc = Accuracy()
            for y, y_pred in dataset_outputs:
                acc(y, y_pred)
            # output dataloader_outputs are list of tuples.
            data[label] = acc.compute()
        self.log_dict(data)


def run(cfg):
    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)

    checkpoint_callback = ModelCheckpoint(save_top_k=-1, save_weights_only=True)
    data: pl.LightningDataModule = instantiate(cfg.DataModule)
    trainer: pl.Trainer = instantiate(cfg.Trainer, callbacks=[checkpoint_callback])
    network: nn.Module = instantiate(cfg.Network)
    module = instantiate(cfg.TrainingModule, network=network, labels=data.labels)
    trainer.test(module, datamodule=data)


if __name__ == "__main__":
    run()
