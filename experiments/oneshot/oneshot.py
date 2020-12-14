import os
from typing import Sequence

import hydra
from hydra.utils import instantiate
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy 
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

from pruneshift import datamodule


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


@hydra.main(config_path="../config", config_name="main")
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
