""" Collect network activations over a dataset."""
import logging
from functools import partial

import hydra
from hydra.utils import instantiate
from hydra.utils import call
import numpy as np
import torch.nn as nn
import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy

from pruneshift.scripts.utils import create_trainer
from pruneshift.scripts.utils import save_config
from pruneshift.scripts.utils import create_optim
from pruneshift.scripts.utils import partial_instantiate
from pruneshift.datamodules import datamodule


logger = logging.getLogger(__name__)


class ActivationCollector(pl.LightningModule):
    """Module that collects activations through a test pass"""

    def __init__(self, network: nn.Module, path: str, dataloader):
        super(ActivationCollector, self).__init__()
        self.network = network
        self.path = path
        self.activations = None
        self.dataloader = dataloader
        self.num_samples = len(self.dataloader.dataset)
        self.test_acc = Accuracy()

    def test_step(self, batch, batch_idx, dataset_idx=0):
        idx, x, y = batch
        activations = self.network(x)

        if self.activations is None:
            assert activations.ndim == 2
            num_classes = activations.shape[1]
            shape = (self.num_samples, num_classes)

            self.activations = np.memmap(
                self.path, dtype=np.float32, mode="w+", shape=shape
            )

        self.activations[idx.cpu().numpy()] = activations.detach().cpu().numpy()
        # self.activations[idx.cpu().numpy()] = torch.normal(0, 1, size=activations.shape)
        self.activations.flush()
        self.test_acc(y, torch.argmax(activations, -1))

    def test_epoch_end(self, outputs):
        self.log("dataset_acc", self.test_acc.compute())


@hydra.main(config_path="configs", config_name="collect.yaml")
def collect(cfg):
    """ Prunes a network and than finetunes it."""
    save_config(cfg)
    # Note that the trainer must be created first, for the seeding.
    trainer = create_trainer(cfg)
    network = call(cfg.network)
    data = datamodule(**cfg.datamodule)

    if cfg.train:
        data.setup("fit")
        loader = data.train_dataloader()
    else:
        data.setup("test")
        loader = data.test_dataloader()[0]


    module = ActivationCollector(network, cfg.save_path, loader)

    trainer.test(module, loader)


if __name__ == "__main__":
    collect()
