""" Collect network activations over a dataset."""
import logging
from functools import partial

import hydra
from hydra.utils import instantiate
from hydra.utils import call
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl

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

    @property
    def num_samples(self):
        return len(self.dataloader.dataset)

    def test_step(self, batch, batch_idx):
        activations = self.network(batch).cpu().numpy()

        if self.activations is None:
            assert activations.ndim == 2
            num_classes = activations.shape[1]
            shape = (self.num_samples, num_classes)

            self.activations = np.memmap(
                self.path, dtype=np.float32, mode="w+", shape=shape
            )

        self.activations[batch.idx.cpu().numpy()] = activations
        self.activations.flush()


@hydra.main(config_path="configs", config_name="collect.yaml")
def collect(cfg):
    """ Prunes a network and than finetunes it."""
    save_config(cfg)
    # Note that the trainer must be created first, for the seeding.
    trainer = create_trainer(cfg)
    network = call(cfg.network)
    data = datamodule(**cfg.datamodule)
    data.setup("fit")
    loader = data.train_dataloader()


    module = ActivationCollector(network, cfg.save_path, loader)

    trainer.test(module, loader)


if __name__ == "__main__":
    collect()
