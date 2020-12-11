import os

import hydra
from hydra.utils import instantiate
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning import loggers as pl_loggers
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

from pruneshift import datamodule


@hydra.main(config_name="config")
def run(cfg):
    print(cfg)
    print(cfg.Trainer.default_root_dir)
    print(os.getcwd())


if __name__ == "__main__":
    run()
