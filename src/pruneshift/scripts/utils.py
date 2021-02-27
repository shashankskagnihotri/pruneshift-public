from functools import partial
from pathlib import Path
import logging
import warnings
from typing import Dict

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only

from pruneshift.teachers import create_teacher

logger = logging.getLogger(__name__)


def save_config(cfg: DictConfig):
    with open(Path.cwd()/"configs.yaml", "w") as file:
        file.write(OmegaConf.to_yaml(cfg))


def check_batch_size(cfg: DictConfig, trainer: pl.Trainer):
    if not cfg.trainer.accelerator == "ddp":
        logger.info("Batch size seems fine!")
        return

    old_batch_size = cfg.datamodule.batch_size
    new_batch_size = old_batch_size // trainer.num_gpus 
    logger.info(f"Change the batch size from {old_batch_size} to {new_batch_size}.") 
    cfg.datamodule.batch_size = new_batch_size


def partial_instantiate(cfg):
    return partial(instantiate, cfg)


def create_optim(cfg: DictConfig):
    """ Creates factory functions regarding optimization."""
    return {"optimizer_fn": partial(instantiate, cfg.optimizer),
            "scheduler_fn": partial(instantiate, cfg.scheduler)}


def create_trainer(cfg: DictConfig):
    """ Creates a `pl.Trainer` for our experiment setup."""
    path = Path.cwd()
    logger.info(f"Save everything into {path}")
    callbacks = []
    # Creating a tensorboard and csv logger.
    tb_logger = TensorBoardLogger(path, name="tensorboard", version="")
    csv_logger = CSVLogger(path, name=None, version="")
    loggers = [tb_logger, csv_logger]

    if "checkpoint" in cfg:
        callbacks.append(instantiate(cfg.checkpoint, dirpath=path/"checkpoint"))

    # We also want to log the learning rate.
    callbacks.append(LearningRateMonitor("epoch"))

    # Add a filter for the filterwarnings.
    warnings.filterwarnings("ignore", module="pytorch_lightning.utilities.distributed", lineno=45)

    deterministic = True

    pl.seed_everything(cfg.seed)

    trainer = instantiate(
        cfg.trainer,
        logger=loggers,
        callbacks=callbacks,
        deterministic=deterministic,
        weights_save_path=cfg.path.logdir,
    )

    # Check if we need to correct the batch_size.
    check_batch_size(cfg, trainer)

    return trainer


def create_loss(cfg: DictConfig, network, datamodule):
    if "teacher" in cfg:
        teacher = create_teacher(**cfg.teacher)
        return instantiate(cfg.loss, network=network, teacher=teacher, datamodule=datamodule)

    return instantiate(cfg.loss, network=network, datamodule=datamodule)

@rank_zero_only
def print_test_results(results):
    logger.info("Test Results:")
    results = results[0]
    for name, value in results.items():
        print(f"\t{name}:\t\t{value}")

