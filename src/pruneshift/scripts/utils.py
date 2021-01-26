from functools import partial
from pathlib import Path
import logging

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor

logger = logging.getLogger(__name__)


def save_config(cfg: DictConfig):
    with open(Path.cwd()/"configs.yaml", "w") as file:
        file.write(OmegaConf.to_yaml(cfg))


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
    callbacks.append(instantiate(cfg.checkpoint))
    # We also want to log the learning rate.
    callbacks.append(LearningRateMonitor("epoch"))

    if cfg.seed.seed is None:
        deterministic = False
    else:
        pl.seed_everything(cfg.seed.seed)
        deterministic = True

    return instantiate(
        cfg.trainer,
        logger=loggers,
        callbacks=callbacks,
        deterministic=deterministic,
        weights_save_path=cfg.path.logdir,
    )
