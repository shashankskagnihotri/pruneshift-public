from functools import partial
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor


def save_config(cfg: DictConfig):
    with open(cfg.path.config, "w") as file:
        file.write(OmegaConf.to_yaml(cfg))


def create_optim(cfg: DictConfig):
    """ Creates factory functions regarding optimization."""
    return {"optimizer_fn": partial(instantiate, cfg.optimizer),
            "scheduler_fn": partial(instantiate, cfg.scheduler)}


def create_trainer(cfg: DictConfig):
    """ Creates a `pl.Trainer` for our experiment setup."""
    path = Path(cfg.path.logdir)
    callbacks = []
    # Creating a tensorboard and csv logger.
    tb_logger = TensorBoardLogger(path, name="tensorboard", version="")
    csv_logger = CSVLogger(path, name=None, version="")
    logger = [tb_logger, csv_logger]
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
        logger=logger,
        callbacks=callbacks,
        deterministic=deterministic,
        weights_save_path=cfg.path.logdir,
    )

