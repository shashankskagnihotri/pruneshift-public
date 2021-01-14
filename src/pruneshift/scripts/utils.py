from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger



def save_config(cfg: DictConfig):
    with open(cfg.config_path, "w") as file:
        file.write(OmegaConf.to_yaml(cfg))


def create_trainer(cfg: DictConfig):
    """ Creates a `pl.Trainer` for our experiment setup."""
    path = Path(cfg.logdir)
    callbacks = []
    # Creating a tensorboard and csv logger.
    tb_logger = TensorBoardLogger(path, name="tensorboard", version="")
    csv_logger = CSVLogger(path, name=None, version="")
    logger = [tb_logger, csv_logger]
    callbacks.append(instantiate(cfg.model_checkpoint))

    if cfg.seed is None:
        deterministic = False
    else:
        pl.seed_everything(cfg.seed)
        deterministic = True

    return instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        deterministic=deterministic,
        weights_save_path=cfg.logdir,
    )
