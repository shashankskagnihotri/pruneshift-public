""" Our scripts.

The experiments build the following structure:
    experiment/
        checkpoints/
        tensorboard/
        hparams.yaml
        metrics.csv
        config.gin

"""
import os
from typing import Type
from functools import partial
from copy import deepcopy
import random
import re
import logging
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger
import click
import gin
import gin.torch

from .datamodules import datamodule
from .topologies import network_topology
from .modules import VisionModule
from .prune import prune


logging.basicConfig(level=logging.DEBUG)


CKPT_FILENAME = "{epoch}-{val_acc:.2f}"
CKPT_FILENAME_REGEX = r"epoch=(?P<epoch>\d+)[-]val_acc=(?P<val_acc>\d+\.\d+)"

# Register the important sutff from gin-stuff.
GinTrainer = gin.external_configurable(pl.Trainer)
GinEarlyStopping = gin.external_configurable(EarlyStopping)
GinModelCheckpoint: Type[ModelCheckpoint] = gin.external_configurable(ModelCheckpoint)


@gin.configurable
def batch_gradient(module: pl.LightningModule, train_data: pl.LightningDataModule):
    """ Calculates a batch gradient and saves it into the module."""
    trainer = GinTrainer(limit_train_batches=5,
                         accumulate_grad_batches=5,
                         max_epochs=1,
                         checkpoint_callback=False,
                         logger=False)
    old_lr, module.lr = module.lr, 0.
    trainer.fit(module, datamodule=train_data)
    module.lr = old_lr 


@gin.configurable
def create_trainer(logdir: str, seed: int = None, early_stopping: bool = False):
    """ Creates a `pl.Trainer` for our experiment setup."""
    path = Path(logdir)
    callbacks = []
    # Creating a tensorboard and csv logger.
    tb_logger = TensorBoardLogger(path, name="tensorboard", version="")
    csv_logger = CSVLogger(path, name=None, version="")
    logger = [tb_logger, csv_logger]
    # Introduce early_stopping
    if early_stopping:
        callbacks.append(GinEarlyStopping("val_acc"))
    # If wanted allow custom checkpointing.
    callbacks.append(
        GinModelCheckpoint(
            dirpath=path/"checkpoints", filename=CKPT_FILENAME, save_weights_only=True
        )
    )

    if seed is None:
        deterministic = False
    else:
        pl.seed_everything(seed)
        deterministic = True

    return GinTrainer(
        logger=logger,
        callbacks=callbacks,
        deterministic=deterministic,
        weights_save_path=logdir,
    )


def create_hparams(bindings, **kw):
    for name, binding in bindings.items():
        kw[name] = gin.query_parameter(binding)
    return kw


@click.group()
@click.argument("config-file", type=str)
@click.option("--gin-binding", "-b", type=str, multiple=True)
@click.option("--logdir", type=str, default=None)
@click.option("--datadir", type=str, envvar="DATASET_PATH")
@click.pass_context
def cli(ctx, config_file, gin_binding, logdir, datadir):
    """Entry point for running pruneshift scripts."""
    ctx.ensure_object(dict)
    # If logdir was not set, we look for the envvar EXPERIMENT_PATH and
    # introduce a version dir mechanism.
    if logdir is None:
        logdir = Path(os.environ["EXPERIMENT_PATH"])
        versions = [int(str(vp.stem).split("_")[-1])
                    for vp in logdir.glob("version_*")]
        new_vers = max(versions) + 1 if versions else 0
        logdir = logdir/("version_" + str(new_vers))
    Path(logdir).mkdir(parents=True, exist_ok=True)

    ctx.obj["logdir"] = logdir
    ctx.obj["datadir"] = datadir

    config_path = Path(__file__).parent/"configs"/config_file
 
    gin.parse_config_files_and_bindings([config_path], gin_binding)

    with open(Path(logdir)/"config.gin", "w") as file:
        file.write(gin.config_str())


@cli.command()
@click.argument("pruning-method", type=str)
@click.argument("ratio", type=int)
@click.pass_obj
def oneshot(obj, pruning_method, ratio):
    """ Does oneshot pruning."""
    network = network_topology(pretrained=True)
    data = datamodule(root=obj["datadir"])
    trainer = create_trainer(obj["logdir"])
    hparams = create_hparams({"network": "network_topology.name", "datamodule": "datamodule.name"},
            ratio=ratio, pruning_method=pruning_method,
        )
    module = VisionModule(network, hparams=hparams)
    # print("Building gradients...")
    # batch_gradient(module, data)
    print("Pruning the network...")
    prune(network, pruning_method, ratio)
    print("Finetuning...")
    trainer.fit(module, datamodule=data)
    trainer.test(module, datamodule=data)


def _find_exp_files(path: str, name="checkpoint"):
    """ Searches a folder through for a checkpoint file."""
    pass

def _collect_ckpts(ckpt_dir: str):
    """Looks for checkpoint files in a dir and returns the paths with
    the corresponding episode number."""
    ckpt_paths = Path(ckpt_dir).glob("*.ckpt")

    if not ckpt_paths:
        raise FileNotFoundError(f"Could not find any checkpoints in {ckpt_dir}")

    ckpt_dict = {}
    for path in ckpt_paths:
        epoch = re.match(CKPT_FILENAME_REGEX, path.stem)["epoch"]
        ckpt_dict[epoch] = path
    return ckpt_dict



@cli.command()
@click.argument("paths", type=str, nargs=-1, required=True)
@click.pass_obj
def rewind(obj, paths):
    """ Does evaluate models for distr shift."""
    pass


if __name__ == "__main__":
    cli({})
