from functools import partial
from copy import deepcopy
import random
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
from .prune import simple_prune


logging.basicConfig(level=logging.DEBUG)


# Register the important sutff from gin-stuff.
GinTrainer = gin.external_configurable(pl.Trainer)
GinEarlyStopping = gin.external_configurable(EarlyStopping)
GinModelCheckpoint = gin.external_configurable(ModelCheckpoint)


@gin.configurable
def create_trainer(
    logdir: str,
    seed: int = None,
    early_stopping: bool = False,
    custom_checkpointing: bool = False,
):
    path = Path(logdir)
    callbacks = []
    # Creating a tensorboard and csv logger.
    tb_logger = TensorBoardLogger(path, name="tensorboard")
    csv_logger = CSVLogger(path, name="csv")
    logger = [tb_logger, csv_logger]
    # Introduce early_stopping
    if early_stopping:
        callbacks.append(GinEarlyStopping("val_acc"))
    # If wanted allow custom checkpointing.
    if custom_checkpointing:
        print("Using custom checkpointing!")
        callbacks.append(GinModelCheckpoint(monitor="val_acc"))

    if seed is None:
        deterministic = False
    else:
        print(f"Setting the seed to {seed}")
        pl.seed_everything(seed)
        deterministic = True

    return GinTrainer(logger=logger, deterministic=deterministic)


def create_hparams(bindings, **kw):
    for name, binding in bindings.items():
        kw[name] = gin.query_parameter(binding)
    return kw 


@click.group()
@click.argument("config-name", type=str)
@click.option("--gin-binding", "-b", type=str, multiple=True)
@click.option("--logdir", type=str, envvar="EXPERIMENT_PATH")
@click.option("--datadir", type=str, envvar="DATASET_PATH")
@click.pass_context
def cli(ctx, config_name, gin_binding, logdir, datadir):
    """Entry point for running pruneshift scripts."""
    config_path = Path(__file__).parent.parent.parent / "configs" / config_name
    ctx.ensure_object(dict)
    ctx.obj["logdir"] = logdir
    ctx.obj["datadir"] = datadir
    ctx.obj["config_name"] = config_name
    gin.parse_config_files_and_bindings([config_path], gin_binding)


@cli.command()
@click.argument("ratios", type=int, nargs=-1)
@click.pass_obj
def oneshot(obj, ratios):
    """ Does oneshot pruning."""

    original_network = network_topology(pretrained=True)
    data = datamodule(root=obj["datadir"])

    for ratio in ratios:
        trainer = create_trainer(obj["logdir"])
        network = deepcopy(original_network)
        simple_prune(network, amount=1 - 1 / ratio)
        hparams = create_hparams(
            {"network": "network_topology.name", "datamodule": "datamodule.name"},
            config_name=obj["config_name"],
            ratio=ratio,
        )
        module = VisionModule(network, hparams=hparams)
        trainer.fit(module, datamodule=data)
        trainer.test(module, datamodule=data)

@cli.command()
def dummy():
    pass

if __name__ == "__main__":
    cli({})
