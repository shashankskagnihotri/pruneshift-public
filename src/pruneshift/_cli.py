from functools import partial
from copy import deepcopy
import random
import logging

import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger
import click

from . import datamodule
from . import network_topology
from . import prune_strategy
from .prune import simple_prune
from .modules import VisionModule


logging.basicConfig(level=logging.DEBUG)


def trainer_factory(epochs, gpus, logdir, **out):
    def create_fn(experiment_name, **kw):
        # Tensorboard logger saves at the same location as the weights are saved.
        # We want to use tensorboard only for monitoring progress.
        tb_logger = TensorBoardLogger(logdir, name=experiment_name)
        # CSVLogger logs into the resultdir which is part of the repository and is
        # used to visualize results.
        csv_logger = CSVLogger(logdir, name=experiment_name)
        logger = [tb_logger, csv_logger]

        return pl.Trainer(logger=logger,
                          benchmark=True,
                          gpus=gpus,
                          max_epochs=epochs,
                          default_root_dir=logdir,
                          **kw)
    return create_fn


@click.group()
@click.option("--epochs", type=int, default=200)
@click.option("--gpus", type=int, default=1)
@click.option("--logdir", type=str, envvar="EXPERIMENT_PATH")
@click.option("--datadir", type=str, envvar="DATASET_PATH")
@click.option("--batch-size", type=int, default=128)
@click.option("--seed", type=int, default=-1)
@click.option("--num-workers", type=int, default=5)
@click.pass_context
def cli(ctx, **kw):
    """Entry point for running pruneshift scripts."""
    ctx.ensure_object(dict)

    # Seed everything.
    if kw["seed"] == -1:
        kw["seed"] = random.randint(0, 10000)
    pl.seed_everything(kw["seed"])

    ctx.obj["trainer_factory"] = trainer_factory(**kw)
    ctx.obj["datamodule_factory"] = partial(datamodule, root=kw["datadir"], batch_size=kw["batch_size"], num_workers=kw["num_workers"])
    ctx.obj["hparams_dict"] = {k: kw[k] for k in ["epochs", "batch_size"]}


@cli.command()
@click.argument("ratios", type=int, nargs=-1)
@click.option("--network", type=str, default="cifar10_resnet50")
@click.option("--strategy", type=str, default="l1")
@click.option("--train-data", type=str, default="cifar10")
@click.option("--test-data", type=str, default="cifar10_corrupted")
@click.option("--learning-rate", type=float, default=0.1)
@click.option("--save-every", type=int, default=10)
@click.pass_obj
def oneshot(obj, **kw):
    """ Does oneshot pruning."""

    original_network = network_topology(kw["network"], pretrained=True)
    trainer = obj["trainer_factory"]("oneshot")
    train_data = obj["datamodule_factory"](kw["train_data"])
    test_data = obj["datamodule_factory"](kw["test_data"])
    print(test_data.labels)

    for ratio in kw["ratios"]:
        network = deepcopy(original_network)
        simple_prune(network, prune_strategy(kw["strategy"]), amount=1 - 1 / ratio)
        module = VisionModule(network, test_data.labels)
        trainer.fit(module, datamodule=train_data)
        trainer.test(module, datamodule=test_data)


@cli.command()
@click.pass_obj
def rewind(obj, **kw):
    """ Introduces rewind."""
    print("Hey ho")
