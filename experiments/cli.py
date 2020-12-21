# TODO: ADD A TRAINER FACTORY AND A DATAMODULE FACTORY. 20 MIN
# TODO: INTEGRATE SETUP TOOLS. 20 MIN
# TODO: Move modules into the main module, maybe even join the train module,
#       with the test module. 10 min
from functools import partial
import random

import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger
import click


from pruneshift import datamodule


def trainer_factory(epochs, gpus, logdir, resultdir, **out):
    # Tensorboard logger saves at the same location as the weights are saved.
    # We want to use tensorboard only for monitoring progress.
    tb_logger = TensorBoardLogger(logdir)
    # CSVLogger logs into the resultdir which is part of the repository and is
    # used to visualize results.
    csv_logger = CSVLogger(resultdir)

    logger = [tb_logger, csv_logger]

    def create_fn(**kw):
        pl.Trainer(logger=logger, benchmark=True, gpus=gpus, max_epochs=epochs,
                **kw)

    return create_fn


@click.group()
@click.option("--epochs", type=int, default=200)
@click.option("--gpus", type=int, default=1)
@click.option("--logdir", type=str, envvar="EXPERIMENT_PATH")
@click.option("--datadir", type=str, envvar="DATASET_PATH")
@click.option("--resultdir", type=str, default="./results")
@click.option("--batch-size", type=int, default=128)
@click.option("--seed", type=int, default=-1)
@click.option("--num-workers", type=int, default=5)
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

