import random

import click
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

import pruneshift


class TrainingModule(pl.LightningModule):
    """Module for training models."""

    def __init__(self, network: nn.Module, lr: float = 0.001):
        super(TrainingModule, self).__init__()
        self.network = network
        self.lr = lr

    def forward(self, x):
        return self.network(x)

    def _predict(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(torch.argmax(logits, 1), y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._predict(batch)
        self.log("Training/Loss", loss)
        self.log("Training/Accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._predict(batch)
        self.log("Validation/Loss", loss)
        self.log("Validation/Accuracy", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 150])
        return {"optimizer": optimizer, "lr_scheduler": scheduler,
                "monitor": "Validation/Loss"}


@click.command()
@click.argument("logpath", type=str)
@click.option("--datapath", type=str, envvar="DATASET_PATH")
@click.option("--network", type=str, default="resnet50")
@click.option("--train-data", type=str, default="CIFAR10")
@click.option("--gpus", type=int, default=1)
@click.option("--epochs", type=int, default=200)
@click.option("--learning-rate", type=float, default=0.1)
@click.option("--batch-size", type=int, default=128)
@click.option("--save-every", type=int, default=10)
def run(logpath, datapath, network, train_data, gpus, epochs, learning_rate, batch_size, save_every):
    seed = random.randint(0, 10000)
    pl.seed_everything(random.randint(0, 10000))
    print(f"The seed of this run is {seed}.")
    checkpoint_callback = ModelCheckpoint(period= save_every)
    trainer = pl.Trainer(
        epochs=epochs,
        default_root_dir=logpath,
        benchmark=True,
        callbacks=[checkpoint_callback],
        gpus=gpus
    )
    network = pruneshift.topology(network)
    module = TrainingModule(network=network, lr=learning_rate)
    datamodule = pruneshift.datamodule(train_data, datapath, batch_size=batch_size)
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    run()
