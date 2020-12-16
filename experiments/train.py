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
    def __init__(self,
                 network: nn.Module,
                 lr: float = 0.001):
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
        return torch.optim.Adam(self.parameters(), lr=self.lr)


@click.command()
@click.argument("root", type=str, envvar="EXPERIMENT_PATH")
@click.option("--network", type=str, default="resnet50")
@click.option("--train-data", type=str, default="CIFAR10")
@click.option("--num-epochs", type=int, default=300)
@click.option("--learning-rate", type=float, default=0.0003)
def run(root, network, train_data, num_epochs, learning_rate):
    # if seed is not None:
    #     pl.seed_everything(seed)
    checkpoint_callback = ModelCheckpoint(save_top_k=-1, save_weights_only=True)
    trainer = pl.Trainer(default_root_dir=root,
                         benchmark=True,                          
                         callbacks=[checkpoint_callback])
    network = pruneshift.topology(network)
    module = TrainingModule(network=network, learning_rate=0.0003)
    datamodule = pruneshift.datamodule(train_data)
    trainer.fit(module, datamodule=datamodule)

if __name__ == "__main__":
    run()

