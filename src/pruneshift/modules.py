from bisect import bisect_right
from functools import partial
import logging
from typing import Sequence
from typing import Callable
import re


import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.metrics import functional
from pytorch_lightning.utilities import rank_zero_only
import torch
import torch.nn as nn
from torch.nn.utils.prune import is_pruned
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F

from pruneshift.losses import StandardLoss
from pruneshift.utils import get_model_complexity_prune
from pruneshift.datamodules import ShiftDataModule


logger = logging.getLogger(__name__)
CORRUPTION_REGEX = re.compile(r"test_acc_[a-z_]+_[0-9]")


class MultiStepWarmUpLr(LambdaLR):
    def __init__(
        self,
        optimizer,
        milestones: Sequence[int],
        warmup_end: int = 0,
        gamma: float = 0.1,
    ):
        """Learning rate scheduler that supports a warmup period with milestones."""

        def lr_schedule(epoch):
            if epoch < warmup_end:
                return (epoch + 1) / warmup_end
            return gamma ** bisect_right(milestones, epoch)

        super(MultiStepWarmUpLr, self).__init__(optimizer, lr_schedule)


class VisionModule(pl.LightningModule):
    """Module for training and testing computer vision models."""

    def __init__(
        self,
        network: nn.Module,
        datamodule: ShiftDataModule,
        train_loss: nn.Module = None,
        optimizer_fn=optim.Adam,
        scheduler_fn=None,
    ):

        super(VisionModule, self).__init__()
        self.network = network
        self.train_loss = StandardLoss(network) if train_loss is None else train_loss
        self.datamodule = datamodule 
        self.val_loss = StandardLoss(network)
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn

        # Prepare the metrics.
        self.test_labels = [f"acc_{l}" for l in self.datamodule.labels]
        self.test_acc = nn.ModuleDict({l: Accuracy() for l in self.test_labels})

    def on_train_start(self):
        # This introduces a bug!!!
        # self.model_stats()
        return

    # @rank_zero_only
    def model_stats(self):
        test_res = self.datamodule.test_resolution
        num_flops, _ = get_model_complexity_prune(self.network, test_res)
        self.print(f"\n The number of flops is {num_flops}")

    def forward(self, x):
        return self.network(x)

    def state_dict(self, *args, **kwargs):
        """ Make sure only the trained network is used for saving."""
        # This is a dirty quickfix, but otherwise we can not switch
        # efficiently, between different losses.
        # Note that this also breaks compatibility with newer lightning
        # versions.
        return self.network.state_dict(*args, **kwargs)

    def training_step(self, batch, batch_idx):        
        loss, stats = self.train_loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        for n, v in stats.items():
            n = "train_" + n
            self.log(n, v, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, stats = self.val_loss(batch)
        acc = stats["acc"]

        self.log(
            "val_acc", acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
        )
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx, dataset_idx=0):
        _, x, y = batch
        #print(x)
        self.test_acc[self.test_labels[dataset_idx]](
            y, torch.argmax(self(x), -1)
        )

    def test_epoch_end(self, outputs):
        values = {}

        for label, metric in self.test_acc.items():
            values[f"test_{label}"] = metric.compute()

        self.log_dict(values)

        if len(values) <= 1:
            return

        # Caluclate the mean corruption error, we need to filter out all non corruption accuracies.
        non_clean = [acc for l, acc in values.items() if CORRUPTION_REGEX.match(l)]
        self.log("test_mCE", 1 - torch.tensor(non_clean).mean())

    def configure_optimizers(self):
        if self.optimizer_fn is None:
            return

        optimizer = self.optimizer_fn(self.parameters())

        if self.scheduler_fn is None:
            return optimizer

        scheduler = {"scheduler": self.scheduler_fn(optimizer)}

        # if self.monitor is not None:
        #     scheduler["monitor"] = self.monitor

        return [optimizer], [scheduler]


class PrunedModule(VisionModule):
    def __init__(
        self,
        network: nn.Module,
        prune_fn: Callable,
        datamodule,
        train_loss: nn.Module = None,
        optimizer_fn=optim.Adam,
        scheduler_fn=None,
    ):
        super(PrunedModule, self).__init__(
            network=network,
            datamodule=datamodule,
            train_loss=train_loss,
            optimizer_fn=optimizer_fn,
            scheduler_fn=scheduler_fn,
        )

        self.prune_fn = prune_fn

    def setup(self, stage: str):
        # Prune only at the beginning of the training phase.
        if stage == "fit":
            info = self.prune_fn(self.network)
            self.print(f"\n {info.summary()}")
            self.print("The effective compression ratio is {}".format(info.network_comp()))
            self.print("The effective num of params is {}".format(info.network_size()))


    # def on_train_epoch_start(self):
    #     if self.trainer.current_epoch == self.T_prune_end:
    #         dehydrate(self.network)

    #         optimizers, lr_schedulers = self.configure_optimizers()
    #         self.trainer.optimizers = optimizers
    #         self.trainer.lr_schedulers = lr_schedulers

