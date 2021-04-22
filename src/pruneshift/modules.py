from bisect import bisect_right
from functools import partial
import logging
from typing import Sequence
from typing import Callable
from typing import Optional
import re
import math

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from pruneshift.losses import StandardLoss
from pruneshift.utils import get_model_complexity_prune
from pruneshift.datamodules import ShiftDataModule


logger = logging.getLogger(__name__)
CORRUPTION_REGEX = re.compile(r"test_acc_[a-z_]+_[0-9]")


def multi_step_warm_up_lr(
    epoch: int,
    milestones: Sequence[int] = (20, 50, 70),
    warmup_end: int = 0,
    gamma: float = 0.1,
):
    """ Learning rate schedule that supports a warmup period with milestones."""
    if epoch < warmup_end:
        return (epoch + 1) / warmup_end
    return gamma ** bisect_right(milestones, epoch)


def cosine_lr(
    epoch: int,
    T_max: int,
    eta_min: float = 0,
):
    """ Cosine learning rate schedule."""
    if eta_min != 0:
        raise NotImplementedError
    return (1 + math.cos(math.pi * epoch / T_max)) / 2


def repeat_lr(epoch: int, scheduler_fn, cycle_length: int):
    epoch %= cycle_length
    return scheduler_fn(epoch)


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
        self.test_acc[self.test_labels[dataset_idx]](y, torch.argmax(self(x), -1))

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

        scheduler = {"scheduler": LambdaLR(optimizer, self.scheduler_fn)}

        # if self.monitor is not None:
        #     scheduler["monitor"] = self.monitor

        return [optimizer], [scheduler]


class PrunedModule(VisionModule):
    def __init__(
        self,
        network: nn.Module,
        prune_fn: Callable,
        datamodule,
        prune_at_start: bool = True,
        rewind_max_epochs: Optional[int] = None,
        prune_interval: Optional[int] = None,
        train_loss: nn.Module = None,
        optimizer_fn=optim.Adam,
        scheduler_fn=None,
    ):
        if rewind_max_epochs is not None and scheduler_fn is not None:
            scheduler_fn = partial(
                repeat_lr,
                scheduler_fn=scheduler_fn,
                cycle_length=rewind_max_epochs,
            )

        super(PrunedModule, self).__init__(
            network=network,
            datamodule=datamodule,
            train_loss=train_loss,
            optimizer_fn=optimizer_fn,
            scheduler_fn=scheduler_fn,
        )

        self.prune_fn = prune_fn
        self.prune_interval = prune_interval
        self.prune_at_start = prune_at_start

    def prune(self):
        info = self.prune_fn(self.network)
        self.print(f"\n {info.summary()}")
        self.print("The effective compression ratio is {}".format(info.network_comp()))
        self.print("The effective num of params is {}".format(info.network_size()))

    def on_train_epoch_start(self):
        # Prune at start if wanted.
        if self.trainer.current_epoch == 0 and self.prune_at_start:
            self.prune()

        # Prune when prune_interval is given.
        if self.prune_interval is not None and self.trainer.current_epoch > 0:
            if self.trainer.current_epoch % self.prune_interval == 0:
                self.prune()
