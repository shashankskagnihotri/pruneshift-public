# TODO: Implement a simple LightningModule where we can easily use pretrained models.
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.nn import functional as F
from pytorch_lightning.metrics.functional import accuracy


class DistrShift(pl.LightningModule):
    def __init__(self,
                 network: nn.Module,
                 lr: float = 0.001):
        super(DistrShift, self).__init__()
        self.network = network 
        self.lr = lr

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(torch.argmax(logits, 1), y)
        self.log("Training/Loss", loss)
        self.log("Training/Accuracy", acc)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        logits = self(x)
        acc = accuracy(torch.argmax(logits, 1), y)
        # print(f"Accuracy of dataset {dataloader_idx} is {acc}")
        log_str = "Test/{}/Accuracy".format(dataloader_idx)
        self.log(log_str, acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class MNISTModel(pl.LightningModule):
    def __init__(self, lr: float = 0.001):
        super(MNISTModel, self).__init__()
        self.l1 = nn.Linear(28 * 28, 500)
        self.l2 = nn.Linear(500, 250)
        self.l3 = nn.Linear(250, 10)
        self.lr = lr

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(torch.argmax(logits, 1), y)
        self.log("Training/Loss", loss)
        self.log("Training/Accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("Validation/Loss", loss)
        self.log("Validation/Accuracy", acc)
        return loss

    def test_step_end(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
