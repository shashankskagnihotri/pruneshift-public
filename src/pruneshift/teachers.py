"""Provides teachers for pruneshift."""
import logging

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class Teacher(nn.Module):
    def __init__(self, network: nn.Module = None):
        super(Teacher, self).__init__()
        self.network = network

    def forward(self, idx, x):
        return self.network(x)


class DatabaseNetwork(Teacher):
    def __init__(self, path: str, num_classes: int):
        super(DatabaseNetwork, self).__init__()
        self.activations = np.memmap(path, mode="r", dtype=np.float32)
        self.activations = np.reshape(self.activations, (-1, num_classes))

    def forward(self, idx, x):
        activations = torch.tensor(np.array(self.activations[idx.cpu().numpy()]))
        return activations.to(idx.device)
