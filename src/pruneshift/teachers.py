"""Provides teachers for pruneshift."""
import logging

import torch
import torch.nn as nn
import numpy as np
from pruneshift.networks import create_network
from pruneshift.networks import ImagenetSubsetWrapper


logger = logging.getLogger(__name__)


def create_teacher(
    group: str = None,
    name: str = None,
    num_classes: int = None,
    activations_path: str = None,
    ckpt_path: str = None,
    model_path: str = None,
    version: int = None,
    download: bool = False,
    imagenet_subset: bool = False,
):
    """Creates teachers."""
    logger.info("Creating teacher network...")

    if activations_path is not None:
        if imagenet_subset:
            logger.info(f"Adopting teacher database to {num_classes} classes.")
            network = DatabaseNetwork(activations_path, 1000)
            return ImagenetSubsetWrapper(network, num_classes)
        return DatabaseNetwork(activations_path, num_classes)

    network = create_network(
        group=group,
        name=name,
        num_classes=num_classes,
        ckpt_path=ckpt_path,
        model_path=model_path,
        version=version,
        download=download,
        imagenet_subset=imagenet_subset,
    )

    return Teacher(network)


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

