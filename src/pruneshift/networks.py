"""Provides network topologies for CIFAR and ImageNet datasets.

Note, that we did not find any pretrained models for Cifar10 and Cifar100:

Thus, they must be manually trained and provided in a checkpoint directory.
The checkpoints must have the following form:
    {network_id}.{version}
"""
from functools import partial
import re
from pathlib import Path
from typing import Callable
from typing import Optional
import logging

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as imagenet_models
import torchvision.models.mnasnet as mnasnet
import timm
import pytorch_lightning as pl

from .utils import safe_ckpt_load
import cifar10_models as cifar_models
import pytorch_resnet_cifar10.resnet as cifar_resnet
import models as models


logger = logging.getLogger(__name__)

# Bugfix otherwise eval mode does not work for mnasnet architectures.
mnasnet._BN_MOMENTUM = 0.1

# NETWORK_REGEX = re.compile(
#     r"(?P<group>[a-zA-Z]+)(?P<num_classes>[0-9]+)_(?P<name>[a-zA-Z0-9_]+)"
# )


def protect_classifier(name: str, network: nn.Module):
    """ Defines which layers are protected. """
    if name[:6] == "resnet":
        if hasattr(network, "linear"):
            network.linear.is_protected = True
        else:
            network.fc.is_protected = True
        network.conv1.is_protected = True
    elif name[:3] == "vgg":
        network.classifier[-1].is_protected = True
    elif name[:8] == "densenet":
        network.classifier.is_protected = True
    elif name[:7] == "mnasnet":
        network.classifier[-1].is_protected = True
    else:
        raise NotImplementedError



def create_network(
    group: str,
    name: str,
    num_classes: int,
    ckpt_path: str = None,
    model_path: str = None,
    version: int = None,
    protect_classifier_fn: Optional[Callable] = None,
    download: bool = False,
    imagenet_subset: Optional[bool] = None,
    **kwargs,
):
    """A function creating networks.

    Args:
        group: The dataset group eg. imagenet, cifar.
        name: The name of model architecture.
        num_classes: The number of classes.
        ckpt_path: Direct path o a checkpoint of a network or activation
            database.
        model_path: Directory to find checkpoints in.
        version: Version of the checkpoint.
        protect_classifier_fn: Function that marks classifiers for
            protectionfrom pruning.
        download: Whether to download a model from torchvision. Only
            possible for imagenet1000.
        imagenet_subset: Whether the model should adjust activations for
            different ordered imagenet subsets. Note that when using a
            downloaded model, with fewer classes, this is done automati-
            cally.

    Returns:
        The desired network.
    """
    logger.info(f"Creating network {name} for {group} with {num_classes} classes.")

    # 2. Find the right factory function for the network.

    # Resolve the path
    path = _resolve_path(group, name, num_classes, ckpt_path, model_path, version)

    if group == "cifar":
        if name[:6] == "resnet":
            network_fn = getattr(models, name)
        else:
            network_fn = getattr(models, name)
    elif group == "imagenet":
        if hasattr(imagenet_models, name):
            network_fn = getattr(imagenet_models, name)
        else:
            # Look at pytorch-image-models
            network_fn = partial(timm.create_model, model_name=name)
    else:
        raise ValueError(f"Unknown group {group}.")

    # 3. Create the network and potentially load a checkpoint.
    if download:
        kwargs["pretrained"] = True

    subset_wrap = imagenet_subset
    create_num_classes = num_classes

    if (
        download
        and group == "imagenet"
        and num_classes != 1000
        and imagenet_subset is None
        or imagenet_subset
    ):
        subset_wrap = True
        create_num_classes = 1000

    network = network_fn(num_classes=create_num_classes, **kwargs)

    if path is not None and group != "dataset":
        safe_ckpt_load(network, path)

    # 4. Adjust network with addtional wrappers, hooks and etc.
    # Protect classifier layers from pruning.
    if protect_classifier_fn is not None:
        protect_classifier_fn(name, network)

    # When downloaded we change the label scheme from the activations..
    if subset_wrap:
        logger.info(f"Adopting network from imagenet1000 to imagenet{num_classes}.")
        network = ImagenetSubsetWrapper(network, num_classes)

    # Add a reset hook for lottery style pruning.
    add_reset_hook(
        network,
        partial(create_network, group=group, name=name, num_classes=num_classes),
        version,
    )

    return network


class ImagenetSubsetWrapper(nn.Module):
    """Changes predictions for models trained on imagenet to a subset."""

    def __init__(
        self,
        network: nn.Module,
        num_classes: int,
        root: str = None,
        super_root: str = None,
    ):
        super(ImagenetSubsetWrapper, self).__init__()
        # TODO: Finally make this parameterizable.
        # root = f"/misc/scratchSSD2/datasets/ILSVRC2012-{num_classes}/train"
        # super_root = "/misc/scratchSSD2/datasets/ILSVRC2012/train"
        super_root = "/data/datasets/ILSVRC2012/train"
        root = f"/work/dlclarge2/hoffmaja-pruneshift/datasets/ILSVRC2012-{num_classes}/train"
        self.num_classes = num_classes
        self.root = root
        self.super_root = super_root
        self.network = network
        self.perm = self.calculate_permutation()

    def calculate_permutation(self):
        class_dirs = sorted([p.stem for p in Path(self.root).iterdir()])
        super_class_dirs = sorted([p.stem for p in Path(self.super_root).iterdir()])

        # The first part contains the class indices of the subset.
        perm = [super_class_dirs.index(cd) for cd in class_dirs]
        # The second part contains the class indices not in the subset.
        perm.extend(set(range(len(super_class_dirs))) - set(perm))
        return torch.tensor(perm)

    def forward(self, *args):
        return self.network(*args)[..., self.perm[: self.num_classes]]


def add_reset_hook(network: nn.Module, network_factory: Callable, version=0):
    """ Adds a reset option to a network, usable for lottery ticket stuff."""

    def reset_hook():
        logger.info(f"Setting the seed to {version} to recreate original network.")
        pl.seed_everything(version)
        orig_state_dict = network_factory().state_dict()
        network.load_state_dict(orig_state_dict, strict=False)

    network.__reset_hook = reset_hook


def _resolve_path(
    group: str,
    name: str,
    num_classes: int,
    ckpt_path: str,
    model_path: str,
    version: int,
):
    if ckpt_path is not None:
        return Path(ckpt_path)
    elif model_path is not None:
        version = 0 if version is None else version
        return Path(model_path) / f"{group}{num_classes}_{name}.{version}"

    return None
