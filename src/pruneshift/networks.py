""" Bundles different kind of network create functions."""
from functools import partial
import re
from pathlib import Path
from typing import Callable, Optional
from typing import Optional
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagenet_models
import torchvision.models.mnasnet as mnasnet
import timm
import pytorch_lightning as pl
from collections import OrderedDict
from pruneshift.teachers import logger, DatabaseNetwork, Teacher
from pruneshift.utils import ImagenetSubsetWrapper

from .utils import safe_ckpt_load
from .utils import ImagenetSubsetWrapper
from .network_markers import protect_classifier
import cifar10_models as cifar_models
import scalable_resnet
import models as models


logger = logging.getLogger(__name__)

# Bugfix otherwise eval mode does not work for mnasnet architectures.
mnasnet._BN_MOMENTUM = 0.1


def create_network(
    group: str,
    name: str,
    num_classes: int,
    ckpt_path: str = None,
    model_path: str = None,
    version: int = None,
    download: bool = False,
    imagenet_subset_path: str = None,
    imagenet_path: str = None,
    scaling_factor: float = 1.0,
    supConLoss: bool = False,
    classifying:bool = False,
    testing : bool = False,
    feat_dim:int =128,
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
        download: Whether to download a model from torchvision. Only
            possible for imagenet1000.
        imagenet_subset_path: When given we wrap networks that were
            trained on imagenet1000, to match the correct logits for
            the subset. Should point to the training folder.
        imagenet_path: The path to imagenet1000, must be passed with
            imagenet_subset_path.
        scaling_factor: float

    Returns:
        The desired network.
    """
    assert (imagenet_path is None) == (imagenet_subset_path is None)

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
            if scaling_factor != 1.0:
                logger.info(f"Scaling the network {scaling_factor} times.")
                network_fn = _scalable_imagenet_models(name, scaling_factor)
            else:
                network_fn = getattr(imagenet_models, name)
        else:
            # Look at pytorch-image-models
            network_fn = partial(timm.create_model, model_name=name)
    else:
        raise ValueError(f"Unknown group {group}.")

    # 3. Create the network and potentially load a checkpoint.
    if download:
        kwargs["pretrained"] = True

    create_num_classes = num_classes
    subset_wrap = False

    if imagenet_path is not None and num_classes != 1000:
        subset_wrap = True
        create_num_classes = 1000

    network = network_fn(num_classes=create_num_classes, **kwargs)


    # Protect classifier layers from pruning must come before
    # ckpt loading for hydra.
    protect_classifier(network)


    # Add the projection head if using SupConLoss:
    if supConLoss:
        if name[:6]=="resnet":
            dim_in=network.fc.in_features
            classifier=network.fc
            network.fc=Identity()
        else:
            dim_in=network.classifier[-1].in_features
            classifier=network.classifier
            network.classifier=Identity()
        normalize=Normalize()
        layers = OrderedDict([("features",network),("flatten", nn.Flatten()), ("contrast", nn.Linear(dim_in, dim_in)), ("Relu",nn.ReLU(inplace=True)),("projection", nn.Linear(dim_in, feat_dim)),("normalize", normalize),])
        network = torch.nn.Sequential(layers)


    #load network weights
    if path is not None and group != "dataset":
        if not testing:
            safe_ckpt_load(network, path)
        else:
            ckpt=torch.load(path)
            state_dict=ckpt['model']
            new_state_dict={}
            for k, v in state_dict.items():
                k=k.replace("module.","")
                k=k.replace("shortcut", "downsample")
                new_state_dict[k]=v
            state_dict=new_state_dict
            network.load_state_dict(state_dict)

    if classifying:
        layers = OrderedDict([("encoder", network.features), ("classifier", classifier)])
        network = torch.nn.Sequential(layers)

    # 4. Adjust network with addtional wrappers, hooks and etc.
    # When downloaded we change the label scheme from the activations..
    if subset_wrap:
        logger.info(f"Adopting network from imagenet1000 to imagenet{num_classes}.")
        network = ImagenetSubsetWrapper(
            network, num_classes, imagenet_subset_path, imagenet_path
        )

    # Add a reset hook for lottery style pruning.
    add_reset_hook(
        network,
        partial(
            create_network,
            group=group,
            name=name,
            num_classes=num_classes,
            scaling_factor=scaling_factor,
        ),
        version,
    )

    return network


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self,x):
        return x

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
    def forward(self,x):
        return F.normalize(x)

class ImagenetSubsetWrapper(nn.Module):
    """Changes predictions for models trained on imagenet to a subset."""

    def __init__(
        self,
        network: nn.Module,
        num_classes: int,
        root: str,
        super_root: str,
    ):
        super(ImagenetSubsetWrapper, self).__init__()
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


def _scalable_imagenet_models(name: str, scaling_factor: float):
    if name == "mobilenet_v2":
        return partial(imagenet_models.mobilenet_v2, width_mult=scaling_factor)
    elif name == "mnasnet":
        return partial(imagenet_models.MNASNet, alpha=scaling_factor)
    elif name[:6] == "resnet":
        return partial(getattr(scalable_resnet, name), scaling_factor=scaling_factor)

    raise NotImplementedError


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


def create_teacher(
    group: str = None,
    name: str = None,
    num_classes: int = None,
    activations_path: str = None,
    ckpt_path: str = None,
    model_path: str = None,
    version: int = None,
    download: bool = False,
    scaling_factor: float = 1.0,
    imagenet_subset_path: Optional[str] = None,
    imagenet_path: Optional[str] = None,
):
    """Creates teachers."""
    logger.info("Creating teacher network...")

    if activations_path is not None:
        if imagenet_path is not None:
            logger.info(f"Adopting teacher database to {num_classes} classes.")
            network = DatabaseNetwork(activations_path, 1000)
            return ImagenetSubsetWrapper(
                network, num_classes, imagenet_subset_path, imagenet_path
            )
        return DatabaseNetwork(activations_path, num_classes)

    network = create_network(
        group=group,
        name=name,
        num_classes=num_classes,
        ckpt_path=ckpt_path,
        model_path=model_path,
        version=version,
        download=download,
        scaling_factor=scaling_factor,
        imagenet_path=imagenet_path,
        imagenet_subset_path=imagenet_subset_path,
    )

    return Teacher(network)
