"""Provides network topologies for CIFAR and ImageNet datasets.

Note, that we did not find any pretrained models for Cifar10 and Cifar100:

Thus, they must be manually trained and provided in a checkpoint directory.
The checkpoints must have the following form:
    {network_id}.{version}
"""
import re
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as imagenet_models

from .utils import load_state_dict
import cifar10_models as cifar_models
import pytorch_resnet_cifar10.resnet as cifar_resnet
import models as models

import logging


logger = logging.getLogger(__name__)


NETWORK_REGEX = re.compile(
    r"(?P<group>[a-zA-Z]+)(?P<num_classes>[0-9]+)_(?P<name>[a-zA-Z0-9_]+)"
)


def protect_classifier(name: str, network: nn.Module):
    """ Defines which layers are protected. """
    if name[: 6] == "resnet":
        network.fc.is_protected = True
    elif name[: 3] == "vgg":
        network.classifier[-1].is_protected = True
    elif name[: 8] == "densenet":
        network.classifier.is_protected = True

    else:
        raise NotImplementedError


def load_checkpoint(
    network: nn.Module,
    network_id: str,
    ckpt_path: str = None,
    model_path: str = None,
    version: int = None
):
    """Loads a checkpoint."""

    if ckpt_path is not None:
        path = Path(ckpt_path)
    else:
        version = 0 if version is None else version
        path = Path(model_path) / f"{network_id}.{version}"
    state = load_state_dict(path)
    network.load_state_dict(state)


def create_network(
    network_id: str,
    ckpt_path: str = None,
    model_path: str = None,
    version: int = None,
    **kwargs
):
    """A function creating common networks for either CIFAR10 and ImageNet

    Args:
        network_id: For example cifar10_resnet50.
        prune: to implement width scaling
        ckpt_path: Direct path o a checkpoint of a network.
        model_path: Directory to find checkpoints in.
        version: Version of the checkpoint.
    Returns:
        The desired network.
    """

    match = NETWORK_REGEX.match(network_id)

    if match is None:
        msg = (
            f"Network Id {network_id} does not match the network regex {NETWORK_REGEX}."
        )
        raise ValueError(msg)

    group = match.group("group")
    num_classes = int(match.group("num_classes"))
    name = match.group("name")

    logger.info(f"Creating Network {name} for {group} with {num_classes} classes.")

    if group == "cifar":
        if name[: 6] == "resnet":
            network_fn = getattr(models, name)
        else:
            network_fn = getattr(models, name)
    elif group == "imagenet":
        network_fn = getattr(imagenet_models, name)
    else:
        raise ValueError(f"Unknown group {group}.")

    network = models.__dict__[name]()
    #print(models.__dict__[name]())

    if ckpt_path is not None or model_path is not None:
        load_checkpoint(network, network_id, ckpt_path, model_path, version)

    # We also want to have the classifier protected from pruning.
    protect_classifier(name, network) 

    return network

