"""Provides network topologies for CIFAR10 and ImageNet."""
import re

import torchvision.models as imagenet_models
import cifar10_models


NETWORK_REGEX = re.compile(r"(?P<group>[a-zA-Z0-9]+)_(?P<name>[a-zA-Z0-9_]+)")


def create_network(name: str, pretrained=True, **kwargs):
    """A function creating common networks for either CIFAR10 and ImageNet

    Args:
        name: For example cifar10_resnet.
        pretrained: Whether the network should be pretrained.

    Returns:
        The desired network."""
    match = NETWORK_REGEX.match(name)

    if match is None:
        msg = f"Name {name} does not match the network regex {NETWORK_REGEX}."
        raise ValueError(msg)

    group = match["group"]
    name = match["name"]

    if group == "cifar10":
        network_fn = getattr(cifar10_models, name)
    elif group == "imagenet":
        network_fn = getattr(imagenet_models, name)
    else:
        raise ValueError(f"Unknown group {group}.")
 
    network = network_fn(pretrained=pretrained, **kwargs)

    # Mark the readout layer as protected.
    if name[: 6] == "resnet":
        network.fc.is_protected = True
    elif name[: 3] == "vgg":
        network.classifier[-1].is_protected = True
    else:
        raise NotImplementedError

    return network

