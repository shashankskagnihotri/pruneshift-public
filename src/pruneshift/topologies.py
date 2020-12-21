"""Provides network topologies for CIFAR10 and ImageNet."""
import torchvision.models as imagenet_models
import cifar10_models


def network_topology(name: str, pretrained=True, **kwargs):
    """A function creating common networks for either CIFAR10 and ImageNet

    Args:
        name: For example cifar10_resnet.
        pretrained: Whether the network should be pretrained.

    Returns:
        The desired network."""
    idx = name.find("_")
    group, name = name[: idx], name[idx + 1:]

    if group == "cifar10":
        network_fn = getattr(cifar10_models, name)
    elif group == "imagenet":
        network_fn = getattr(imagenet_models, name)
    else:
        raise ValueError(f"Unknown group {group}.")
  
    return network_fn(pretrained=pretrained, **kwargs)
