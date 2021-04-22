""" Provides annotations to networks that we used."""
import torch.nn as nn
import torchvision.models as tv_models
import timm.models as timm_models
import scalable_resnet

import models as custom_models
from pruneshift.teachers import ImagenetSubsetWrapper, Teacher


RESNET_CLASSES = (tv_models.ResNet, custom_models.ResNet, timm_models.ResNet, scalable_resnet.ResNet)


def _unwrap(network: nn.Module):
    """ Unwraps modules that are wrapped by wrappers we use."""
    if isinstance(network, (ImagenetSubsetWrapper, Teacher)):
        return _unwrap(network.network)
    return network


def protected(network: nn.Module):
    """ Returns layers that should be protected."""
    network = _unwrap(network)

    # To be consistent.
    if isinstance(network, RESNET_CLASSES):
        return [classifier(network), network.conv1]

    return [classifier(network)]


def classifier(network: nn.Module):
    """ Returns the classification layer of the resnet."""
    network = _unwrap(network)

    if isinstance(network, RESNET_CLASSES):
        return network.fc
    elif isinstance(network, tv_models.MNASNet):
        return network.classifier[-1]
    else:
        raise NotImplementedError


def at_entry_points(network: nn.Module):
    """ Retunrs the classification layer of the resnet."""
    network = _unwrap(network)

    if isinstance(network, RESNET_CLASSES):
        return {"layer1": network.layer1,
                "layer2": network.layer2,
                "layer3": network.layer3,
                "layer4": network.layer4}
    else:
        raise NotImplementedError
