""" Provides annotations to networks that we used."""
import torch.nn as nn
import torchvision.models as tv_models
import timm.models as timm_models
import scalable_resnet 

import models as custom_models
import cifar_resnet as cifar
from .teachers import Teacher
from .utils import ImagenetSubsetWrapper 


RESNET_CLASSES = (cifar.ResNet, tv_models.ResNet, custom_models.ResNet, timm_models.ResNet, scalable_resnet.ResNet)


def _unwrap(network: nn.Module):
    """ Unwraps modules that are wrapped by wrappers we use."""
    if isinstance(network, (ImagenetSubsetWrapper, Teacher)):
        return _unwrap(network.network)
    return network


def protect_classifier(network: nn.Module, ensemble: bool):
    """ Marks the classification layers as non prunable."""
    network = _unwrap(network)

    if isinstance(network, RESNET_CLASSES):
        network.conv1.is_protected = True

    classifier(network, ensemble).is_protected = True


def classifier(network: nn.Module, ensemble):
    """ Returns the classification layer of the resnet."""
    network = _unwrap(network)

    if isinstance(network, RESNET_CLASSES):
        return network.fc
    elif isinstance(network, tv_models.MNASNet):
        return network.classifier[-1]
    elif isinstance(network, tv_models.MobileNetV2):
        return network.classifier[-1]
    elif ensemble:
        return network.classifier
    else:
        #raise NotImplementedError
        network=network.network
        #return network[-1]
        return classifier(network)


def at_entry_points(network: nn.Module):
    """ Retunrs the classification layer of the resnet."""
    network = _unwrap(network)

    if isinstance(network, RESNET_CLASSES):
        return {"layer0": network.layer1,
                "layer1": network.layer2,
                "layer2": network.layer3,
                "layer3": network.layer4}
    elif isinstance(network, tv_models.MobileNetV2):
        num_blocks = len(network.features)
        indices = [i for i in range(num_blocks - 1, 0, - num_blocks // 4)][::-1]
        return {f"layer{num}": network.features[idx] for num, idx in enumerate(indices)}
    elif isinstance(network, tv_models.MNASNet):
        num_blocks = len(network.layers)
        indices = [i for i in range(num_blocks - 1, 0, - num_blocks // 4)][::-1]
        return {f"layer{num}": network.layers[idx] for num, idx in enumerate(indices)}
    else:
        network=network.network
        at_entry_points(network)
        #raise NotImplementedError

