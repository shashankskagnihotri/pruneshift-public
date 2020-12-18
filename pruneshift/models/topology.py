"""Provides network topologies for CIFAR10 and ImageNet."""
import re

import torch.nn as nn
import torchvision.models as imagenet_models 
import pruneshift.models.cifar10_models as cifar10_models



def topology(name: str, pretrained=True, **kwargs):
    """A function creating common networks for either CIFAR10 and ImageNet

    Args:
        name: For example cifar10_resnet.
        pretrained: Whether the network should be pretrained.

    Returns:
        The desired network."""
    idx = name.find("_")
    group, name = name[ : idx], name[idx + 1: ]

    if group == "cifar10":
        network_fn = getattr(cifar10_models, name)
    elif group == "imagenet":
        network_fn = getattr(imagenet_models, name)
    else:
        raise ValueError(f"Unknwon group {group}.")
  
    return network_fn(pretrained=True, **kwargs)


# def conv_n_lottery(N: int = 2, num_classes: int = 10):
#     """Creates the Conv-2, Conv-4 and Conv-6 of the lottery ticket
#     hypothesis paper."""
#     assert N in {2, 4, 6}
#     layers = []
# 
#     in_channels = 3 
#     for n in range(N // 2):
#         out_channels = 64 * 2**n
#         layers.extend([nn.Conv2d(in_channels, out_channels, 3),
#                        nn.ReLU(),
#                        nn.Conv2d(out_channels, out_channels, 3),
#                        nn.ReLU(),
#                        nn.MaxPool2d(2, 2)])
#         in_channels = out_channels
# 
#     layers.extend([nn.Flatten(),
#                    nn.Linear(12544, 256),
#                    nn.ReLU(),
#                    nn.Linear(256, 256),
#                    nn.ReLU(),
#                    nn.Linear(256, num_classes)])
# 
#     return nn.Sequential(*layers)
# 
