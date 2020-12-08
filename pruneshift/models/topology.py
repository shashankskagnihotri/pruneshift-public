"""Provides network topologies."""
import torch.nn as nn


def conv_n_lottery(N: int = 2, num_classes: int = 10):
    """Creates the Conv-2, Conv-4 and Conv-6 of the lottery ticket
    hypothesis paper."""
    assert N in {2, 4, 6}
    layers = []

    in_channels = 3 
    for n in range(N // 2):
        out_channels = 64 * 2**n
        layers.extend([nn.Conv2d(in_channels, out_channels, 3),
                       nn.ReLU(),
                       nn.Conv2d(out_channels, out_channels, 3),
                       nn.ReLU(),
                       nn.MaxPool2d(2, 2)])
        in_channels = out_channels

    layers.extend([nn.Flatten(),
                   nn.Linear(12544, 256),
                   nn.ReLU(),
                   nn.Linear(256, 256),
                   nn.ReLU(),
                   nn.Linear(256, num_classes)])

    return nn.Sequential(*layers)

