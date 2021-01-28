import os
from pathlib import Path
import math
from functools import partial

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import pytorch_lightning as pl
from torch.optim import SGD

from pruneshift.modules import VisionModule
from pruneshift.networks import network
from pruneshift.datamodules import datamodule
from pruneshift.prune_info import PruneInfo
from pruneshift.prune import prune


class Maskarade(torch.autograd.Function):
    """ Pretends to be the original param!"""

    @staticmethod
    def forward(ctx, scores, weight, mask):
        hydrated_weight = weight * mask
        return hydrated_weight

    @staticmethod
    def backward(ctx, g_weight):
        return g_weight, None, None


# We do not have the stacking problem :)
class HydraHook:
    def __init__(self, module, name, amount: float):
        self.name = name
        self.amount = amount
        param = getattr(module, name)
        # Register buffers the same way as the pruning model.
        del module._parameters[name]
        module.register_buffer(name + "_orig", param)  # TODO: Should this be a buffer?
        if param.dim() < 2:
            n = 1
        else:
            n = nn.init._calculate_correct_fan(param, "fan_in")
        score = math.sqrt(6 / n) * param / torch.max(param)
        module.register_parameter(name + "_score", nn.Parameter(score))
        # nn.init.kaiming_uniform_(getattr(module, name + "_score"), a=math.sqrt(5))
        module.register_buffer(name, torch.zeros_like(param))

        module.register_forward_pre_hook(self)

    def abs_score(self, module):
        return torch.abs(getattr(module, self.name + "_score"))

    def compute_mask(self, module):
        score = self.abs_score(module)
        num_prune = round(score.numel() * (1 - self.amount))

        mask = torch.zeros_like(score, dtype=torch.bool)
        topk = torch.topk(score.view(-1), k=num_prune)
        mask.view(-1)[topk.indices] = 1

        return mask

    def __call__(self, module, inputs):
        score = self.abs_score(module)
        weight = getattr(module, self.name + "_orig")
        mask = self.compute_mask(module)
        hydrated_weight = Maskarade.apply(score, weight, mask)
        setattr(module, self.name, hydrated_weight)

    def remove(self, module):
        # Make the original param a param again.
        mask = self.compute_mask(module)
        param = module._buffers.pop(self.name + "_orig")
        score = module._parameters.pop(self.name + "_score")

        module.register_parameter(nn.Parameter(param))

        return self.name, mask


# TODO: Do we want to freeze protected layers.
def hydrate(network: nn.Module, ratio: float, init: str = None):
    """ Returns HydraHooks"""
    info = PruneInfo(network)
    amount = info.ratio_to_amount(ratio)
    for module, param_name in list(info.target_pairs()):
        HydraHook(module, param_name, amount)

    # network.fc.weight.requires_grad = False
    # network.fc.bias.requires_grad = False


def dehydrate(network: nn.Module):
    """ Change the hydrated network to a normal pruned network."""
    for module in network.modules():
        forward_pre_hooks = module._forward_pre_hooks
        print(type(forward_pre_hooks), len(forward_pre_hooks))
        for k, hook in module._forward_pre_hooks.items():
            print(k, hook)


if __name__ == "__main__":
    net = nn.Sequential(nn.Linear(1, 10), nn.Linear(10, 1))
    hydrate(net, 2)
    dehydrate(net)
