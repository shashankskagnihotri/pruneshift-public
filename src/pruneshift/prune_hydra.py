import os
from pathlib import Path
import math
from functools import partial

import torch
import torch.nn as nn
from torch.nn.utils.prune import custom_from_mask
import torch.autograd as autograd
import torch.optim as optim
import pytorch_lightning as pl

from pruneshift.prune_info import PruneInfo


# def percentile(t, q):
#     k = 1 + round(.01 * float(q) * (t.numel() - 1))
#     return t.view(-1).kthvalue(k).values.item()
# 
# 
# class GetSubnet(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, scores, sparsity):
#         k_val = percentile(scores, (1 - sparsity) *100)
#         return scores < k_val
# 
#     @staticmethod
#     def backward(ctx, g):
#         return g, None 

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # scores = scores.clone()
        # num_params = round(k * scores.numel())
        # mask = torch.ones_like(
        #     scores, dtype=torch.bool, memory_format=torch.contiguous_format
        # )
        # topk = torch.topk(scores.view(-1), k=num_params, largest=False)
        # mask.view(-1)[topk.indices] = 0

        # return mask

        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        # ! We flipped it around !!!
        j = int(k * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class HydraHook:
    def __init__(self, module, name, amount: float):
        self.name = name
        self.amount = amount
        self.module = module

        param = getattr(module, name)
        # Register buffers the same way as the pruning model.
        del module._parameters[name]
        module.register_buffer(name + "_orig", param)  # TODO: Should this be a buffer?
        n = nn.init._calculate_correct_fan(param, "fan_in")
        score = math.sqrt(6 / n) * param / torch.max(torch.abs(param))
        module.register_parameter(name + "_score", nn.Parameter(score))
        # nn.init.kaiming_uniform_(getattr(module, name + "_score"), a=math.sqrt(5))
        module.register_buffer(name, torch.zeros_like(param))
        module.register_buffer(name + "_mask", GetSubnet.apply(score, amount))

        module.register_forward_pre_hook(self)

    def __call__(self, module, inputs):
        score = getattr(module, self.name + "_score")
        weight = getattr(module, self.name + "_orig")

        mask = GetSubnet.apply(score.abs(), self.amount)
        setattr(module, self.name + "_mask", mask)
        setattr(module, self.name, mask * weight)

    def remove(self):
        # Make the original param a param again.
        mask = self.module._buffers.pop(self.name + "_mask")
        param = self.module._buffers.pop(self.name + "_orig")
        score = self.module._parameters.pop(self.name + "_score")
        # Register the main buffer.
        del self.module._buffers[self.name]
        self.module.register_parameter(self.name, nn.Parameter(param))

        return self.name, mask


def freeze_protected(info, reverse: bool = False):
    for submodule in info.network.modules():
        for name, param in submodule.named_parameters():
            if not info.is_protected(submodule):
                continue
            param.requires_grad = reverse


def hydrate(network: nn.Module, ratio: float):
    """ Prunes a model and prepare it for the hydra pruning phase."""
    info = PruneInfo(network, {nn.Linear: ["weight"], nn.Conv2d: ["weight"]})

    amount = info.ratio_to_amount(ratio)

    target_pairs = set(info.target_pairs())

    for module, param_name in target_pairs:
        HydraHook(module, param_name, amount)

    # freeze_protected(info)
    # network.fc.bias.requires_grad = False
    return info


def dehydrate(network: nn.Module):
    """ Changes the hydrated network to a normal pruned network."""
    # Collects modules, param_names and mask from the hooks.
    info = PruneInfo(network, {nn.Linear: ["weight", "bias"], nn.Conv2d: ["weight"]})

    for module in network.modules():
        hooks_to_remove = []
        for k, hook in list(module._forward_pre_hooks.items()):
            param_name, mask = hook.remove()
            custom_from_mask(module, param_name, mask)
            del module._forward_pre_hooks[k]

    # network.fc.bias.requires_grad = True
    # freeze_protected(info, True)
