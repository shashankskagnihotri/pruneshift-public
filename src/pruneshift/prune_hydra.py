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


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
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


class GetSubnetLn(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Taken from the torch prune module!
        # Calculates the norm accross the correct dimensions.
        n=1
        dim=0

        dims = list(range(scores.dim()))
        # convert negative indexing
        if dim < 0:
            dim = dims[dim]
        dims.remove(dim)

        norm = torch.norm(scores, p=n, dim=dims)

        num_to_keep = round((1 - k) * norm.numel())

        topk = torch.topk(norm, k=num_to_keep, largest=True)
        # topk will have .indices and .values

        # Compute binary mask by initializing it to all 0s and then filling in
        # 1s wherever topk.indices indicates, along self.dim.
        # mask has the same shape as tensor t
        def make_mask(t, dim, indices):
            # init mask to 0
            mask = torch.zeros_like(t)
            # e.g.: slc = [None, None, None], if len(t.shape) = 3
            slc = [slice(None)] * len(t.shape)
            # replace a None at position=dim with indices
            # e.g.: slc = [None, None, [0, 2, 3]] if dim=2 & indices=[0,2,3]
            slc[dim] = indices
            # use slc to slice mask and replace all its entries with 1s
            # e.g.: mask[:, :, [0, 2, 3]] = 1
            mask[slc] = 1
            return mask

        mask = make_mask(scores, dim, topk.indices)
        return mask 

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class HydraHook:
    def __init__(self, module, name, amount: float, mask_cls=GetSubnet):
        self.name = name
        self.amount = amount
        self.module = module
        self.mask_cls = mask_cls

        param = getattr(module, name)
        # Register buffers the same way as the pruning model.
        del module._parameters[name]
        module.register_buffer(name + "_orig", param)  # TODO: Should this be a buffer?
        n = nn.init._calculate_correct_fan(param, "fan_in")
        score = math.sqrt(6 / n) * param / torch.max(torch.abs(param))
        module.register_parameter(name + "_score", nn.Parameter(score))
        # nn.init.kaiming_uniform_(getattr(module, name + "_score"), a=math.sqrt(5))
        module.register_buffer(name, torch.zeros_like(param))
        module.register_buffer(name + "_mask", self.mask_cls.apply(score, amount))

        module.register_forward_pre_hook(self)

    def __call__(self, module, inputs):
        score = getattr(module, self.name + "_score")
        weight = getattr(module, self.name + "_orig")

        mask = self.mask_cls.apply(score.abs(), self.amount)
        setattr(module, self.name + "_mask", mask)
        setattr(module, self.name, mask * weight)

    def remove(self):
        # Make the original param a param again.
        mask = self.module._buffers.pop(self.name + "_mask").detach().clone()
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


def hydrate(network: nn.Module, ratio: float, method="weight"):
    """ Prunes a model and prepare it for the hydra pruning phase."""
    if method == "weight":
        mask_cls = GetSubnet
    elif method == "l1_channels":
        mask_cls = GetSubnetLn
    else:
        raise ValueError(f"Unknown mask selection type {method}.")

    info = PruneInfo(network, {nn.Linear: ["weight"], nn.Conv2d: ["weight"]})

    amount = info.ratio_to_amount(ratio)

    target_pairs = set(info.target_pairs())

    for module, param_name in target_pairs:
        HydraHook(module, param_name, amount, mask_cls=mask_cls)

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
            del module._forward_pre_hooks[k]
            custom_from_mask(module, param_name, mask)

    # network.fc.bias.requires_grad = True
    # freeze_protected(info, True)

def is_hydrated(network: nn.Module):
    for module in network.modules():
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, HydraHook):
                return True

    return False

