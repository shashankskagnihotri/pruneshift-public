from typing import Type, Dict, Union, Callable
from functools import partial
import math

import torch
from torch import nn as nn
from torch.nn.utils import prune as prune_torch
import numpy as np

from .prune_info import PruneInfo


class MaskShuffel(prune_torch.L1Unstructured):
    def compute_mask(self):
        mask = super(MaskShuffel, self).compute_mask()
        idx = torch.randperm(mask.numel())
        return mask.view(-1)[idx].view(mask.size())


class L1GradUnstructered(prune_torch.L1Unstructured):
    """
    Examples:
        >>> net = nn.Sequential(nn.Linear(1, 4),
        ...                     nn.Linear(4, 1))
        >>> net(torch.tensor([2.])).backward()
        >>> simple_prune(net, AbsoluteGradMethod, amount=2)
    """

    def compute_mask(self, t, default_mask):
        return super(L1GradUnstructered, self).compute_mask(t.grad, default_mask)


class ZeroWeights(prune_torch.CustomFromMask):
    """ Prunes all the zero weights, for checkpoints from other frameworks."""
    def compute_mask(self, t, default_mask):
        return super(ZeroWeights, self).__init__(t == 0, default_mask)


def unwider_resnet(module: nn.Module, name: str, amount: float):
    """ Uses the pruning module to scale down the width of the resnet."""
    t = getattr(module, name)

    if isinstance(module, nn.Conv2d):
        # Scale the initialization with the amount that was pruned.
        gain = math.sqrt(2.0)
        fan_out = t.size(0) * t[0][0].numel() * (1 - amount)
        std = gain / math.sqrt(fan_out)
        init = torch.zeros_like(t).normal_(0, std)
        setattr(module, name, nn.Parameter(init))
        prune_torch.random_structured(module, name, amount, dim=0)

    else:
        raise NotImplementedError


def prune(
    network: nn.Module,
    method: str,
    ratio: float = None,
    amount: float = None,
    reset_weights: bool = False,
) -> PruneInfo:
    """Prunes a network inplace.

    Note that some networks require having loaded a gradient already.

    Args:
        network: The network to prune.
        method: The name of the pruning method.
        ratio: The compresson ratio.
        amount: The amount top prune.
        reset_weights: If passed resets the weight with the given seed.

    Returns:
        Returns info about the pruning.
    """
    assert ratio != amount

    if amount is not None:
        ratio = 1 / (1 - amount)

    shuffle = False

    if method == "global_weight":
        pruning_cls = prune_torch.L1Unstructured
        layerwise = False
    elif method == "global_weight_shuffle":
        pruning_cls = prune_torch.L1Unstructured
        layerwise = False
        shuffle = True
    elif method == "layer_weight":
        pruning_cls = prune_torch.L1Unstructured
        layerwise = True
    elif method == "global_grad":
        pruning_cls = L1GradUnstructered
        layerwise = False
    elif method == "layer_grad":
        pruning_cls = L1GradUnstructered
        layerwise = True
        raise NotImplementedError
    elif method == "uniform":
        pruning_cls = unwider_resnet
        layerwise = True
    elif method == "l1_channels":
        pruning_cls = partial(prune_torch.ln_structured, n=1, dim=0)
        layerwise = True
    elif method == "l1_global":
        pruning_cls = partial(prune_torch.ln_structured, n=1, dim=0)
        layerwise = False
    elif method == "random_channels":
        pruning_cls = partial(prune_torch.random_structured, dim=0) 
        layerwise = True 
    elif method == "random_weight":
        pruning_cls = prune_torch.RandomUnstructured
        layerwise = False
    elif method == "zero_weights":
        pruning_cls = ZeroWeights
        layerwise = True
    else:
        raise ValueError(f"Unknown pruning method: {method}")

    prune_info = PruneInfo(network)
    threshold=0.0
    
    amount = prune_info.ratio_to_amount(ratio)

    if method=="l1_global":
        weights = np.array(1,)
        for m in network.modules():
            if isinstance(m, nn.Conv2d):
                w = m.weight.data.abs().detach().cpu().numpy().flatten()
                weights = np.hstack([weights, w])
        weights=np.sort(weights)
        threshold=weights[int(amount*len(weights))]

    # Calculate the amount that needs to pruned, to reach the 
    # target size.
    simple_prune(prune_info, pruning_cls, layerwise, method=method, threshold=threshold, amount=amount)

    if shuffle:
        shuffle_masks(prune_info)

    if reset_weights:
        # The reset hook should have been added by the create_network
        # function.
        network.__reset_hook()

    return prune_info


def shuffle_masks(prune_info: PruneInfo):
    for module, param_name in prune_info.target_pairs():
        mask = getattr(module, param_name + "_mask")
        idx = torch.randperm(mask.numel())
        shuffled_mask = mask.view(-1)[idx].view(mask.size())
        setattr(module, param_name + "_mask", shuffled_mask)


def simple_prune(
    prune_info: PruneInfo,
    pruning_method: Union[Callable, Type[prune_torch.BasePruningMethod]],
    layerwise: bool = False,
    method: str = "l1_global",
    threshold: float = 0.0,
    **kwargs,
):
    pairs = list(prune_info.target_pairs())

    if method=="l1_global":#isinstance(prune_method, prune_torch.LnStructured) and not layerwise:
        for submodule, param_name in pairs:
            weight=submodule.weight.data.abs().detach().cpu().numpy().flatten()
            amt=((weight<threshold).sum())/len(weight)
            pruning_method(submodule, param_name, amount=amt)
    elif not layerwise:
        prune_torch.global_unstructured(pairs, pruning_method, **kwargs)
    elif isinstance(prune_method, prune_torch.LnStructured) and not layerwise:
        for submodule, param_name in pairs:
            weight=submodule.weight.data.abs().detach().cpu().numpy().flatten()
            amt=((weight<threshold).sum())/len(weight)
            pruning_method(submodule, param_name, amount=amt)
    else:
        for submodule, param_name in pairs:
            if not hasattr(pruning_method, "apply"):
                pruning_method(submodule, param_name, **kwargs)
                continue

            pruning_method.apply(submodule, param_name, **kwargs)

