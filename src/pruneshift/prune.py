from typing import Type, Dict, Union, Callable
from functools import partial

import torch
from torch import nn as nn
from torch.nn.utils import prune as prune_torch


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


def prune(network: nn.Module, method: str, ratio: float) -> PruneInfo:
    """Prunes a network inplace.

    Note that some networks require having loaded a gradient already.

    Args:
        network: The network to prune.
        method: The name of the pruning method.

    Returns:
        Returns info about the pruning.
    """

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
    elif method == "l1_channels":
        pruning_cls = partial(prune_torch.ln_structured, n=1, dim=0)
        layerwise = True
    elif method == "random":
        pruning_cls = prune_torch.RandomUnstructured
        layerwise = False
    else:
        raise ValueError(f"Unknown pruning method: {method}")

    prune_info = PruneInfo(network)
    amount = prune_info.ratio_to_amount(ratio)
    simple_prune(prune_info, pruning_cls, layerwise, amount=amount)

    if shuffle:
        shuffle_masks(prune_info)

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
    **kwargs
):
    pairs = list(prune_info.target_pairs())

    if not layerwise:
        prune_torch.global_unstructured(pairs, pruning_method, **kwargs)
    else:
        for submodule, param_name in pairs:
            if not hasattr(pruning_method, "apply"):
                pruning_method(submodule, param_name, **kwargs)
                continue

            pruning_method.apply(submodule, param_name, **kwargs)

