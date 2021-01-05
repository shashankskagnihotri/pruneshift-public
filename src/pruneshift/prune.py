# TODO: Add structured pruning.
from typing import Type, Dict, Union
import logging

import re
import torch
from torch import nn as nn
from torch.nn.utils import prune as prune_torch
import gin


from .prune_info import PruneInfo


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


@gin.configurable
def prune(info: PruneInfo, method: str, ratio: float):
    """Prunes a network inplace.

    Note that some networks require having loaded a gradient already.

    Args:
        info: Information about the network and its prunable parameters.
        method: The name of the pruning method.

    Returns:
        Returns the same module but pruned. 
    """

    if method == "global_weight":
        pruning_cls = prune_torch.L1Unstructured
        layerwise = False
    elif method == "layer_weight":
        pruning_cls = prune_torch.L1Unstructured
        layerwise = True
    elif method == "global_grad":
        pruning_cls = L1GradUnstructered
        layerwise = False
    elif method == "layer_grad":
        raise NotImplementedError
        pruning_cls = L1GradUnstructered
        layerwise = True
        raise NotImplementedError
    elif method == "channels":
        pruning_cls = prune_torch.LnStructured
        layerwise = True
        raise NotImplementedError
    elif method == "random":
        pruning_cls = prune_torch.RandomUnstructured
        layerwise = False 
    else:
        raise ValueError(f"Unknown pruning method: {method}")
    
    amount = prune_info.ratio_to_amount(ratio)
    simple_prune(prune_info, pruning_cls, layerwise, amount=amount)
 

def simple_prune(
    prune_info: PruneInfo,
    pruning_method: Union[str, Type[prune_torch.BasePruningMethod]],
    layerwise: bool = False,
    **kwargs
):
    pairs = prune_info.target_pairs()

    if not layerwise:
        prune_torch.global_unstructured(pairs, pruning_method, **kwargs)
        return

    for submodule, param_name in pairs:
        pruning_method.apply(submodule, param_name, **kwargs)
