from typing import Type, Dict, Union
import logging

import torch
from torch import nn as nn
from torch.nn.utils import prune as prune_torch
import gin


logger = logging.getLogger(__name__)
BASIC_MODULE_MAP = {nn.Linear: ["weight", "bias"], nn.Conv2d: ["weight", "bias"]}


class L1GradUnstructered(prune_torch.L1Unstructured):
    """
    Examples:
        >>> net = nn.Sequential(nn.Linear(1, 4),
        ...                     nn.Linear(4, 1))
        >>> net(torch.tensor([2.])).backward()
        >>> simple_prune(net, AbsoluteGradMethod, amount=2)
    """
    def compute_mask(self, t, default_mask):
        print(t)
        print(t.grad)
        return super(L1GradUnstructered, self).compute_mask(t.grad, default_mask)



@gin.configurable
def prune(module: nn.Module, method: str, ratio: float):
    """Prunes a network.

    Note that some networks require having loaded a gradient already.

    Args:
        module: The module that should be pruned (inplace).
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

    amount = 1 - 1 / ratio 
    simple_prune(module, pruning_cls, layerwise, amount=amount)

def simple_prune(
    module: nn.Module,
    pruning_method: Union[str, Type[prune_torch.BasePruningMethod]],
    layerwise: bool = False,
    module_map: Dict = None,
    **kwargs
):
    """
    Examples:
        >>> net = nn.Sequential(nn.Linear(10, 4),
        ...                     nn.Linear(4, 1))
        >>> simple_prune(net, prune.L1Unstructured, amount=2)
        >>> simple_prune(net, prune.L1Unstructured, layerwise=True, amount=1)
    """
    if module_map is None:
        module_map = BASIC_MODULE_MAP

    pairs = []
    for m in module.modules():
        if type(m) not in module_map:
            continue

        for param_name in module_map[type(m)]:
            if not hasattr(m, param_name):
                continue
            param = getattr(m, param_name)
            if param is not None:
                pairs.append((m, param_name))

    if not layerwise:
        prune_torch.global_unstructured(pairs, pruning_method, **kwargs)
        return

    for m, param_name in pairs:
        pruning_method.apply(m, param_name, **kwargs)

