from typing import Callable, Dict

import torch.nn.utils.prune as prune
import torch.nn as nn


BASIC_MODULE_MAP = {nn.Linear: ["bias", "weight"], nn.Conv2d: ["bias", "weight"]}



def simple_prune(
    module: nn.Module,
    pruning_method: prune.BasePruningMethod,
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

    pairs = [(m, p) for m in module.modules() for p in module_map.get(type(m), [])]

    if not layerwise:
        prune.global_unstructured(pairs, pruning_method, **kwargs)
        return

    for m, p in pairs:
        pruning_method.apply(m, p, **kwargs)
