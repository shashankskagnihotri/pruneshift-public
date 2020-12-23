from typing import Type, Dict
import logging

import torch
from torch import nn as nn
from torch.nn.utils import prune as prune
import gin


logger = logging.getLogger(__name__)
BASIC_MODULE_MAP = {nn.Linear: ["weight"], nn.Conv2d: ["weight"]}


L1Unstructured = gin.external_configurable(prune.L1Unstructured)

# @gin.configurable
# class L1Unstructured(prune.BasePruningMethod):
#     def compute_mask(self, t, default_mask):
#         return super(L1Unstructured, self).compute_mask(t.grad, default_mask)


@gin.configurable
class L1GradUnstructered(prune.BasePruningMethod):
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
def simple_prune(
    module: nn.Module,
    pruning_method: Type[prune.BasePruningMethod],
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

