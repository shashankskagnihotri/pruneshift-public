from typing import Type, Dict
import logging

import torch
from torch import nn as nn
from torch.nn.utils import prune as prune


logger = logging.getLogger(__name__)
BASIC_MODULE_MAP = {nn.Linear: ["weight"], nn.Conv2d: ["weight"]}


def prune_strategy(name: str) -> Type["RegisteredPruningMethod"]:
    """ Returns a RegisteredPruningMethod class."""
    return RegisteredPruningMethod.subclasses[name]


class RegisteredPruningMethod(prune.BasePruningMethod):
    subclasses = {}
    name = None

    def compute_mask(self, t, default_mask):
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs):
        # Add subclasses to the module.
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.name] = cls


class AbsoluteMethod(prune.L1Unstructured, RegisteredPruningMethod):
    name = "l1"


class AbsoluteGradMethod(RegisteredPruningMethod):
    """
    Examples:
        >>> net = nn.Sequential(nn.Linear(1, 4),
        ...                     nn.Linear(4, 1))
        >>> net(torch.tensor([2.])).backward()
        >>> simple_prune(net, AbsoluteGradMethod, amount=2)
    """
    name = "l1grad"

    def compute_mask(self, t, default_mask):
        return super(AbsoluteGradMethod, self).compute_mask(t.grad, default_mask)


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
    # logger.info(f"Prune module with {pruning_method}.")
    if module_map is None:
        module_map = BASIC_MODULE_MAP

    pairs = [(m, p) for m in module.modules() for p in module_map.get(type(m), [])]
    # logger.info(f"Will prune the following pairs:\n {pairs}")

    if not layerwise:
        prune.global_unstructured(pairs, pruning_method, **kwargs)
        return

    for m, p in pairs:
        pruning_method.apply(m, p, **kwargs)
