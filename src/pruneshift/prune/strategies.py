from typing import Type

import torch
import torch.nn.utils.prune as prune
import torch.nn as nn

from pruneshift.prune.utils import simple_prune


def strategy(name: str) -> Type["RegisteredPruningMethod"]:
    """ Returns a RegisteredPruningMethod class."""

    return RegisteredPruningMethod.subclasses[name]



class RegisteredPruningMethod(prune.BasePruningMethod):
    subclasses = {}

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
        >>> simple_prune(net, AbsoluteGrad, amount=2)
    """
    name = "l1grad"

    def compute_mask(self, t, default_mask):
        return super(AbsoluteGradMethod, self).compute_mask(t.grad, default_mask)

