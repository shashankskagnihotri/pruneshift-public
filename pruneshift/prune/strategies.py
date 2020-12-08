import torch
import torch.nn.utils.prune as prune
import torch.nn as nn

from pruneshift.prune.utils import simple_prune


Absolute = prune.L1Unstructured


class AbsoluteGrad(prune.L1Unstructured):
    """
    Examples:
        >>> net = nn.Sequential(nn.Linear(1, 4),
        ...                     nn.Linear(4, 1))
        >>> net(torch.tensor([2.])).backward()
        >>> simple_prune(net, AbsoluteGrad, amount=2)
    """

    def compute_mask(self, t, default_mask):
        return super(AbsoluteGrad, self).compute_mask(t.grad, default_mask)

