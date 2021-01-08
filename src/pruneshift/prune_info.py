""" Provides information about which modules of a net should be pruned
and allows to calculate the information accordingly."""
from typing import Iterable, Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import prune


# Taken from shrinkbench!
DTYPE2BITS = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1,
}


def param_size(
    module: nn.Module,
    param_name: str = None,
    original: bool = False,
    as_bits: bool = False,
):
    """ Calculates the size of a parameter in a module."""
    param = getattr(module, param_name)
    factor = DTYPE2BITS[param.dtype] if as_bits else 1

    if param_name[-5:] == "_orig" and not original:
        mask = getattr(module, param_name[:-5] + "_mask")
        return int(factor * mask.sum().item())

    return int(factor * param.numel())


DEFAULT_TARGETS = {nn.Linear: ["weight", "bias"], nn.Conv2d: ["weight", "bias"]}


class PruneInfo:
    def __init__(self, network: nn.Module, target_map: Dict = None):
        self.network = network
        self.target_map = DEFAULT_TARGETS if target_map is None else target_map

    def is_protected(self, module: nn.Module) -> bool:
        """ Iterates over the protected modules of the network."""
        return getattr(module, "is_protected", False)

    def is_target(self, module: nn.Module, param_name: str):
        """ Returns whether a module param pair is a target for pruning."""
        if param_name[-5:] == "_orig":
            param_name = param_name[:-5]
        return param_name in self.target_map.get(type(module), [])

    def _iter_pairs(self, complete=False):
        """ Iterates over all module, param pairs."""
        for module_name, module in self.network.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if complete:
                    yield module_name, module, param_name, param
                else:
                    yield module, param_name

    def target_pairs(self):
        """ Returns the module parameter pairs that are marked for pruning."""
        for module, param_name in self._iter_pairs():
            if self.is_protected(module):
                continue
            if self.is_target(module, param_name):
                yield module, param_name

    def ratio_to_amount(self, ratio: float) -> float:
        """ Converts the ratio into the amount to prune.

        Takes into account that the wrapped network might be already pruned."""
        assert ratio >= 1
        curr_size = self.network_size()
        curr_target_size = self.target_size()
        amount = (1 - 1 / ratio) * curr_size / curr_target_size
        if amount > 1.00000001:  # There must be a better way.
            msg = (
                f"Compression ratio {ratio} can not be achieved, as the "
                 "percentage of prunable parameters is too low."
            )
            raise ValueError(msg)
        return amount

    def network_size(self, original: bool = False, as_bits: bool = False) -> int:
        """ The size of the network."""
        return sum(
            [param_size(m, pn, original, as_bits) for m, pn in self._iter_pairs()]
        )

    def target_size(self, original: bool = False, as_bits: bool = False) -> int:
        """ The size of all target modules."""
        return sum(
            [param_size(m, pn, original, as_bits) for m, pn in self.target_pairs()]
        )

    def network_comp(self):
        """ The current compression rate of the network."""
        return self.network_size(True) / self.network_size()

    def summary(self):
        """ Returns a summary of the pruning statistics of the model.

        Adopted from shrinkbench."""
        rows = []
        for module_name, module, param_name, param in self._iter_pairs(True):
            orig_size = param_size(module, param_name, original=True)
            # When the original size was zero the param was only an
            # auxiliarly buffer.
            if orig_size == 0:
                continue
            curr_size = param_size(module, param_name)

            if param_name[-5: ] == "_orig":
                param_name = param_name[:-5]

            comp = orig_size / curr_size
            shape = tuple(param.shape)
            rows.append(
                [
                    module_name,
                    param_name,
                    comp,
                    1 - 1 / comp,
                    orig_size,
                    shape,
                    self.is_target(module, param_name),
                    self.is_protected(module)
                ]
            )
        columns = ["module", "param", "comp", "amount", "size", "shape", "target", "protected"]
        return pd.DataFrame(rows, columns=columns)
