""" Provides information about which modules of a net should be pruned and allows to calculate the information accordingly."""
from typing import List 
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


class PruneInfo:
    def __init__(self, network: nn.Module):
        self.network = network
        
        if not len(self.readout_layers):
            raise RuntimeError("Found no module annotated with is_classifier!")


    def readout_layers(self) -> List[nn.Module]:
        """ Returns the readout_layer of the network.

        This should be subclassed and is necessary to prevent pruning
        of the readout_layer."""
        layers = []
        for submodule in self.network.modules():
            if not hasattr(submodule, "is_protected"):
                continue
            if submodule.is_protected:
                layers.append(submodule)

        return layers

    def is_target(self, module: nn.Module, param_name: str):
        """ Returns whether a module param combination should be a
        target for pruning."""
        raise NotImplementedError

    def target_pairs(self):
        """ Returns the module parameter pairs that are marked for pruning."""
        readout_layers = self.readout_layers()
        for submodule in module.modules():
            if (submodule):
                continue
            for _, param_name in module.named_parameters():
                if self.is_target(submodule, param_name):
                    yield submodule, param_name

    def ratio_to_amount(self, ratio: float) -> float:
        """ Converts the ratio into the amount to prune."""
        amount = (1 - 1/ratio) * self.model_size()/self.pruned_model_size()
        if amount > 1:
            msg = f"Compression ratio {ratio} can not be achieved, as the" \
                   "percentage of prunable modules is too low."
            raise RuntimeError(msg)
        return amount

    def model_size(self, as_bits: bool = False):
        """ Returns the size of the model."""
        size = 0
        for param in self.network.parameters():
            factor = DTYPE2BITS[param.dtype] if as_bits else 1
            size += factor * param.numel()
        return size

    def pruned_model_size(self, as_bits: bool = False):
        """ Returns the size of the pruned model."""
        size = 0
        for submodule, param_name in self.target_pairs():
            param = getattr(submodule, param_name)
            mask = getattr(submodule, param_name + "_mask")
            factor = DTYPE2BITS[param.dtype] if as_bits else 1
            size += factor * torch.sum(mask).item()
        return size

    def summary(self):
        raise NotImplementedError

