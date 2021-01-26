from pathlib import Path
from typing import Union

import torch.nn.utils.prune as torch_prune
import torch.nn as nn
import torch


def load_state_dict(path: Union[str, Path]):
    """ Loads a state_dict """
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)

    if "state_dict" not in state_dict:
        return state_dict

    state_dict = state_dict["state_dict"]

    def prune_name(name):
        idx = name.find(".")
        return name[idx + 1:]

    return {prune_name(n): p for n, p in state_dict.items()}


def load_pruned_state_dict(network: nn.Module, path: Union[str, Path]):
    """ Changes and loads the network accordingly to the checkpoint."""
    # 1. First convert the state_dict with the flat parameter names.
    state_dict = load_state_dict(path)

    # 2. Find all params that need to be pruned.
    for param_name, param in list(network.named_parameters()):
        if param_name in state_dict:
            continue
        elif param_name + "_orig" in state_dict:
            # TODO: This is probably wrong and can be simplified.
            idx = param_name.rfind(".")
            module_name, param_name = param_name[:idx], param_name[idx + 1:]
            module = network
            for submodule_name in module_name.split("."):
                module = getattr(module, submodule_name)
            # Prune with identity so we can load into the pruning sheme.
            torch_prune.identity(module, param_name)
        else:
            raise ValueError("Checkpoints might be not compatible.")

    network.load_state_dict(state_dict)
    return network
