from pathlib import Path
from typing import Union
from typing import Tuple
import logging

import numpy as np
from ptflops import get_model_complexity_info
import torch.nn.utils.prune as torch_prune
import torch.nn as nn
import torch

from pruneshift.prune_hydra import hydrate
from pruneshift.prune_hydra import dehydrate
from pruneshift.prune_hydra import is_hydrated



def safe_ckpt_load(network: nn.Module, path: Union[str, Path]):
    """ Safe way to load a network."""
    state_dict = load_state_dict(path)
    hydrated, pruned = False, False

    for param_name in state_dict:
        if param_name[-5:] == "score":
            hydrated = True
        if param_name[-4:] == "mask":
            pruned = True

    if hydrated:
        load_hydrated_state_dict(network, state_dict)
    elif pruned:
        load_pruned_state_dict(network, state_dict)
    else:
        network.load_state_dict(state_dict)


def load_state_dict(path: Union[str, Path]):
    """ Loads a state_dict """
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)

    if "state_dict" not in state_dict:
        return state_dict

    state_dict = state_dict["state_dict"]

    def prune_name(name):
        idx = name.find(".")
        # Prune the network part due to the lightning module.
        if name[: idx] == "network":
            return name[idx + 1 :]
        return name 

    return {prune_name(n): p for n, p in state_dict.items()}


def load_hydrated_state_dict(network: nn.Module, state_dict):
    """ Loads a hydrated network and brings it into finetuning phase."""
    hydrate(network, ratio=1)
    network.load_state_dict(state_dict)
    dehydrate(network)

    return network


def load_pruned_state_dict(network: nn.Module, state_dict):
    """ Changes and loads the network accordingly to the checkpoint."""
    # 2. Find all params that need to be pruned.
    for param_name, param in list(network.named_parameters()):
        if param_name + "_orig" in state_dict:
            # If the checkpoint was from hydra delete the scores.
            # del state_dict[param_name]

            idx = param_name.rfind(".")
            module_name, param_name = param_name[:idx], param_name[idx + 1 :]
            module = network
            for submodule_name in module_name.split("."):
                module = getattr(module, submodule_name)
            # Prune with identity so we can load into the pruning sheme.
            torch_prune.identity(module, param_name)

        elif not param_name in state_dict:
            raise ValueError(f"Missing {param_name}.")

    network.load_state_dict(state_dict)
    return network


def _consider_pruning(flops: int, module: nn.Module, param_name: str) -> int:
    if not hasattr(module, param_name + "_mask"):
        factor = 1.0
    else:
        factor = getattr(module, param_name + "_mask").mean().item()

    return int(factor * flops)


def _conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = (
        int(np.prod(kernel_dims)) * in_channels * filters_per_channel
    )

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count
    overall_conv_flops = _consider_pruning(overall_conv_flops, conv_module, "weight")

    bias_flops = 0

    if conv_module.bias is not None:
        bias_flops = out_channels * active_elements_count
        # Modification!
        bias_flops = _consider_pruning(bias_flops, conv_module, "bias")

    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)


def _linear_flops_counter_hook(module, input, output):
    input = input[0]
    # pytorch checks dimensions, so here we don't care much
    output_last_dim = output.shape[-1]
    if module.bias is not None:
        bias_flops = _consider_pruning(output_last_dim, module, "bias")
    else:
        bias_flops = 0

    weight_flops = int(np.prod(input.shape)) * output_last_dim
    weight_flops = _consider_pruning(weight_flops, module, "weight")

    module.__flops__ += weight_flops + bias_flops


def get_model_complexity_prune(
    module: nn.Module,
    input_res: Tuple[int, ...],
    print_per_layer_stat: bool = False,
    as_strings: bool = False,
):
    """Calculates the model complexity taking into account pruned conv-
    olutional layers and linear layers.
    """

    custom_modules_hooks = {
        nn.Conv2d: _conv_flops_counter_hook,
        nn.Linear: _linear_flops_counter_hook,
    }

    return get_model_complexity_info(
        module,
        input_res,
        print_per_layer_stat,
        as_strings=as_strings,
        custom_modules_hooks=custom_modules_hooks,
    )

