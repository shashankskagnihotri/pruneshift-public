"""
Metrics, here are adopted from shrinkbench:

https://github.com/JJGO/shrinkbench/blob/master/pruning/utils.py
"""
from typing import Tuple
import torch.nn as nn
import numpy as np

def model_size(model: nn.Module, as_bits: bool = False) -> Tuple[int, int]:
    """ Returns absolute and nonzero model size of a model.

    Args:
        model: The model to measure the size of.
        as_bits: Whether taking account the precision of tensors.

    Returns:
        The number of parameters of the models
    """
    total_params = 0
    nonzero_params = 0
    for tensor in model.parameters():
        length = np.prod(tensor.shape)
        if tensor 
        num_ = nonzero(tensor.detach().cpu().numpy())
        if as_bits:
            bits = dtype2bits[tensor.dtype]
            length *= bits
            nz *= bits
        total_params += length 
        nonzero_params += nz
    return int(total_params), int(nonzero_params)
