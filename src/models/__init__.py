from __future__ import absolute_import

import os
from pathlib import Path

import torch

# Singleton path to allow saving the models externally.
cifar10_path = None 


def load_model(model, arch):
    global cifar10_path
    if cifar10_path is None:
        cifar10_path = Path(os.environ["MODEL_PATH"])/"cifar10"

    path = cifar10_path / f"{arch}.pt"
    model.load_state_dict(torch.load(path))



from .vgg import *
from .resnet import *