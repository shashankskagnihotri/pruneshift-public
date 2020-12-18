import os
from pathlib import Path

import torch

# Singleton path to allow saving the models externally.
cifar10_path = None 


def load_model(model, arch, device):
    global cifar10_path
    if cifar10_path is None:
        cifar10_path = Path(os.environ["MODEL_PATH"]) / "cifar10"

    path = cifar10_path / f"{arch}.pt"
    model.load_state_dict(torch.load(path, map_location=device))


from .mobilenetv2 import *
from .resnet import *
from .vgg import *
from .densenet import *
from .resnet_orig import *
from .googlenet import *
from .inception import *

