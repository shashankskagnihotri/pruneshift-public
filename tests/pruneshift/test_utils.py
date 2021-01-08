from pathlib import Path

import pytest
import torch
import torch.nn.utils.prune as torch_prune

from pruneshift.utils import load_prune_ckpt
from pruneshift.prune_info import PruneInfo
import pytorch_lightning as pl
from conftest import DummyNet


class DummyVisionModule(pl.LightningModule):
    def __init__(self, network):
        super(DummyVisionModule, self).__init__()
        self.network = network


@pytest.fixture
def ckpt_path(tmpdir, dummy_net, lightning: bool = False):
    path = Path(tmpdir)/"dummy.ckpt"
    torch_prune.l1_unstructured(dummy_net.conv1, "weight", 2)

    if lightning:
        dummy_net = DummyVisionModule(dummy_net)
        state_dict = {"state_dict": dummy_net.state_dict()}
    else:
        state_dict = dummy_net.state_dict()

    torch.save(state_dict, path)

    return path


def test_load_pruned_ckpt(dummy_net, ckpt_path):
    loaded_net = load_prune_ckpt(DummyNet(), ckpt_path)
    assert (PruneInfo(dummy_net).summary() == PruneInfo(loaded_net).summary()).to_numpy().all()
