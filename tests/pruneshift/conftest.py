import pytest
import torch.nn as nn

from pruneshift.prune_info import PruneInfo


class DummyNet(nn.Module):
    def __init__(self):
        super(DummyNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(2, 2, 1)

        self.batch = nn.BatchNorm2d(2)
        self.linear = nn.Linear(2, 2)
        self.linear.is_protected = True


@pytest.fixture
def dummy_net():
    return DummyNet()


@pytest.fixture
def prune_info(dummy_net):
    return PruneInfo(dummy_net)