import pytest
import torch.nn.utils.prune as prune


class TestPruneInfo:
    def test_is_protected(self, dummy_net, prune_info):
        assert not prune_info.is_protected(dummy_net.conv1)
        assert not prune_info.is_protected(dummy_net.batch)
        assert prune_info.is_protected(dummy_net.linear)

    def test_is_target(self, dummy_net, prune_info):
        assert prune_info.is_target(dummy_net.conv1, "weight")
        assert prune_info.is_target(dummy_net.conv1, "bias")
        assert prune_info.is_target(dummy_net.linear, "bias")
        assert not prune_info.is_target(dummy_net.batch, "weight")
        prune.l1_unstructured(dummy_net.conv2, "weight", 3)
        assert prune_info.is_target(dummy_net.conv2, "weight")

    def test_target_pairs(self, dummy_net, prune_info):
        target_pairs = {(dummy_net.conv1, "weight"),
                        (dummy_net.conv2, "weight"),
                        (dummy_net.conv2, "bias")}
        assert len(list(prune_info.target_pairs())) == 3
        assert set(prune_info.target_pairs()) == target_pairs

    def test_network_size(self, dummy_net, prune_info):
        size = sum(p.numel() for p in dummy_net.parameters())
        assert prune_info.network_size() == size
        # Prune one param.
        prune.l1_unstructured(dummy_net.conv2, "weight", 3)
        assert prune_info.network_size() == size - 3
        prune.l1_unstructured(dummy_net.conv1, "weight", 2)
        assert prune_info.network_size() == size - 5
        assert prune_info.network_size(True) == size

    def test_target_size(self, dummy_net, prune_info):
        size = sum(getattr(m, pn).numel() for m, pn in prune_info.target_pairs())
        assert prune_info.target_size() == size
        # Prune one param.
        prune.l1_unstructured(dummy_net.conv2, "weight", 3)
        assert prune_info.target_size() == size - 3
        prune.l1_unstructured(dummy_net.conv1, "weight", 2)
        assert prune_info.target_size() == size - 5
        assert prune_info.target_size(True) == size

    def test_ratio_to_amount(self, dummy_net, prune_info):
        with pytest.raises(ValueError):
            prune_info.ratio_to_amount(10)

        assert prune_info.ratio_to_amount(2) == 1.
        ratio = 20 / 18
        assert pytest.approx(prune_info.ratio_to_amount(ratio), 0.000001) == 0.2
        prune.l1_unstructured(dummy_net.conv1, "weight", 2)
        prune.l1_unstructured(dummy_net.conv2, "weight", 3)
        assert pytest.approx(prune_info.ratio_to_amount(1.5), 0.000001) == 1.

    def test_summary(self, dummy_net, prune_info):
        prune.l1_unstructured(dummy_net.conv2, "weight", 3)
        print(prune_info.summary())
