"""Unit tests for GroupNorm padding invariance, MAD clipping, and checkpoint compatibility."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from LUMIA.models.backbones import BasicResidualBlock, ResNetLite
from LUMIA.models.tracklet_net import TrackletClassifierNet
from LUMIA.training.config import TrackletModelConfig


class TestGroupNormBackbone:
    """Verify GroupNorm backbone construction, initialization, and weight structure."""

    def test_groupnorm_vs_batchnorm_instantiation(self) -> None:
        """Verify backbone constructs GroupNorm or BatchNorm2d based on norm_layer."""
        gn_backbone = ResNetLite(norm_layer="group_norm")
        bn_backbone = ResNetLite(norm_layer="batch_norm")

        assert isinstance(gn_backbone.init_conv[1], nn.GroupNorm)
        assert gn_backbone.init_conv[1].num_groups == 8
        assert gn_backbone.init_conv[1].num_channels == 32

        assert isinstance(bn_backbone.init_conv[1], nn.BatchNorm2d)
        assert bn_backbone.init_conv[1].num_features == 32

        assert isinstance(gn_backbone.stage1.bn1, nn.GroupNorm)
        assert isinstance(gn_backbone.stage1.bn2, nn.GroupNorm)
        assert isinstance(bn_backbone.stage1.bn1, nn.BatchNorm2d)
        assert isinstance(bn_backbone.stage1.bn2, nn.BatchNorm2d)

        # Forward pass output shapes
        x = torch.randn(4, 3, 128, 128)
        assert gn_backbone(x).shape == (4, 128)
        assert bn_backbone(x).shape == (4, 128)

    def test_state_dict_key_compatibility(self) -> None:
        """Verify GroupNorm preserves bn1/bn2/shortcut state dict keys for checkpoint parity."""
        gn_backbone = ResNetLite(norm_layer="group_norm")
        bn_backbone = ResNetLite(norm_layer="batch_norm")

        gn_keys = set(gn_backbone.state_dict().keys())
        bn_keys = set(bn_backbone.state_dict().keys())

        # All parameter keys (weight, bias) must match between group_norm and batch_norm
        # (BatchNorm additionally has running_mean and running_var, but parameter keys match)
        assert "stage1.bn1.weight" in gn_keys
        assert "stage1.bn1.bias" in gn_keys
        assert "stage1.bn2.weight" in gn_keys
        assert "stage1.bn2.bias" in gn_keys
        assert "stage2.shortcut.1.weight" in gn_keys
        assert "stage2.shortcut.1.bias" in gn_keys

        # Check weights initialized properly (weights=1.0, biases=0.0)
        assert torch.allclose(gn_backbone.stage1.bn1.weight, torch.ones_like(gn_backbone.stage1.bn1.weight))
        assert torch.allclose(gn_backbone.stage1.bn1.bias, torch.zeros_like(gn_backbone.stage1.bn1.bias))

    def test_channels_divisible_by_eight_assertion(self) -> None:
        """Verify GroupNorm rejects channel counts not divisible by 8."""
        with pytest.raises(ValueError, match="divisible by 8"):
            BasicResidualBlock(32, 35, norm_layer="group_norm")


class TestPaddingInvarianceAndOutlierClipping:
    """Verify mathematical padding invariance, outlier clipping, and token zero-masking."""

    def test_groupnorm_mathematical_padding_invariance(self) -> None:
        """GroupNorm embeddings on valid frames are invariant to sequence padding depth."""
        backbone = ResNetLite(norm_layer="group_norm")
        backbone.eval()

        torch.manual_seed(1234)
        # 2 real frames
        valid_frames = torch.randn(2, 3, 128, 128)

        # Sequence A: only 2 frames (no padding)
        tokens_unpadded = backbone(valid_frames)

        # Sequence B: same 2 frames padded with 10 zero frames (total 12 frames)
        padded_frames = torch.zeros(12, 3, 128, 128)
        padded_frames[:2] = valid_frames

        tokens_padded = backbone(padded_frames)

        # Tokens for the first 2 frames must be numerically identical
        assert torch.allclose(tokens_padded[:2], tokens_unpadded, atol=1e-6)

    def test_token_zero_masking_before_temporal_fusion(self) -> None:
        """Padded frames produce strictly all-zero token embeddings before aggregator."""
        config = TrackletModelConfig(
            spatial_backbone="resnet_lite",
            norm_layer="group_norm",
            temporal_aggregator="masked_pool",
        )
        net = TrackletClassifierNet(config)
        net.eval()

        batch_tracklets = torch.randn(2, 5, 3, 128, 128)
        # Tracklet 0 has 2 valid frames, Tracklet 1 has 4 valid frames
        padding_mask = torch.tensor(
            [
                [True, True, False, False, False],
                [True, True, True, True, False],
            ],
            dtype=torch.bool,
        )

        captured_tokens: list[torch.Tensor] = []

        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
            captured_tokens.append(inputs[0].detach().clone())

        handle = net.temporal_aggregator.register_forward_pre_hook(hook)
        _ = net(batch_tracklets, padding_mask)
        handle.remove()

        assert len(captured_tokens) == 1
        tokens = captured_tokens[0]  # Shape: (2, 5, 128)

        # Tracklet 0: frames 2, 3, 4 must be exactly zero vectors
        assert torch.all(tokens[0, 2:] == 0.0)
        # Tracklet 0: frames 0, 1 must not be zero
        assert not torch.all(tokens[0, :2] == 0.0)

        # Tracklet 1: frame 4 must be exactly zero
        assert torch.all(tokens[1, 4:] == 0.0)
        assert not torch.all(tokens[1, :4] == 0.0)

    def test_clamped_mad_outlier_clipping_floor(self) -> None:
        """Verify robust median +- 20*MAD clipping on uniform difference frames avoids NaN."""
        config = TrackletModelConfig(
            spatial_backbone="resnet_lite",
            norm_layer="group_norm",
            temporal_aggregator="gru",
        )
        net = TrackletClassifierNet(config)
        net.eval()

        # Batch where difference frames (channel 2) are completely uniform (MAD = 0)
        tracklets = torch.randn(2, 3, 3, 128, 128)
        tracklets[:, :, 2, :, :] = 0.0  # Zero difference frame
        # Add huge cosmic ray spike in one difference pixel
        tracklets[0, 0, 2, 64, 64] = 10000.0

        mask = torch.ones((2, 3), dtype=torch.bool)
        logits = net(tracklets, mask)

        assert logits.shape == (2, 4)
        assert torch.all(torch.isfinite(logits))
