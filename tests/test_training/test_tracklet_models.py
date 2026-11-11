"""Unit tests for ResNetLite spatial backbone and TrackletClassifierNet architectures."""

from __future__ import annotations

import pytest
import torch

from LUMIA.models.backbones import ResNetLite
from LUMIA.models.tracklet_net import (
    BiGRUAggregator,
    MaskedPoolAggregator,
    TrackletClassifierNet,
    TransformerAggregator,
)
from LUMIA.training.config import TrackletModelConfig


class TestSpatialBackbone:
    """Verify ResNetLite feature extractor shapes and behavior."""

    def test_resnet_lite_forward_shape(self) -> None:
        """ResNetLite maps (M, 3, 128, 128) to (M, spatial_embed_dim)."""
        backbone = ResNetLite(in_channels=3, spatial_embed_dim=128)
        x = torch.randn(4, 3, 128, 128)
        out = backbone(x)
        assert out.shape == (4, 128)
        assert torch.all(torch.isfinite(out))

    def test_resnet_lite_custom_embed_dim(self) -> None:
        """ResNetLite supports custom projection embedding dimension."""
        backbone = ResNetLite(in_channels=3, spatial_embed_dim=64)
        x = torch.randn(2, 3, 128, 128)
        out = backbone(x)
        assert out.shape == (2, 64)

    def test_resnet_lite_rejects_invalid_dimensions(self) -> None:
        """ResNetLite raises ValueError on non-4D input."""
        backbone = ResNetLite()
        with pytest.raises(ValueError, match="Expected 4D input"):
            backbone(torch.randn(3, 128, 128))


class TestTemporalAggregators:
    """Verify swappable temporal aggregators with variable padding masks."""

    @pytest.mark.parametrize(
        ("agg_cls", "kwargs"),
        [
            (MaskedPoolAggregator, {"spatial_embed_dim": 128}),
            (
                BiGRUAggregator,
                {
                    "spatial_embed_dim": 128,
                    "hidden_dim": 128,
                    "num_layers": 2,
                },
            ),
            (
                TransformerAggregator,
                {
                    "spatial_embed_dim": 128,
                    "hidden_dim": 128,
                    "num_layers": 2,
                    "num_heads": 4,
                },
            ),
        ],
    )
    def test_aggregator_forward_shapes_and_finiteness(
        self,
        agg_cls: type,
        kwargs: dict[str, object],
    ) -> None:
        """Each aggregator ingests (B, N_max, D) + mask (B, N_max) -> (B, D_out)."""
        agg = agg_cls(**kwargs)
        batch_size = 4
        max_seq_len = 6
        tokens = torch.randn(batch_size, max_seq_len, 128)
        mask = torch.tensor(
            [
                [True, True, True, True, True, True],
                [True, True, True, False, False, False],
                [True, True, True, True, False, False],
                [True, True, False, False, False, False],
            ],
            dtype=torch.bool,
        )

        out = agg(tokens, mask)
        assert out.shape == (batch_size, agg.output_dim)
        assert torch.all(torch.isfinite(out))

    def test_masked_pool_ignores_padded_tokens(self) -> None:
        """MaskedPoolAggregator output is strictly invariant to values in padded slots."""
        agg = MaskedPoolAggregator(spatial_embed_dim=32)
        tokens = torch.randn(2, 5, 32)
        mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, False, False, False],
            ],
            dtype=torch.bool,
        )

        out1 = agg(tokens, mask)

        # Corrupt padded positions with huge values
        tokens_corrupted = tokens.clone()
        tokens_corrupted[0, 3:] = 9999.0
        tokens_corrupted[1, 2:] = -9999.0

        out2 = agg(tokens_corrupted, mask)
        torch.testing.assert_close(out1, out2, rtol=1e-5, atol=1e-5)


class TestTrackletClassifierNet:
    """Verify integrated TrackletClassifierNet forward pass and gradient flow."""

    @pytest.mark.parametrize("agg_type", ["masked_pool", "gru", "transformer"])
    def test_end_to_end_forward_pass_and_logits(self, agg_type: str) -> None:
        """TrackletClassifierNet yields (B, 4) logits across all aggregators."""
        cfg = TrackletModelConfig(
            spatial_backbone="resnet_lite",
            spatial_embed_dim=128,
            temporal_aggregator=agg_type,  # type: ignore[arg-type]
            temporal_hidden_dim=128,
            temporal_layers=2,
            dropout_rate=0.2,
            num_classes=4,
        )
        model = TrackletClassifierNet(cfg)

        batch_size = 2
        max_seq_len = 4
        tracklets = torch.randn(batch_size, max_seq_len, 3, 128, 128)
        padding_mask = torch.tensor(
            [
                [True, True, True, True],
                [True, True, False, False],
            ],
            dtype=torch.bool,
        )

        logits = model(tracklets, padding_mask)
        assert logits.shape == (batch_size, 4)
        assert torch.all(torch.isfinite(logits))

    @pytest.mark.parametrize("agg_type", ["masked_pool", "gru", "transformer"])
    def test_gradient_backpropagation(self, agg_type: str) -> None:
        """Gradients propagate through spatial backbone and temporal aggregator."""
        cfg = TrackletModelConfig(
            spatial_embed_dim=64,
            temporal_aggregator=agg_type,  # type: ignore[arg-type]
            temporal_hidden_dim=64,
            temporal_layers=1,
            num_classes=4,
        )
        model = TrackletClassifierNet(cfg)
        model.train()

        tracklets = torch.randn(2, 3, 3, 128, 128, requires_grad=True)
        padding_mask = torch.tensor(
            [[True, True, True], [True, True, False]],
            dtype=torch.bool,
        )
        targets = torch.tensor([1, 2], dtype=torch.long)

        logits = model(tracklets, padding_mask)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()

        assert tracklets.grad is not None
        assert torch.any(tracklets.grad != 0.0)

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Missing grad for {name}"
                assert torch.all(torch.isfinite(param.grad)), f"Non-finite grad for {name}"
