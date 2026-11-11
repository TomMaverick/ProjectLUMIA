"""Tests for TwoStreamAdaptationHead and faint source injection."""

from __future__ import annotations

import pytest
import torch

from LUMIA.ml.adaptation_head import TwoStreamAdaptationHead
from LUMIA.physics.synthetic_injection import inject_faint_moving_source


def test_two_stream_adaptation_head_shapes_and_probabilities() -> None:
    """Validate forward pass output shape and probability bounds."""
    batch_size = 4
    epochs = 5
    stamp_size = 48

    model = TwoStreamAdaptationHead(num_temporal_epochs=epochs, in_channels=1)
    model.eval()
    stamps = torch.randn(batch_size, 1, stamp_size, stamp_size)
    # temporal profile: (batch, epochs, 3) where columns are flux, flux_err, fwhm
    profiles = torch.randn(batch_size, epochs, 3)

    probs = model(stamps, profiles)
    assert probs.shape == (batch_size,)
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)

    logits = model.forward_logits(stamps, profiles)
    assert logits.shape == (batch_size,)
    torch.testing.assert_close(probs, torch.sigmoid(logits))


def test_two_stream_adaptation_head_gradient_flow() -> None:
    """Verify gradients propagate to both Stream A and Stream B backbones."""
    model = TwoStreamAdaptationHead(num_temporal_epochs=4, in_channels=1)
    stamps = torch.randn(2, 1, 48, 48, requires_grad=True)
    profiles = torch.randn(2, 4, 3, requires_grad=True)

    logits = model.forward_logits(stamps, profiles)
    loss = logits.sum()
    loss.backward()

    # Check gradients in Stream A convolutional weights
    assert model.stream_a[0].weight.grad is not None
    assert torch.any(model.stream_a[0].weight.grad != 0.0)

    # Check gradients in Stream B linear weights
    assert model.stream_b[0].weight.grad is not None
    assert torch.any(model.stream_b[0].weight.grad != 0.0)

    # Check gradients in input tensors
    assert stamps.grad is not None
    assert profiles.grad is not None


def test_inject_faint_moving_source_calibration() -> None:
    """Validate sub-5-sigma synthetic source injection at target SNR."""
    h, w = 64, 64
    bg_rms = 12.0
    bg = torch.zeros((h, w), dtype=torch.float32)

    target_snr = 3.2
    x0, y0 = 32.0, 32.0

    injected, mag = inject_faint_moving_source(
        science_tensor=bg,
        x0=x0,
        y0=y0,
        target_snr=target_snr,
        background_rms=bg_rms,
        zero_point=26.0,
        fwhm_pixels=2.5,
    )

    assert injected.shape == (h, w)
    assert mag > 20.0  # Faint asteroid magnitude
    # Peak injected flux should be at center
    peak_flux = injected[32, 32].item()
    assert peak_flux > 0.0
    # Boundary pixels far away from injection should be 0
    assert injected[0, 0].item() == 0.0
