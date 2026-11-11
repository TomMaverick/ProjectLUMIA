"""Two-stream neural adaptation head for sub-5-sigma candidate triage."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "TwoStreamAdaptationHead",
]


class TwoStreamAdaptationHead(nn.Module):
    """Dual-stream neural classifier evaluating spatial morphology and temporal consistency.

    Stream A processes the coadded spatial stamp to detect asteroid point-source morphology
    and reject optical artifacts (dipoles, bleeding columns, diffraction spikes).
    Stream B evaluates multi-epoch temporal profiles (flux, error, FWHM) to reject
    single-exposure transient noise spikes and cosmic rays.
    """

    def __init__(
        self,
        num_temporal_epochs: int = 5,
        in_channels: int = 1,
        hidden_dim: int = 128,
        temporal_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_temporal_epochs = num_temporal_epochs
        self.in_channels = in_channels

        # Stream A: Spatial Morphology (coadded stamp)
        self.stream_a = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, hidden_dim),
            nn.GELU(),
        )

        # Stream B: Temporal Flux Consistency (epochs x [flux, flux_err, fwhm])
        self.stream_b = nn.Sequential(
            nn.Linear(num_temporal_epochs * 3, temporal_dim),
            nn.GELU(),
            nn.Linear(temporal_dim, temporal_dim),
            nn.GELU(),
        )

        # Representation Fusion & Classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + temporal_dim, 64),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d | nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_logits(
        self,
        stamp: torch.Tensor,
        temporal_profile: torch.Tensor,
    ) -> torch.Tensor:
        """Returns unnormalized logits of shape (B,) for BCE-with-logits training."""
        if stamp.ndim == 3:
            stamp = stamp.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        if temporal_profile.ndim == 3:
            temporal_profile = temporal_profile.flatten(start_dim=1)

        feat_a = self.stream_a(stamp)
        feat_b = self.stream_b(temporal_profile)
        fused = torch.cat([feat_a, feat_b], dim=-1)
        logits = self.classifier(fused)
        return logits.squeeze(-1)

    def forward(
        self,
        stamp: torch.Tensor,
        temporal_profile: torch.Tensor,
    ) -> torch.Tensor:
        """Returns calibrated real-asteroid probabilities in [0, 1] of shape (B,)."""
        return torch.sigmoid(self.forward_logits(stamp, temporal_profile))
