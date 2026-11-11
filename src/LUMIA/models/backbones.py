"""Spatial convolutional backbones for per-epoch astronomical cutout feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_norm_relu(
    in_ch: int,
    out_ch: int,
    stride: int = 1,
    norm_layer: str = "group_norm",
) -> list[nn.Module]:
    """Helper constructing Conv2d -> Norm -> ReLU."""
    norm_mod: nn.Module
    if norm_layer == "group_norm":
        if out_ch % 8 != 0:
            raise ValueError(f"Number of channels ({out_ch}) must be divisible by 8 for GroupNorm.")
        norm_mod = nn.GroupNorm(num_groups=8, num_channels=out_ch)
    elif norm_layer == "batch_norm":
        norm_mod = nn.BatchNorm2d(out_ch)
    else:
        raise ValueError(
            f"Unsupported norm_layer '{norm_layer}'. Expected 'group_norm' or 'batch_norm'."
        )

    return [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        norm_mod,
        nn.ReLU(inplace=True),
    ]


class BasicResidualBlock(nn.Module):
    """Two 3x3 convolutions with residual skip connection."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        norm_layer: str = "group_norm",
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        if norm_layer == "group_norm":
            if out_ch % 8 != 0:
                raise ValueError(
                    f"Number of channels ({out_ch}) must be divisible by 8 for GroupNorm."
                )
            self.bn1: nn.Module = nn.GroupNorm(num_groups=8, num_channels=out_ch)
            self.bn2: nn.Module = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        elif norm_layer == "batch_norm":
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.bn2 = nn.BatchNorm2d(out_ch)
        else:
            raise ValueError(
                f"Unsupported norm_layer '{norm_layer}'. Expected 'group_norm' or 'batch_norm'."
            )

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.shortcut: nn.Module
        if stride != 1 or in_ch != out_ch:
            shortcut_norm = (
                nn.GroupNorm(num_groups=8, num_channels=out_ch)
                if norm_layer == "group_norm"
                else nn.BatchNorm2d(out_ch)
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                shortcut_norm,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ResNetLite(nn.Module):
    """Shallow 3-stage residual backbone extracting spatial embedding tokens."""

    def __init__(
        self,
        in_channels: int = 3,
        spatial_embed_dim: int = 128,
        norm_layer: str = "group_norm",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.spatial_embed_dim = spatial_embed_dim
        self.norm_layer = norm_layer

        # Initial Conv-Norm-ReLU (3 -> 32)
        self.init_conv = nn.Sequential(
            *_conv_norm_relu(in_channels, 32, stride=1, norm_layer=norm_layer)
        )

        # 3 Residual Stages: 32 -> 32, 32 -> 64 (s=2), 64 -> 128 (s=2)
        self.stage1 = BasicResidualBlock(32, 32, stride=1, norm_layer=norm_layer)
        self.stage2 = BasicResidualBlock(32, 64, stride=2, norm_layer=norm_layer)
        self.stage3 = BasicResidualBlock(64, 128, stride=2, norm_layer=norm_layer)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        if spatial_embed_dim != 128:
            self.proj: nn.Module = nn.Linear(128, spatial_embed_dim)
        else:
            self.proj = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial embeddings from (M, C, H, W) frames."""
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (M, C, H, W), got shape {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}.")

        feat = self.init_conv(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        pooled = self.pool(feat)
        flat = self.flatten(pooled)
        out: torch.Tensor = self.proj(flat)
        return out
