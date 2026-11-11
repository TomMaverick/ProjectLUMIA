"""VGG-style and ResNet-Lite PyTorch backbones for Real/Bogus classification."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from LUMIA.ml.config import CNNClassifierConfig

INPUT_CHANNELS: int = 3  # Science, Reference, Difference


class RealBogusNet(nn.Module):
    """Base class: validated (B, 3, H, W) input, scalar logit head, sigmoid probability."""

    def __init__(self) -> None:
        super().__init__()
        self.features: nn.Module
        self.head: nn.Module

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        if x.ndim != 4:
            msg = f"Expected 4D input (B, 3, H, W), got shape {tuple(x.shape)}."
            raise ValueError(msg)
        if x.shape[1] != INPUT_CHANNELS:
            msg = (
                f"Expected {INPUT_CHANNELS} input channels (Science, Reference, "
                f"Difference), got {x.shape[1]}."
            )
            raise ValueError(msg)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw Real/Bogus logits of shape (B,) for BCE-with-logits training."""
        self._validate_input(x)
        pooled: torch.Tensor = self.features(x)
        logits: torch.Tensor = self.head(pooled)
        return logits.squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Real probabilities in [0, 1] of shape (B,) via sigmoid activation."""
        return torch.sigmoid(self.forward_logits(x))


def _conv_bn_relu(in_ch: int, out_ch: int) -> list[nn.Module]:
    return [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class VGGClassifier(RealBogusNet):
    """Compact VGG-style network: stacked 3x3 conv blocks, pooling, dense head."""

    def __init__(self, dropout_rate: float = 0.25) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = INPUT_CHANNELS
        for out_ch in (32, 64, 128):
            layers += _conv_bn_relu(in_ch, out_ch)
            layers += _conv_bn_relu(out_ch, out_ch)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = out_ch
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]

        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(in_ch, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 1),
        )
        self.apply(_init_weights)


class _BasicResidualBlock(nn.Module):
    """Two 3x3 convolutions with an identity or 1x1-projection shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut: nn.Module
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNetLiteClassifier(RealBogusNet):
    """Shallow residual network with skip connections and global average pooling."""

    def __init__(self, dropout_rate: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            *_conv_bn_relu(INPUT_CHANNELS, 32),
            _BasicResidualBlock(32, 32, stride=1),
            _BasicResidualBlock(32, 64, stride=2),
            _BasicResidualBlock(64, 128, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 1),
        )

        # 1. Standard-Initialisierung anwenden (setzt Linear.bias vorerst auf 0)
        self.apply(_init_weights)

        # 2. BIAS FIX FÜR EXTREME IMBALANCE:
        # 584 echte Transienten vs 511422 Rauschen
        prior_prob = 584.0 / 511422.0
        bias_init = float(np.log(prior_prob / (1.0 - prior_prob)))

        # Greife auf den nn.Linear Layer (Index 1 in self.head) zu und überschreibe den Bias
        nn.init.constant_(self.head[1].bias, bias_init)


def build_model(config: CNNClassifierConfig) -> RealBogusNet:
    """Instantiate the configured backbone with freshly initialized (seeded) weights."""
    if config.backbone == "vgg":
        return VGGClassifier(dropout_rate=config.dropout_rate)
    if config.backbone == "resnet_lite":
        return ResNetLiteClassifier(dropout_rate=config.dropout_rate)
    msg = f"Unsupported backbone '{config.backbone}'."
    raise ValueError(msg)