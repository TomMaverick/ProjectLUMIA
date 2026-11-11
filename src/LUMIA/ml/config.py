"""Pydantic configuration for the Module 7 CNN Real/Bogus classifier."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - required at runtime by Pydantic
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from LUMIA.core.runtime import DEVICE_SPEC_PATTERN

BackboneType = Literal["vgg", "resnet_lite"]
StampSize = Literal[31, 51]


class CNNClassifierConfig(BaseModel):
    """Validated architecture, inference, determinism, and hardware parameters."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    backbone: BackboneType = Field(
        default="vgg",
        description="Feature-extractor topology: 'vgg' or 'resnet_lite'.",
    )
    stamp_size: StampSize = Field(
        default=31,
        description="Odd, square cutout side length in pixels (31 or 51).",
    )
    dropout_rate: float = Field(
        default=0.25,
        ge=0.0,
        lt=1.0,
        description="Dropout probability applied in the classification head.",
    )
    inference_batch_size: int = Field(
        default=256,
        ge=1,
        description="Maximum number of triplets per VRAM-bounded mini-batch.",
    )
    decision_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Scores at or above this value are flagged as Real.",
    )
    model_version: str = Field(
        default="lumia-rb-untrained-0.1",
        min_length=1,
        description="Model version identifier recorded for downstream provenance.",
    )
    checkpoint_path: Path | None = Field(
        default=None,
        description="Optional state_dict checkpoint; random weights are used if None.",
    )
    device: str = Field(
        default="auto",
        pattern=DEVICE_SPEC_PATTERN,
        description="Execution device: 'cpu', 'cuda', 'cuda:N' or 'auto'.",
    )
    num_cpu_threads: int | None = Field(
        default=None,
        ge=1,
        description="Optional limit on intra-op CPU threads.",
    )
    deterministic: bool = Field(
        default=True,
        description="Enforce torch.use_deterministic_algorithms(True).",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Seed for Python, NumPy, and PyTorch RNGs.",
    )
