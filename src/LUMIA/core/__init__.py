"""Shared canonical Pydantic schemas and deterministic PyTorch runtime utilities."""

from __future__ import annotations

from LUMIA.core.concurrency import (
    ResourceBudgetConfig,
    configure_global_runtime,
    get_active_resource_budget,
)
from LUMIA.core.runtime import (
    DEVICE_SPEC_PATTERN,
    configure_deterministic_torch,
    resolve_torch_device,
)
from LUMIA.core.schemas import (
    CUTOUT_TRIPLET_CHANNELS,
    CandidateDetection,
    ClassificationResult,
    CutoutBatchResult,
    CutoutStampSize,
    GroundTruthBatch,
    KnownObjectEphemeris,
    SNRPeakCandidate,
)

__all__ = [
    "CUTOUT_TRIPLET_CHANNELS",
    "CandidateDetection",
    "ClassificationResult",
    "CutoutBatchResult",
    "CutoutStampSize",
    "DEVICE_SPEC_PATTERN",
    "GroundTruthBatch",
    "KnownObjectEphemeris",
    "ResourceBudgetConfig",
    "SNRPeakCandidate",
    "configure_deterministic_torch",
    "configure_global_runtime",
    "get_active_resource_budget",
    "resolve_torch_device",
]
