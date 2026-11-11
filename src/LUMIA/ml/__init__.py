"""Module 7 PyTorch CNN Real/Bogus classification architectures, runtime, and inference."""

from __future__ import annotations

from LUMIA.ml.classifier import (
    ClassificationResult,
    CutoutClassificationBatchResult,
    InferenceMemoryError,
    RealBogusClassifier,
    attach_classifications_to_detections,
    classify_cutout_batch,
    preprocess_triplets,
)
from LUMIA.ml.config import (
    BackboneType,
    CNNClassifierConfig,
    StampSize,
)
from LUMIA.ml.models import (
    INPUT_CHANNELS,
    RealBogusNet,
    ResNetLiteClassifier,
    VGGClassifier,
    build_model,
)
from LUMIA.ml.runtime import (
    configure_deterministic_torch,
    resolve_torch_device,
)

__all__ = [
    "BackboneType",
    "CNNClassifierConfig",
    "ClassificationResult",
    "CutoutClassificationBatchResult",
    "INPUT_CHANNELS",
    "InferenceMemoryError",
    "RealBogusClassifier",
    "RealBogusNet",
    "ResNetLiteClassifier",
    "StampSize",
    "VGGClassifier",
    "attach_classifications_to_detections",
    "build_model",
    "classify_cutout_batch",
    "configure_deterministic_torch",
    "preprocess_triplets",
    "resolve_torch_device",
]
