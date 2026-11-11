"""Model training, optimization, loss functions, and evaluation for Project LUMIA."""

from __future__ import annotations

from LUMIA.training.calibration import TemperatureScaler, ThresholdOptimizer
from LUMIA.training.checkpoint import TopKModelCheckpointer
from LUMIA.training.config import (
    LossFunctionType,
    LRSchedulerType,
    PrimaryMetricType,
    TemporalAggregatorType,
    TrackletModelConfig,
    TrackletTrainingConfig,
)
from LUMIA.training.loss import (
    ClassBalancedCrossEntropyLoss,
    ClassBalancedFocalLoss,
    FocalLoss,
    compute_class_balanced_weights,
)
from LUMIA.training.metrics import (
    TrackletMetricsTracker,
    compute_completeness_at_fpr,
    compute_fpr90,
    compute_purity_at_completeness,
)
from LUMIA.training.trainer import TrackletTrainer, extract_dataset_labels

__all__ = [
    "ClassBalancedCrossEntropyLoss",
    "ClassBalancedFocalLoss",
    "FocalLoss",
    "LRSchedulerType",
    "LossFunctionType",
    "PrimaryMetricType",
    "TemperatureScaler",
    "TemporalAggregatorType",
    "ThresholdOptimizer",
    "TopKModelCheckpointer",
    "TrackletMetricsTracker",
    "TrackletModelConfig",
    "TrackletTrainer",
    "TrackletTrainingConfig",
    "compute_class_balanced_weights",
    "compute_completeness_at_fpr",
    "compute_fpr90",
    "compute_purity_at_completeness",
    "extract_dataset_labels",
]
