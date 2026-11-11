"""Strict Pydantic v2 configuration schemas for tracklet models and training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from LUMIA.core.schemas import LumiaBaseModel

TemporalAggregatorType = Literal["masked_pool", "gru", "transformer"]
LossFunctionType = Literal["cross_entropy", "focal", "class_balanced_focal"]
PrimaryMetricType = Literal["val_map", "val_macro_f1", "val_ap_real"]
LRSchedulerType = Literal["cosine_decay", "cosine_annealing_warm_restarts"]


class TrackletModelConfig(LumiaBaseModel):
    """Configuration for TrackletClassifierNet spatial backbone and temporal aggregator."""

    spatial_backbone: str = Field(
        default="resnet_lite",
        description="Spatial feature extractor topology.",
    )
    spatial_embed_dim: int = Field(
        default=128,
        ge=32,
        le=512,
        description="Dimension of extracted spatial feature tokens per frame.",
    )
    temporal_aggregator: TemporalAggregatorType = Field(
        default="gru",
        description="Temporal sequence fusion mechanism ('masked_pool', 'gru', 'transformer').",
    )
    temporal_hidden_dim: int = Field(
        default=128,
        ge=32,
        le=512,
        description="Hidden layer dimension for recurrent or attention aggregators.",
    )
    temporal_layers: int = Field(
        default=2,
        ge=1,
        le=6,
        description="Number of recurrent or attention encoder layers.",
    )
    dropout_rate: float = Field(
        default=0.25,
        ge=0.0,
        le=0.8,
        description="Dropout rate for regularization in aggregator and head.",
    )
    norm_layer: Literal["group_norm", "batch_norm"] = Field(
        default="group_norm",
        description="Normalization layer for spatial backbone ('group_norm' or 'batch_norm').",
    )
    num_classes: int = Field(
        default=4,
        frozen=True,
        description="Number of target categories (Bogus, Point Source, Streak, Comet Coma).",
    )

    @field_validator("temporal_aggregator", mode="before")
    @classmethod
    def _normalize_aggregator(cls, v: Any) -> str:
        if isinstance(v, str):
            clean = v.lower().strip()
            if clean in ("bigru", "gru"):
                return "gru"
            if clean in ("masked_pool", "maskedpool", "pool"):
                return "masked_pool"
            if clean in ("transformer", "trans"):
                return "transformer"
        return v


class TrackletTrainingConfig(LumiaBaseModel):
    """Configuration for training run hyperparameters, dataset paths, and optimization."""

    dataset_path: Path = Field(
        default=Path("data/ground_truth/ztf_sso_tracklets_144x144.h5"),
        description="Path to CSR-indexed HDF5 training archive.",
    )
    model_output_dir: Path = Field(
        default=Path("data/models"),
        description="Destination directory for top-K model checkpoints.",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Tracklet batch size.",
    )
    epochs: int = Field(
        default=30,
        ge=1,
        le=200,
        description="Maximum training epochs.",
    )
    learning_rate: float = Field(
        default=3e-4,
        gt=0.0,
        le=1e-1,
        description="Base AdamW learning rate.",
    )
    weight_decay: float = Field(
        default=1e-3,
        ge=0.0,
        description="Weight decay for regularization.",
    )
    loss_function: LossFunctionType = Field(
        default="class_balanced_focal",
        description="Loss function formulation.",
    )
    focal_gamma: float = Field(
        default=2.0,
        ge=0.0,
        le=5.0,
        description="Focusing parameter for down-weighting easy examples.",
    )
    class_balanced_beta: float = Field(
        default=0.99,
        ge=0.9,
        lt=1.0,
        description="Hyperparameter beta for inverse effective sample weighting.",
    )
    primary_metric: PrimaryMetricType = Field(
        default="val_map",
        description="Primary validation metric for top-K checkpoint ranking.",
    )
    lr_scheduler_type: LRSchedulerType = Field(
        default="cosine_decay",
        description="Learning rate schedule (cosine decay or warm restarts SGDR).",
    )
    restart_t0_epochs: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of epochs in first SGDR cycle.",
    )
    restart_t_mult: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Cycle multiplication factor for SGDR.",
    )
    enable_temperature_scaling: bool = Field(
        default=True,
        description="Enable post-hoc Platt temperature scaling on validation logits.",
    )
    enable_threshold_optimization: bool = Field(
        default=True,
        description="Enable post-hoc per-class decision threshold tuning.",
    )
    warmup_epochs: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Number of linear warmup epochs before cosine decay.",
    )
    grad_clip_norm: float = Field(
        default=1.0,
        gt=0.0,
        description="Maximum gradient L2 norm for gradient clipping.",
    )
    use_amp: bool = Field(
        default=True,
        description="Enable Automatic Mixed Precision (AMP) where available.",
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        description="Number of DataLoader worker subprocesses.",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility.",
    )
    model: TrackletModelConfig = Field(
        default_factory=TrackletModelConfig,
        description="Model architecture sub-configuration.",
    )

    @field_validator("dataset_path", "model_output_dir", mode="before")
    @classmethod
    def _coerce_paths(cls, v: Any) -> Path:
        if isinstance(v, str):
            return Path(v)
        return v
