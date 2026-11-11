"""Unit tests for training configurations and CLI entry point."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from scripts.train_tracklets import parse_args

from LUMIA.training.config import TrackletModelConfig, TrackletTrainingConfig


class TestTrainingConfig:
    """Verify strict Pydantic v2 validation contracts for model and training configs."""

    def test_default_config_values_match_blueprint(self) -> None:
        """Default hyperparameters match technical specification in blueprint."""
        cfg = TrackletTrainingConfig()
        assert cfg.batch_size == 32
        assert cfg.epochs == 30
        assert cfg.learning_rate == pytest.approx(3e-4)
        assert cfg.loss_function == "class_balanced_focal"
        assert cfg.focal_gamma == pytest.approx(2.0)
        assert cfg.warmup_epochs == 3
        assert cfg.grad_clip_norm == pytest.approx(1.0)
        assert cfg.use_amp is True

        m_cfg = cfg.model
        assert m_cfg.spatial_backbone == "resnet_lite"
        assert m_cfg.spatial_embed_dim == 128
        assert m_cfg.temporal_aggregator == "gru"
        assert m_cfg.temporal_hidden_dim == 128
        assert m_cfg.temporal_layers == 2
        assert m_cfg.num_classes == 4

    def test_path_string_coercion_and_immutability(self) -> None:
        """String paths are cleanly coerced to Path objects, and config is immutable."""
        cfg = TrackletTrainingConfig(
            dataset_path="data/test.h5",  # type: ignore[arg-type]
            model_output_dir="data/out",  # type: ignore[arg-type]
        )
        assert isinstance(cfg.dataset_path, Path)
        assert isinstance(cfg.model_output_dir, Path)
        assert cfg.dataset_path == Path("data/test.h5")

        with pytest.raises(ValidationError):
            cfg.batch_size = 64  # type: ignore[misc]

    def test_rejects_extra_fields_and_out_of_bounds_parameters(self) -> None:
        """Pydantic v2 strictness rejects unknown fields and illegal values."""
        with pytest.raises(ValidationError):
            TrackletTrainingConfig(unknown_field=123)  # type: ignore[call-arg]

        with pytest.raises(ValidationError):
            TrackletTrainingConfig(batch_size=0)

        with pytest.raises(ValidationError):
            TrackletTrainingConfig(learning_rate=-0.01)

        with pytest.raises(ValidationError):
            TrackletModelConfig(dropout_rate=1.5)


class TestCLIArgumentParser:
    """Verify scripts/train_tracklets.py CLI argument parsing."""

    def test_cli_parser_defaults_and_overrides(self) -> None:
        """CLI parser correctly parses arguments and overrides config defaults."""
        args = parse_args(
            [
                "--dataset",
                "data/test.h5",
                "--output-dir",
                "data/models_test",
                "--aggregator",
                "transformer",
                "--batch-size",
                "16",
                "--epochs",
                "5",
                "--lr",
                "0.001",
                "--loss",
                "class_balanced_focal",
                "--workers",
                "2",
                "--device",
                "cpu",
                "--dry-run",
            ]
        )

        assert args.dataset == Path("data/test.h5")
        assert args.output_dir == Path("data/models_test")
        assert args.aggregator == "transformer"
        assert args.batch_size == 16
        assert args.epochs == 5
        assert args.lr == pytest.approx(0.001)
        assert args.loss == "class_balanced_focal"
        assert args.workers == 2
        assert args.device == "cpu"
        assert args.dry_run is True
