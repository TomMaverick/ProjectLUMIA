"""Unit tests for validation metrics (FPR90, Macro-F1, OvR-AUC) and TopK checkpointer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import torch.nn as nn

from LUMIA.training.checkpoint import TopKModelCheckpointer
from LUMIA.training.config import TrackletTrainingConfig
from LUMIA.training.metrics import (
    TrackletMetricsTracker,
    compute_completeness_at_fpr,
    compute_fpr90,
    compute_purity_at_completeness,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestEvaluationMetrics:
    """Verify multiclass accuracy, Macro-F1, OvR-AUC, and FPR90 calculations."""

    def test_compute_fpr90_clean_separation(self) -> None:
        """When Real and Bogus are perfectly separated, FPR at 90% Recall is 0.0."""
        # 10 samples: 5 Bogus (class 0) with p_real ~ 0.05, 5 Real (class 1) with p_real ~ 0.95
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_probs = np.zeros((10, 4))
        y_probs[:5, 0] = 0.95
        y_probs[:5, 1] = 0.05
        y_probs[5:, 0] = 0.05
        y_probs[5:, 1] = 0.95

        fpr90 = compute_fpr90(y_true, y_probs)
        assert fpr90 == pytest.approx(0.0, abs=1e-5)

    def test_compute_fpr90_mixed_distribution(self) -> None:
        """Compute FPR90 on known binary probabilities."""
        # Bogus scores: [0.1, 0.2, 0.3, 0.4, 0.7] (one high false positive)
        # Real scores:  [0.5, 0.6, 0.8, 0.85, 0.9]
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 2, 3, 1])
        y_probs = np.zeros((10, 4))
        p_real = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.5, 0.6, 0.8, 0.85, 0.9])
        y_probs[:, 0] = 1.0 - p_real
        y_probs[:, 1] = p_real

        fpr90 = compute_fpr90(y_true, y_probs)
        assert 0.0 <= fpr90 <= 1.0

    def test_tracklet_metrics_tracker_computes_all_required_metrics(self) -> None:
        """TrackletMetricsTracker returns complete dictionary of validation metrics."""
        tracker = TrackletMetricsTracker(num_classes=4)

        # 40 synthetic samples with known distribution
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 4, size=40)
        # Generate random logits
        logits = rng.normal(0, 1, size=(40, 4))

        metrics = tracker.compute_metrics(y_true=y_true, logits=logits)

        expected_keys = {
            "accuracy",
            "macro_f1",
            "per_class_f1",
            "map",
            "ap_per_class",
            "ap_real",
            "roc_auc_macro",
            "fpr90",
            "p_at_r90",
            "p_at_r95",
            "r_at_fpr01",
            "r_at_fpr05",
            "fpr_reliable",
        }
        assert expected_keys.issubset(metrics.keys())
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert 0.0 <= metrics["map"] <= 1.0
        assert 0.0 <= metrics["ap_real"] <= 1.0
        assert 0.0 <= metrics["fpr90"] <= 1.0
        assert len(metrics["per_class_f1"]) == 4

    def test_compute_purity_at_completeness_and_completeness_at_fpr(self) -> None:
        """Verify operating points P@R and R@FPR."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_probs = np.zeros((10, 4))
        y_probs[:5, 0] = 0.9
        y_probs[:5, 1] = 0.1
        y_probs[5:, 0] = 0.1
        y_probs[5:, 1] = 0.9

        p_at_r90 = compute_purity_at_completeness(y_true, y_probs, target_recall=0.90)
        assert p_at_r90 == pytest.approx(1.0, abs=1e-3)

        r_at_fpr01 = compute_completeness_at_fpr(y_true, y_probs, target_fpr=0.01)
        assert r_at_fpr01 == pytest.approx(1.0, abs=1e-3)

    def test_tracklet_metrics_tracker_with_calibrated_thresholds(self) -> None:
        """Thresholds vector alters classification predictions appropriately."""
        tracker = TrackletMetricsTracker(num_classes=2)
        y_true = np.array([0, 1])
        # Logits where class 1 is marginally preferred for both samples
        logits = np.array([[0.4, 0.6], [0.4, 0.6]])

        # Uncalibrated: both predict 1 -> only 1 of 2 correct
        uncal = tracker.compute_metrics(y_true=y_true, logits=logits)
        assert uncal["accuracy"] == 0.5

        # Calibrated: lower threshold for class 0 (tau_0 = 0.5) shifts first prediction to 0
        cal = tracker.compute_metrics(
            y_true=y_true,
            logits=np.array([[0.55, 0.6], [0.1, 0.9]]),
            thresholds=[0.5, 1.0],
        )
        assert cal["accuracy"] == 1.0


class TestTopKModelCheckpointer:
    """Verify top-K checkpoint saving, rank ordering, and best-model management."""

    def test_checkpointer_saves_and_ranks_top_k(self, tmp_path: Path) -> None:
        """Checkpointer retains only top-K models ranked by validation Macro-F1."""
        ckpt_dir = tmp_path / "models"
        checkpointer = TopKModelCheckpointer(
            output_dir=ckpt_dir,
            k=3,
            metric_name="val_macro_f1",
            mode="max",
        )

        model = nn.Linear(10, 4)
        optimizer = torch.optim.Adam(model.parameters())
        cfg = TrackletTrainingConfig(model_output_dir=ckpt_dir)

        # Epoch 1: score 0.70 -> Rank 1
        saved1 = checkpointer.save_if_top_k(
            epoch=1,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            metrics={"val_macro_f1": 0.70, "val_loss": 0.5},
            config=cfg,
        )
        assert saved1 is True
        assert (ckpt_dir / "best_tracklet_net_rank1.pth").exists()
        assert (ckpt_dir / "tracklet_net_best.pth").exists()

        # Epoch 2: score 0.65 -> Rank 2
        saved2 = checkpointer.save_if_top_k(
            epoch=2,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            metrics={"val_macro_f1": 0.65, "val_loss": 0.6},
            config=cfg,
        )
        assert saved2 is True
        assert (ckpt_dir / "best_tracklet_net_rank2.pth").exists()

        # Epoch 3: score 0.85 -> New Rank 1!
        saved3 = checkpointer.save_if_top_k(
            epoch=3,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            metrics={"val_macro_f1": 0.85, "val_loss": 0.3},
            config=cfg,
        )
        assert saved3 is True

        # Checkpoint contents inspection
        best_data = torch.load(ckpt_dir / "tracklet_net_best.pth", weights_only=False)
        assert best_data["epoch"] == 3
        assert best_data["metrics"]["val_macro_f1"] == 0.85
        assert best_data["lumia_version"] == "1.0.0"
        assert "model_state_dict" in best_data
        assert "config" in best_data

        # Epoch 4: score 0.75 (beats 0.65) -> Rank 2, evicts 0.65
        saved4 = checkpointer.save_if_top_k(
            epoch=4,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            metrics={"val_macro_f1": 0.75, "val_loss": 0.4},
            config=cfg,
        )
        assert saved4 is True
        assert len(checkpointer.best_entries) == 3
        # Rank 1: 0.85 (epoch 3), Rank 2: 0.75 (epoch 4), Rank 3: 0.70 (epoch 1)
        assert checkpointer.best_entries[0][0] == 0.85
        assert checkpointer.best_entries[1][0] == 0.75
        assert checkpointer.best_entries[2][0] == 0.70

        # Epoch 5: score 0.50 (worse than all top 3) -> Not saved!
        saved5 = checkpointer.save_if_top_k(
            epoch=5,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            metrics={"val_macro_f1": 0.50, "val_loss": 0.9},
            config=cfg,
        )
        assert saved5 is False
