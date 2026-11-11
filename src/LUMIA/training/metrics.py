"""Domain-optimized validation metrics: PR-AUC, AP, Macro-F1, OvR-AUC, and operating points."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

CLASS_NAMES: dict[int, str] = {
    0: "BOGUS_ARTIFACT",
    1: "REAL_POINT_SOURCE",
    2: "REAL_STREAK",
    3: "REAL_COMET_COMA",
}

MIN_NEGATIVE_SAMPLES_FOR_FPR: int = 30


def compute_fpr90(
    y_true: np.ndarray,
    y_probs: np.ndarray,
) -> float:
    """Compute False Positive Rate at 90% Recall (TPR) for Real vs Bogus.

    Grouping:
        - Bogus: Class 0 (satellites, cosmic rays, subtraction artifacts)
        - Real: Classes 1, 2, 3 (Point Source, Streak, Comet Coma)
        - Score for Real: sum of predicted probabilities for classes 1, 2, 3.

    Args:
        y_true: 1D array of ground truth class indices in [0, 3].
        y_probs: 2D array of predicted class probabilities of shape (N, 4).

    Returns:
        FPR at the operating threshold where TPR >= 0.90.
    """
    y_binary = (y_true > 0).astype(int)
    if y_probs.ndim == 2 and y_probs.shape[1] > 1:
        p_real = np.sum(y_probs[:, 1:], axis=1)
    else:
        p_real = y_probs.squeeze()

    n_neg = int(np.sum(y_binary == 0))
    if n_neg < MIN_NEGATIVE_SAMPLES_FOR_FPR:
        logger.debug(
            "FPR90 evaluated on only %d negative samples (<%d). Reading has high variance.",
            n_neg,
            MIN_NEGATIVE_SAMPLES_FOR_FPR,
        )

    if np.sum(y_binary == 1) == 0 or n_neg == 0:
        return 0.0

    fpr, tpr, _ = roc_curve(y_binary, p_real)
    target_idx = np.where(tpr >= 0.90)[0]
    if len(target_idx) == 0:
        return 1.0
    return float(fpr[target_idx[0]])


def compute_purity_at_completeness(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    target_recall: float = 0.90,
) -> float:
    """Compute Precision (Purity) at a fixed Recall (Completeness) for Real vs Bogus.

    Args:
        y_true: 1D array of ground truth class indices.
        y_probs: 2D array of predicted class probabilities.
        target_recall: Target recall threshold (e.g. 0.90 or 0.95).

    Returns:
        Precision at target recall, or 0.0 if not reached.
    """
    y_binary = (y_true > 0).astype(int)
    if y_probs.ndim == 2 and y_probs.shape[1] > 1:
        p_real = np.sum(y_probs[:, 1:], axis=1)
    else:
        p_real = y_probs.squeeze()

    if np.sum(y_binary == 1) == 0 or np.sum(y_binary == 0) == 0:
        return 0.0

    precision, recall, _ = precision_recall_curve(y_binary, p_real)
    valid_indices = np.where(recall >= target_recall)[0]
    if len(valid_indices) == 0:
        return 0.0
    return float(np.max(precision[valid_indices]))


def compute_completeness_at_fpr(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    target_fpr: float = 0.01,
) -> float:
    """Compute Recall (Completeness) at a strict False Positive Rate budget.

    Args:
        y_true: 1D array of ground truth class indices.
        y_probs: 2D array of predicted class probabilities.
        target_fpr: Maximum allowable false positive rate (e.g. 0.01 or 0.05).

    Returns:
        Maximum Recall achievable within the FPR budget.
    """
    y_binary = (y_true > 0).astype(int)
    if y_probs.ndim == 2 and y_probs.shape[1] > 1:
        p_real = np.sum(y_probs[:, 1:], axis=1)
    else:
        p_real = y_probs.squeeze()

    if np.sum(y_binary == 1) == 0 or np.sum(y_binary == 0) == 0:
        return 0.0

    fpr, tpr, _ = roc_curve(y_binary, p_real)
    valid_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_indices) == 0:
        return 0.0
    return float(np.max(tpr[valid_indices]))


class TrackletMetricsTracker:
    """Tracker computing multiclass, PR-AUC, and astronomical operational metrics."""

    def __init__(self, num_classes: int = 4) -> None:
        self.num_classes = num_classes

    def compute_metrics(
        self,
        y_true: np.ndarray,
        logits: np.ndarray,
        thresholds: Sequence[float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute comprehensive multiclass and operational metrics.

        Args:
            y_true: 1D integer array of ground truth labels (N,).
            logits: 2D float array of predicted class logits (N, num_classes).
            thresholds: Optional per-class decision thresholds (num_classes,).

        Returns:
            Dictionary containing accuracy, macro_f1, map, ap_real, operating points, etc.
        """
        # Softmax probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Predictions (calibrated if thresholds supplied)
        if thresholds is not None:
            safe_tau = np.clip(np.array(thresholds, dtype=np.float64), 1e-3, 10.0)
            preds = np.argmax(probs / safe_tau, axis=-1)
        else:
            preds = np.argmax(probs, axis=-1)

        # 1. Accuracy
        accuracy = float(np.mean(preds == y_true))

        # 2. Per-class Precision, Recall, F1
        _, _, f1_per_class, _ = precision_recall_fscore_support(
            y_true,
            preds,
            labels=list(range(self.num_classes)),
            zero_division=0.0,  # type: ignore[arg-type]
        )
        macro_f1 = float(
            f1_score(
                y_true,
                preds,
                average="macro",
                zero_division=0.0,  # type: ignore[arg-type]
            )
        )

        per_class_f1_dict = {
            CLASS_NAMES.get(i, f"class_{i}"): float(f1_per_class[i])
            for i in range(self.num_classes)
        }

        # 3. Per-class Average Precision (AP) and Mean AP (mAP)
        ap_per_class: dict[str, float] = {}
        valid_aps: list[float] = []
        for i in range(self.num_classes):
            binary_target = (y_true == i).astype(int)
            class_key = CLASS_NAMES.get(i, f"class_{i}")
            if np.sum(binary_target == 1) > 0 and np.sum(binary_target == 0) > 0:
                try:
                    ap_val = float(average_precision_score(binary_target, probs[:, i]))
                    ap_per_class[class_key] = ap_val
                    valid_aps.append(ap_val)
                except ValueError:
                    ap_per_class[class_key] = 0.0
            else:
                ap_per_class[class_key] = 0.0

        mean_ap = float(np.mean(valid_aps)) if valid_aps else 0.0

        # 4. Binary Real vs Bogus Average Precision
        y_binary = (y_true > 0).astype(int)
        p_real = np.sum(probs[:, 1:], axis=1) if probs.shape[1] > 1 else probs.squeeze()
        if np.sum(y_binary == 1) > 0 and np.sum(y_binary == 0) > 0:
            ap_real = float(average_precision_score(y_binary, p_real))
        else:
            ap_real = 0.0

        # 5. Macro One-vs-Rest ROC-AUC
        roc_auc_per_class: dict[str, float] = {}
        valid_aucs: list[float] = []
        for i in range(self.num_classes):
            binary_target = (y_true == i).astype(int)
            class_key = CLASS_NAMES.get(i, f"class_{i}")
            if np.sum(binary_target == 1) > 0 and np.sum(binary_target == 0) > 0:
                try:
                    auc_val = float(roc_auc_score(binary_target, probs[:, i]))
                    roc_auc_per_class[class_key] = auc_val
                    valid_aucs.append(auc_val)
                except ValueError:
                    roc_auc_per_class[class_key] = 0.5
            else:
                roc_auc_per_class[class_key] = 0.5

        roc_auc_macro = float(np.mean(valid_aucs)) if valid_aucs else 0.5

        # 6. Operational Metrics
        fpr90 = compute_fpr90(y_true, probs)
        p_at_r90 = compute_purity_at_completeness(y_true, probs, target_recall=0.90)
        p_at_r95 = compute_purity_at_completeness(y_true, probs, target_recall=0.95)
        r_at_fpr01 = compute_completeness_at_fpr(y_true, probs, target_fpr=0.01)
        r_at_fpr05 = compute_completeness_at_fpr(y_true, probs, target_fpr=0.05)
        n_neg = int(np.sum(y_binary == 0))

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_class_f1": per_class_f1_dict,
            "map": mean_ap,
            "ap_per_class": ap_per_class,
            "ap_real": ap_real,
            "roc_auc_macro": roc_auc_macro,
            "roc_auc_per_class": roc_auc_per_class,
            "fpr90": fpr90,
            "p_at_r90": p_at_r90,
            "p_at_r95": p_at_r95,
            "r_at_fpr01": r_at_fpr01,
            "r_at_fpr05": r_at_fpr05,
            "num_negative_samples": n_neg,
            "fpr_reliable": bool(n_neg >= MIN_NEGATIVE_SAMPLES_FOR_FPR),
        }
