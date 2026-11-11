"""Post-processing calibration: Temperature Scaling and Multi-Class Decision Thresholds."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import f1_score

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class TemperatureScaler:
    """Post-hoc logit calibration using Platt/Guo temperature scaling.

    Optimizes a single scalar parameter T > 0 on validation logits by minimizing
    cross-entropy negative log-likelihood (NLL).
    """

    def __init__(self, init_temperature: float = 1.0) -> None:
        self.temperature: float = max(1e-2, float(init_temperature))

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Fit optimal temperature parameter on validation set.

        Args:
            logits: Uncalibrated predicted logits of shape (N, C).
            labels: Ground truth class integer indices of shape (N,).

        Returns:
            Optimized temperature scalar T.
        """
        if logits.ndim != 2 or labels.ndim != 1:
            raise ValueError(
                f"Expected logits (N, C) and labels (N,), got {logits.shape} and {labels.shape}."
            )
        if len(logits) == 0:
            return self.temperature

        num_samples = len(labels)
        one_hot = np.zeros_like(logits)
        one_hot[np.arange(num_samples), labels] = 1.0

        def _nll_loss(temp: float) -> float:
            scaled = logits / max(temp, 1e-3)
            # Numerically stable log-softmax
            max_s = np.max(scaled, axis=-1, keepdims=True)
            log_sum_exp = max_s + np.log(np.sum(np.exp(scaled - max_s), axis=-1, keepdims=True))
            log_probs = scaled - log_sum_exp
            nll = -np.mean(np.sum(one_hot * log_probs, axis=-1))
            return float(nll)

        res = minimize_scalar(
            _nll_loss,
            bounds=(0.05, 10.0),
            method="bounded",
            options={"xatol": 1e-3, "maxiter": 100},
        )
        if res.success:
            self.temperature = float(res.x)
            logger.info("Temperature scaling converged to T = %.4f", self.temperature)
        else:
            logger.warning(
                "Temperature scaling optimization did not converge. Keeping T = %.4f",
                self.temperature,
            )

        return self.temperature

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Scale logits by learned temperature."""
        return logits / self.temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Convert scaled logits to well-calibrated posterior probabilities."""
        scaled = self.transform(logits)
        max_s = np.max(scaled, axis=-1, keepdims=True)
        exp_s = np.exp(scaled - max_s)
        return exp_s / np.sum(exp_s, axis=-1, keepdims=True)


class ThresholdOptimizer:
    """Optimizes per-class decision thresholds to maximize macro-averaged F1."""

    def __init__(
        self,
        num_classes: int = 4,
        metric: Literal["macro_f1"] = "macro_f1",
    ) -> None:
        self.num_classes = num_classes
        self.metric = metric
        self.thresholds: np.ndarray = np.ones(num_classes, dtype=np.float64)

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        init_thresholds: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Find optimal class-specific thresholds maximizing validation Macro-F1.

        Args:
            probs: Posterior probability matrix of shape (N, C).
            labels: Ground truth class integer indices of shape (N,).
            init_thresholds: Optional starting threshold vector of length C.

        Returns:
            Optimal threshold array of shape (C,).
        """
        if probs.ndim != 2 or probs.shape[1] != self.num_classes:
            raise ValueError(f"Expected probs of shape (N, {self.num_classes}), got {probs.shape}.")

        if init_thresholds is not None:
            x0 = np.array(init_thresholds, dtype=np.float64)
        else:
            x0 = np.ones(self.num_classes, dtype=np.float64)

        def _objective(tau: np.ndarray) -> float:
            safe_tau = np.clip(tau, 1e-3, 10.0)
            scaled = probs / safe_tau
            preds = np.argmax(scaled, axis=-1)
            score = f1_score(labels, preds, average="macro", zero_division=0.0)
            return -float(score)

        bounds = [(0.01, 2.0) for _ in range(self.num_classes)]
        res = minimize(
            _objective,
            x0=x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": 500, "xatol": 1e-2, "fatol": 1e-3},
        )

        if res.success or res.fun < _objective(x0):
            self.thresholds = np.clip(res.x, 1e-3, 10.0)
            logger.info(
                "Threshold optimizer reached objective F1=%.4f with tau=%s",
                -res.fun,
                np.round(self.thresholds, 4),
            )
        else:
            self.thresholds = x0
            logger.warning("Threshold optimization failed to improve F1. Preserving baseline.")

        return self.thresholds

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Predict class indices using learned per-class thresholds."""
        safe_tau = np.clip(self.thresholds, 1e-3, 10.0)
        scaled = probs / safe_tau
        return np.argmax(scaled, axis=-1)
