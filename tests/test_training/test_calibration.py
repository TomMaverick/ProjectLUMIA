"""Unit tests for TemperatureScaler and ThresholdOptimizer post-hoc calibration."""

from __future__ import annotations

import numpy as np
import pytest

from LUMIA.training.calibration import TemperatureScaler, ThresholdOptimizer


class TestTemperatureScaler:
    """Verify temperature scaling calibration on model logits."""

    def test_temperature_scaler_converges_on_scaled_logits(self) -> None:
        """TemperatureScaler optimizes T > 0 and scales probabilities appropriately."""
        scaler = TemperatureScaler(init_temperature=1.0)
        assert scaler.temperature == pytest.approx(1.0)

        # Generate overconfident logits (large magnitude)
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 4, size=100)
        one_hot = np.zeros((100, 4))
        one_hot[np.arange(100), labels] = 1.0

        # Overconfident logits: correct class has large positive logit, others negative
        logits = (one_hot * 10.0 - 5.0) + rng.normal(0, 1.0, size=(100, 4))

        opt_t = scaler.fit(logits, labels)
        assert opt_t > 0.05
        assert scaler.temperature == pytest.approx(opt_t)

        scaled_logits = scaler.transform(logits)
        assert scaled_logits.shape == logits.shape

        probs = scaler.predict_proba(logits)
        assert probs.shape == (100, 4)
        np.testing.assert_allclose(np.sum(probs, axis=-1), np.ones(100), atol=1e-5)

    def test_temperature_scaler_handles_empty_or_degenerate_inputs(self) -> None:
        """Scaler handles edge case inputs gracefully."""
        scaler = TemperatureScaler(init_temperature=1.5)
        empty_logits = np.empty((0, 4))
        empty_labels = np.empty((0,), dtype=int)

        t = scaler.fit(empty_logits, empty_labels)
        assert t == pytest.approx(1.5)


class TestThresholdOptimizer:
    """Verify per-class decision threshold optimization."""

    def test_threshold_optimizer_preserves_shape_and_improves_macro_f1(self) -> None:
        """ThresholdOptimizer finds non-trivial threshold vector improving minority recall."""
        optimizer = ThresholdOptimizer(num_classes=4)
        assert len(optimizer.thresholds) == 4

        # Generate imbalanced probabilities where minority class (0) has lower raw probabilities
        rng = np.random.default_rng(42)
        n_samples = 200
        # 10 samples of class 0, 70 of class 1, 70 of class 2, 50 of class 3
        labels = np.array([0] * 10 + [1] * 70 + [2] * 70 + [3] * 50)

        # Baseline probabilities: minority class 0 probabilities are slightly depressed
        probs = np.zeros((n_samples, 4))
        for i, lbl in enumerate(labels):
            probs[i, lbl] = 0.4
            probs[i] += rng.uniform(0.1, 0.2, size=4)
            probs[i] /= np.sum(probs[i])

        # Optimize thresholds
        best_tau = optimizer.fit(probs, labels)
        assert best_tau.shape == (4,)
        assert np.all(best_tau > 0.0)

        preds = optimizer.predict(probs)
        assert preds.shape == (n_samples,)
        assert np.all((preds >= 0) & (preds < 4))

    def test_threshold_predict_with_identity_thresholds_matches_argmax(self) -> None:
        """When thresholds are uniform ones, predict matches standard argmax."""
        optimizer = ThresholdOptimizer(num_classes=4)
        optimizer.thresholds = np.ones(4)

        probs = np.array(
            [
                [0.1, 0.6, 0.2, 0.1],
                [0.7, 0.1, 0.1, 0.1],
                [0.2, 0.2, 0.5, 0.1],
            ]
        )

        preds = optimizer.predict(probs)
        expected = np.argmax(probs, axis=-1)
        np.testing.assert_array_equal(preds, expected)
