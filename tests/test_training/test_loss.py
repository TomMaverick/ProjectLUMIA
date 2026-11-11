"""Unit tests for ClassBalancedFocalLoss and imbalance mitigation strategies."""

from __future__ import annotations

import pytest
import torch

from LUMIA.training.loss import (
    ClassBalancedCrossEntropyLoss,
    ClassBalancedFocalLoss,
    FocalLoss,
    compute_class_balanced_weights,
)


class TestClassBalancedWeights:
    """Verify Cui et al. effective number sample weighting formulation."""

    def test_effective_weights_formula(self) -> None:
        """Rare classes receive higher effective sample weights than dominant ones."""
        # Dominant point sources (8000) vs rare comets (20)
        counts = [1500, 8000, 300, 20]
        weights = compute_class_balanced_weights(counts, beta=0.9999)
        assert len(weights) == 4
        # Normalization: average weight is 1.0 (sum is C = 4)
        assert pytest.approx(weights.sum().item(), rel=1e-4) == 4.0
        # Rarest class must have highest weight
        assert weights[3] > weights[0]
        assert weights[3] > weights[1]
        assert weights[1] < weights[2]

    def test_uniform_counts_produce_equal_weights(self) -> None:
        """Equal class counts yield uniform weights."""
        counts = [500, 500, 500, 500]
        weights = compute_class_balanced_weights(counts, beta=0.999)
        torch.testing.assert_close(weights, torch.ones(4), rtol=1e-4, atol=1e-4)

    def test_zero_count_class_handled_gracefully(self) -> None:
        """Class with 0 samples does not divide by zero."""
        counts = [100, 0, 50, 10]
        weights = compute_class_balanced_weights(counts, beta=0.999)
        assert torch.all(torch.isfinite(weights))
        assert weights[1] > 0.0


class TestFocalAndClassBalancedLoss:
    """Verify ClassBalancedFocalLoss forward pass, reductions, and focal modulation."""

    def test_focal_loss_dampens_easy_examples(self) -> None:
        """Focal loss with gamma > 0 penalizes well-classified samples less than CE."""
        # Easy example: logits strongly favoring true class 1
        logits_easy = torch.tensor([[-2.0, 5.0, -2.0, -2.0]])
        targets = torch.tensor([1])

        loss_ce = FocalLoss(gamma=0.0)(logits_easy, targets)
        loss_focal = FocalLoss(gamma=2.0)(logits_easy, targets)

        # High confidence in true class -> (1 - p)^gamma << 1 -> lower loss
        assert loss_focal.item() < loss_ce.item()

    def test_class_balanced_focal_loss_reductions(self) -> None:
        """Loss supports 'mean', 'sum', and 'none' reduction modes."""
        counts = [1000, 5000, 200, 50]
        logits = torch.randn(4, 4, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        loss_none = ClassBalancedFocalLoss(class_counts=counts, reduction="none")(logits, targets)
        loss_mean = ClassBalancedFocalLoss(class_counts=counts, reduction="mean")(logits, targets)
        loss_sum = ClassBalancedFocalLoss(class_counts=counts, reduction="sum")(logits, targets)

        assert loss_none.shape == (4,)
        assert pytest.approx(loss_mean.item(), rel=1e-5) == loss_none.mean().item()
        assert pytest.approx(loss_sum.item(), rel=1e-5) == loss_none.sum().item()

    def test_loss_backward_gradient_flow(self) -> None:
        """ClassBalancedFocalLoss propagates finite gradients to input logits."""
        counts = [500, 2000, 100, 25]
        loss_fn = ClassBalancedFocalLoss(class_counts=counts, beta=0.9999, gamma=2.0)

        logits = torch.randn(8, 4, requires_grad=True)
        targets = torch.randint(0, 4, (8,), dtype=torch.long)

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert torch.all(torch.isfinite(logits.grad))
        assert torch.any(logits.grad != 0.0)

    def test_class_balanced_cross_entropy_matches_gamma_zero(self) -> None:
        """ClassBalancedCrossEntropyLoss equals ClassBalancedFocalLoss with gamma=0."""
        counts = [100, 300, 50, 10]
        logits = torch.randn(4, 4)
        targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        loss_cb_ce = ClassBalancedCrossEntropyLoss(class_counts=counts)(logits, targets)
        loss_cb_focal0 = ClassBalancedFocalLoss(class_counts=counts, gamma=0.0)(logits, targets)

        torch.testing.assert_close(loss_cb_ce, loss_cb_focal0, rtol=1e-5, atol=1e-5)
