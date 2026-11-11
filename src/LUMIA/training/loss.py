"""Loss functions and class imbalance mitigation strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Sequence


def compute_class_balanced_weights(
    class_counts: Sequence[int] | torch.Tensor,
    beta: float = 0.99,
) -> torch.Tensor:
    """Compute Cui et al. effective number sample weights.

    Formula:
        E_c = (1 - beta^N_c) / (1 - beta)
        w_c = (1 / E_c) / sum(1 / E_k) * num_classes

    Args:
        class_counts: Number of samples per class.
        beta: Hyperparameter beta in [0.9, 0.9999].

    Returns:
        1D FloatTensor of normalized class weights summing to num_classes.
    """
    counts = np.array(class_counts, dtype=np.float64)
    num_classes = len(counts)
    if num_classes == 0:
        return torch.empty(0, dtype=torch.float32)

    # Effective number: (1 - beta^n) / (1 - beta)
    # When n == 0, effective number is 1.0 (to avoid division by zero)
    effective_num = np.ones(num_classes, dtype=np.float64)
    for i, n in enumerate(counts):
        if n > 0:
            effective_num[i] = (1.0 - np.power(beta, n)) / (1.0 - beta)
        else:
            effective_num[i] = 1.0

    weights = 1.0 / effective_num
    normalized = weights / np.sum(weights) * float(num_classes)
    return torch.from_numpy(normalized).float()


class FocalLoss(nn.Module):
    """Multi-class Focal Loss for down-weighting well-classified easy samples."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Sequence[float] | torch.Tensor | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer(
                "alpha",
                torch.as_tensor(alpha, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute multi-class focal loss.

        Args:
            logits: Predicted class logits of shape (B, C).
            targets: Ground truth class indices of shape (B,).

        Returns:
            Scalar loss or unreduced tensor of shape (B,).
        """
        if logits.ndim != 2:
            raise ValueError(f"Expected 2D logits (B, C), got {tuple(logits.shape)}.")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError("targets must be 1D tensor matching batch dimension of logits.")

        log_p = torch.log_softmax(logits, dim=-1)
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = torch.exp(log_pt)

        focal_weight = torch.pow(1.0 - pt, self.gamma)
        loss = -focal_weight * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss combining effective sample weighting with focal modulation."""

    def __init__(
        self,
        class_counts: Sequence[int] | torch.Tensor | None = None,
        beta: float = 0.99,
        gamma: float = 2.0,
        alpha: Sequence[float] | torch.Tensor | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            weights = torch.as_tensor(alpha, dtype=torch.float32)
        elif class_counts is not None:
            weights = compute_class_balanced_weights(class_counts, beta=beta)
        else:
            weights = None

        if weights is not None:
            self.register_buffer("weights", weights, persistent=False)
        else:
            self.weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute class-balanced focal loss."""
        if logits.ndim != 2:
            raise ValueError(f"Expected 2D logits (B, C), got {tuple(logits.shape)}.")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError("targets must be 1D tensor matching batch dimension of logits.")

        log_p = torch.log_softmax(logits, dim=-1)
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = torch.exp(log_pt)

        focal_weight = torch.pow(1.0 - pt, self.gamma)
        loss = -focal_weight * log_pt

        if self.weights is not None:
            w_t = self.weights.to(logits.device).gather(0, targets)
            loss = w_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ClassBalancedCrossEntropyLoss(ClassBalancedFocalLoss):
    """Class-Balanced Cross Entropy (equivalent to ClassBalancedFocalLoss with gamma=0.0)."""

    def __init__(
        self,
        class_counts: Sequence[int] | torch.Tensor | None = None,
        beta: float = 0.99,
        alpha: Sequence[float] | torch.Tensor | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__(
            class_counts=class_counts,
            beta=beta,
            gamma=0.0,
            alpha=alpha,
            reduction=reduction,
        )
