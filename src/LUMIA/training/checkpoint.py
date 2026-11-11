"""Top-K model checkpointing engine ranked by validation metrics."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


class TopKModelCheckpointer:
    """Manages top-K model checkpoints ranked by a target validation metric."""

    def __init__(
        self,
        output_dir: Path | str,
        k: int = 3,
        metric_name: str = "val_macro_f1",
        mode: Literal["max", "min"] = "max",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.k = max(1, k)
        self.metric_name = metric_name
        self.mode = mode

        # List of tuples: (score, epoch, checkpoint_dict)
        self.best_entries: list[tuple[float, int, dict[str, Any]]] = []

    def _is_better(self, new_val: float, existing_val: float) -> bool:
        if self.mode == "max":
            return new_val > existing_val
        return new_val < existing_val

    def save_if_top_k(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer | None,
        scheduler: LRScheduler | None,
        metrics: dict[str, Any],
        config: Any,
    ) -> bool:
        """Evaluate if the current epoch metrics qualify for the top-K ranking and persist.

        Args:
            epoch: Current training epoch index.
            model: PyTorch model instance.
            optimizer: Optimizer instance.
            scheduler: LR scheduler instance.
            metrics: Dictionary of evaluated metrics (must contain self.metric_name).
            config: Training configuration object or dictionary.

        Returns:
            True if checkpoint was saved among top-K, False otherwise.
        """
        if self.metric_name not in metrics:
            logger.warning(
                "Metric '%s' missing from evaluation metrics. Checkpoint skipped.",
                self.metric_name,
            )
            return False

        score = float(metrics[self.metric_name])

        # Check if score qualifies for top-K
        if len(self.best_entries) < self.k:
            qualifies = True
        else:
            worst_top_k = self.best_entries[-1][0]
            qualifies = self._is_better(score, worst_top_k)

        if not qualifies:
            return False

        config_dict = config.model_dump() if hasattr(config, "model_dump") else config

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": config_dict,
            "lumia_version": "1.0.0",
        }

        self.best_entries.append((score, epoch, checkpoint_payload))
        # Sort entries: highest score first if mode=="max", lowest first if mode=="min"
        self.best_entries.sort(
            key=lambda x: x[0],
            reverse=(self.mode == "max"),
        )

        # Evict beyond K
        if len(self.best_entries) > self.k:
            self.best_entries = self.best_entries[: self.k]

        # Write files on disk for current top-K ranks
        for rank, (_sc, ep, payload) in enumerate(self.best_entries, start=1):
            rank_path = self.output_dir / f"best_tracklet_net_rank{rank}.pth"
            torch.save(payload, rank_path)
            if rank == 1:
                # Copy / update best model
                best_path = self.output_dir / "tracklet_net_best.pth"
                shutil.copyfile(rank_path, best_path)
                logger.info(
                    "New best model (Epoch %d, %s=%.4f) saved to %s",
                    ep,
                    self.metric_name,
                    _sc,
                    best_path,
                )

        return True
