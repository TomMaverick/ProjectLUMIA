"""Unit tests for TrackletTrainer optimization loop, LR scheduler, and validation engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from LUMIA.models.tracklet_net import TrackletClassifierNet
from LUMIA.training.config import TrackletModelConfig, TrackletTrainingConfig
from LUMIA.training.trainer import TrackletTrainer, extract_dataset_labels

if TYPE_CHECKING:
    from pathlib import Path


class MockTrackletDataset(Dataset[dict[str, Any]]):
    """Synthetic dataset yielding variable-length tracklet tensors."""

    def __init__(self, size: int = 8, max_len: int = 4) -> None:
        self.size = size
        self.max_len = max_len

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq_len = (idx % self.max_len) + 1
        return {
            "tracklet": torch.randn(seq_len, 3, 128, 128),
            "label": torch.tensor(idx % 4, dtype=torch.long),
            "length": seq_len,
            "target_coords": torch.zeros((seq_len, 2), dtype=torch.float32),
            "tracklet_id": f"trk_{idx}",
            "object_id": f"obj_{idx}",
            "taxonomy_class": "NEO",
        }


def mock_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable sequences to max batch length."""
    b_size = len(batch)
    max_len = max(item["length"] for item in batch)
    padded = torch.zeros(b_size, max_len, 3, 128, 128)
    mask = torch.zeros(b_size, max_len, dtype=torch.bool)
    labels = torch.stack([item["label"] for item in batch])

    for i, item in enumerate(batch):
        seq_len = item["length"]
        padded[i, :seq_len] = item["tracklet"]
        mask[i, :seq_len] = True

    return {
        "tracklets": padded,
        "labels": labels,
        "padding_mask": mask,
    }


class TestTrackletTrainer:
    """Verify TrackletTrainer training epochs, evaluation, and scheduler stepping."""

    def test_trainer_smoke_run(self, tmp_path: Path) -> None:
        """Trainer executes train and validation passes and writes top-K checkpoints."""
        model_cfg = TrackletModelConfig(
            spatial_embed_dim=32,
            temporal_aggregator="masked_pool",
            temporal_hidden_dim=32,
            num_classes=4,
        )
        model = TrackletClassifierNet(model_cfg)

        train_ds = MockTrackletDataset(size=8, max_len=3)
        val_ds = MockTrackletDataset(size=4, max_len=3)

        train_loader = DataLoader(train_ds, batch_size=4, collate_fn=mock_collate)
        val_loader = DataLoader(val_ds, batch_size=4, collate_fn=mock_collate)

        ckpt_dir = tmp_path / "models"
        train_cfg = TrackletTrainingConfig(
            model_output_dir=ckpt_dir,
            batch_size=4,
            epochs=2,
            learning_rate=1e-3,
            warmup_epochs=1,
            grad_clip_norm=1.0,
            use_amp=False,
            model=model_cfg,
        )

        trainer = TrackletTrainer(
            model=model,
            config=train_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            device="cpu",
        )

        results = trainer.train()

        assert "best_val_macro_f1" in results
        assert "history" in results
        assert len(results["history"]) == 2
        assert (ckpt_dir / "tracklet_net_best.pth").exists()

    def test_lr_scheduler_warmup_and_decay(self, tmp_path: Path) -> None:
        """Scheduler linearly ramps up during warmup then decays cosine-wise."""
        model_cfg = TrackletModelConfig(
            spatial_embed_dim=32,
            temporal_aggregator="masked_pool",
            num_classes=4,
        )
        model = TrackletClassifierNet(model_cfg)
        train_ds = MockTrackletDataset(size=4, max_len=2)
        loader = DataLoader(train_ds, batch_size=4, collate_fn=mock_collate)

        train_cfg = TrackletTrainingConfig(
            model_output_dir=tmp_path / "models",
            epochs=4,
            learning_rate=1e-3,
            warmup_epochs=2,
            use_amp=False,
            model=model_cfg,
        )

        trainer = TrackletTrainer(
            model=model,
            config=train_cfg,
            train_loader=loader,
            val_loader=loader,
            device="cpu",
        )

        lrs: list[float] = []
        for epoch in range(1, 5):
            trainer.train_epoch(epoch)
            current_lr = trainer.get_current_lr()
            lrs.append(current_lr)
            trainer.scheduler.step()

        # Epoch 1: warmup starts at 0.1 * lr (0.0001)
        assert lrs[0] < lrs[1]
        # At epoch 2 (end of warmup), lr is close to base_lr (0.001)
        assert pytest.approx(lrs[1], rel=0.1) == 1e-3
        # After epoch 2, cosine decay sets in: lr3 < lr2
        assert lrs[2] <= lrs[1]

    def test_extract_dataset_labels_from_mock_dataset(self) -> None:
        """extract_dataset_labels extracts all labels from indexable dataset."""
        ds = MockTrackletDataset(size=8, max_len=2)
        labels = extract_dataset_labels(ds, num_classes=4)
        assert labels is not None
        assert len(labels) == 8
        assert set(labels.tolist()) == {0, 1, 2, 3}

    def test_cosine_annealing_warm_restarts_scheduler(self, tmp_path: Path) -> None:
        """Trainer instantiates CosineAnnealingWarmRestarts and steps successfully."""
        model_cfg = TrackletModelConfig(
            spatial_embed_dim=32,
            temporal_aggregator="masked_pool",
            num_classes=4,
        )
        model = TrackletClassifierNet(model_cfg)
        train_ds = MockTrackletDataset(size=4, max_len=2)
        loader = DataLoader(train_ds, batch_size=4, collate_fn=mock_collate)

        train_cfg = TrackletTrainingConfig(
            model_output_dir=tmp_path / "models",
            epochs=3,
            learning_rate=1e-3,
            lr_scheduler_type="cosine_annealing_warm_restarts",
            restart_t0_epochs=2,
            restart_t_mult=1,
            use_amp=False,
            model=model_cfg,
        )

        trainer = TrackletTrainer(
            model=model,
            config=train_cfg,
            train_loader=loader,
            val_loader=loader,
            device="cpu",
        )

        trainer.train_epoch(1)
        trainer.scheduler.step()
        assert trainer.get_current_lr() > 0.0
