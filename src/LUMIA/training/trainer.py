"""Training coordinator managing epochs, validation, AMP, clipping, and schedulers."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LambdaLR,
    LRScheduler,
)
from tqdm import tqdm

from LUMIA.training.calibration import TemperatureScaler, ThresholdOptimizer
from LUMIA.training.checkpoint import TopKModelCheckpointer
from LUMIA.training.loss import (
    ClassBalancedFocalLoss,
    FocalLoss,
)
from LUMIA.training.metrics import TrackletMetricsTracker

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from LUMIA.models.tracklet_net import TrackletClassifierNet
    from LUMIA.training.config import TrackletTrainingConfig

logger = logging.getLogger(__name__)


def extract_dataset_labels(dataset: Any, num_classes: int = 4) -> np.ndarray | None:
    """Attempt to extract ground truth class labels from a PyTorch Dataset."""
    valid_types = (np.ndarray, torch.Tensor, list)
    if hasattr(dataset, "labels") and isinstance(dataset.labels, valid_types):
        return np.asarray(dataset.labels, dtype=np.int64)
    if hasattr(dataset, "targets") and isinstance(dataset.targets, valid_types):
        return np.asarray(dataset.targets, dtype=np.int64)

    if hasattr(dataset, "h5_path") and hasattr(dataset, "split"):
        try:
            import h5py

            with h5py.File(dataset.h5_path, "r") as h5:
                if "splits" in h5 and dataset.split in h5["splits"]:
                    grp = h5["splits"][dataset.split]
                    if "tracklet_labels" in grp:
                        return np.array(grp["tracklet_labels"][:], dtype=np.int64)
        except Exception as err:
            logger.debug("Could not read labels from HDF5 dataset: %s", err)

    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        sub_labels = extract_dataset_labels(dataset.dataset, num_classes)
        if sub_labels is not None:
            return sub_labels[dataset.indices]

    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        try:
            if len(dataset) <= 1000:
                labels_list: list[int] = []
                for idx in range(len(dataset)):
                    item = dataset[idx]
                    if isinstance(item, dict) and "label" in item:
                        lbl = item["label"]
                        labels_list.append(int(lbl.item() if hasattr(lbl, "item") else lbl))
                if labels_list:
                    return np.array(labels_list, dtype=np.int64)
        except Exception:
            pass

    return None


class TrackletTrainer:
    """Manages the full multi-epoch tracklet model training lifecycle."""

    def __init__(
        self,
        model: TrackletClassifierNet,
        config: TrackletTrainingConfig,
        train_loader: DataLoader[dict[str, Any]],
        val_loader: DataLoader[dict[str, Any]],
        device: torch.device | str | None = None,
        loss_fn: nn.Module | None = None,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 1. Device resolution
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
        else:
            self.device = device

        self.model = model.to(self.device)

        # 2. Loss Function
        if loss_fn is not None:
            self.loss_fn = loss_fn.to(self.device)
        else:
            self.loss_fn = self._build_loss_fn().to(self.device)

        # 3. Optimizer (AdamW)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # 4. Learning Rate Scheduler
        self.scheduler: LRScheduler = self._build_lr_scheduler()

        # 5. AMP GradScaler
        self.use_amp = bool(config.use_amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler(
            "cuda" if self.device.type == "cuda" else "cpu",
            enabled=self.use_amp,
        )

        # 6. Metrics & Checkpointing
        self.metrics_tracker = TrackletMetricsTracker(num_classes=config.model.num_classes)
        self.temp_scaler = TemperatureScaler()
        self.threshold_optimizer = ThresholdOptimizer(num_classes=config.model.num_classes)
        self.checkpointer = TopKModelCheckpointer(
            output_dir=config.model_output_dir,
            k=3,
            metric_name=config.primary_metric,
            mode="max",
        )

    def _build_loss_fn(self) -> nn.Module:
        loss_type = self.config.loss_function
        if loss_type == "class_balanced_focal":
            class_counts = None
            if hasattr(self.train_loader, "dataset"):
                labels = extract_dataset_labels(
                    self.train_loader.dataset,
                    num_classes=self.config.model.num_classes,
                )
                if labels is not None and len(labels) > 0:
                    class_counts = np.bincount(labels, minlength=self.config.model.num_classes)
                    logger.info("ClassBalancedFocalLoss initialized with counts: %s", class_counts)

            return ClassBalancedFocalLoss(
                class_counts=class_counts,
                beta=self.config.class_balanced_beta,
                gamma=self.config.focal_gamma,
            )
        if loss_type == "focal":
            return FocalLoss(gamma=self.config.focal_gamma)
        if loss_type == "cross_entropy":
            return nn.CrossEntropyLoss()
        raise ValueError(f"Unsupported loss function '{loss_type}'.")

    def _build_lr_scheduler(self) -> LRScheduler:
        if self.config.lr_scheduler_type == "cosine_annealing_warm_restarts":
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.restart_t0_epochs,
                T_mult=self.config.restart_t_mult,
                eta_min=1e-6,
            )

        warmup_epochs = self.config.warmup_epochs
        total_epochs = self.config.epochs
        base_lr = self.config.learning_rate

        def lr_lambda(current_epoch: int) -> float:
            if warmup_epochs > 0 and current_epoch < warmup_epochs:
                fraction = float(current_epoch + 1) / float(warmup_epochs)
                return 0.1 + 0.9 * fraction

            decay_epochs = max(1, total_epochs - warmup_epochs)
            progress = float(current_epoch + 1 - warmup_epochs) / float(decay_epochs)
            progress = min(max(progress, 0.0), 1.0)
            min_ratio = 1e-6 / max(1e-9, base_lr)
            return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def get_current_lr(self) -> float:
        """Return the current learning rate from optimizer param groups."""
        return float(self.optimizer.param_groups[0]["lr"])

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Execute one complete training epoch over train_loader."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.epochs} [Train]",
            leave=False,
        )
        for batch in pbar:
            tracklets: torch.Tensor = batch["tracklets"].to(self.device)
            padding_mask: torch.Tensor = batch["padding_mask"].to(self.device)
            labels: torch.Tensor = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            device_type = self.device.type if self.device.type in ("cuda", "cpu") else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                logits = self.model(tracklets, padding_mask)
                loss = self.loss_fn(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.grad_clip_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            preds = torch.argmax(logits, dim=-1)
            correct += int((preds == labels).sum().item())
            total_samples += batch_size

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / max(1, total_samples):.3f}",
            )

        avg_loss = total_loss / max(1, total_samples)
        acc = correct / max(1, total_samples)
        return {"train_loss": avg_loss, "train_accuracy": acc}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate model on the specified DataLoader."""
        self.model.eval()
        all_logits: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        total_loss = 0.0
        total_samples = 0

        device_type = self.device.type if self.device.type in ("cuda", "cpu") else "cpu"

        for batch in loader:
            tracklets: torch.Tensor = batch["tracklets"].to(self.device)
            padding_mask: torch.Tensor = batch["padding_mask"].to(self.device)
            labels: torch.Tensor = batch["labels"].to(self.device)

            with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                logits = self.model(tracklets, padding_mask)
                loss = self.loss_fn(logits, labels)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        y_true = np.concatenate(all_labels, axis=0)
        y_logits = np.concatenate(all_logits, axis=0)

        metrics = self.metrics_tracker.compute_metrics(y_true=y_true, logits=y_logits)
        metrics["val_loss"] = total_loss / max(1, total_samples)
        metrics["val_map"] = metrics["map"]
        metrics["val_macro_f1"] = metrics["macro_f1"]
        metrics["val_fpr90"] = metrics["fpr90"]
        metrics["val_ap_real"] = metrics["ap_real"]

        if len(y_true) >= self.config.model.num_classes:
            if self.config.enable_temperature_scaling:
                temp = self.temp_scaler.fit(y_logits, y_true)
                calibrated_logits = self.temp_scaler.transform(y_logits)
                metrics["temperature"] = temp
            else:
                calibrated_logits = y_logits

            if self.config.enable_threshold_optimization:
                exp_l = np.exp(
                    calibrated_logits - np.max(calibrated_logits, axis=-1, keepdims=True)
                )
                calibrated_probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
                opt_thresh = self.threshold_optimizer.fit(calibrated_probs, y_true)
                cal_metrics = self.metrics_tracker.compute_metrics(
                    y_true=y_true,
                    logits=calibrated_logits,
                    thresholds=opt_thresh,
                )
                metrics["calibrated_macro_f1"] = cal_metrics["macro_f1"]
                metrics["optimal_thresholds"] = opt_thresh.tolist()

        return metrics

    def train(self) -> dict[str, Any]:
        """Execute full training loop across all configured epochs."""
        logger.info(
            "Starting Tracklet training on device '%s' for %d epochs.",
            self.device,
            self.config.epochs,
        )
        history: list[dict[str, Any]] = []
        best_primary_metric = 0.0

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)
            self.scheduler.step()

            epoch_record = {
                "epoch": epoch,
                "lr": self.get_current_lr(),
                **train_metrics,
                **val_metrics,
            }
            history.append(epoch_record)

            logger.info(
                "Epoch %d/%d - Train Loss: %.4f, Val Loss: %.4f, "
                "Val mAP: %.4f, Val Macro-F1: %.4f, Val AP_real: %.4f, Val FPR90: %.4f",
                epoch,
                self.config.epochs,
                train_metrics["train_loss"],
                val_metrics["val_loss"],
                val_metrics["val_map"],
                val_metrics["val_macro_f1"],
                val_metrics["val_ap_real"],
                val_metrics["val_fpr90"],
            )

            is_top_k = self.checkpointer.save_if_top_k(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics=val_metrics,
                config=self.config,
            )
            current_primary = float(val_metrics.get(self.config.primary_metric, 0.0))
            if is_top_k and current_primary > best_primary_metric:
                best_primary_metric = current_primary

        return {
            "history": history,
            "best_primary_metric": best_primary_metric,
            f"best_{self.config.primary_metric}": best_primary_metric,
            "best_val_macro_f1": (
                max(h.get("val_macro_f1", 0.0) for h in history) if history else 0.0
            ),
            "best_val_map": max(h.get("val_map", 0.0) for h in history) if history else 0.0,
            "best_model_path": str(self.config.model_output_dir / "tracklet_net_best.pth"),
        }
