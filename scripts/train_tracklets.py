"""CLI execution script for training Project LUMIA multi-epoch tracklet classifiers."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from LUMIA.data_mining.dataset import create_ztf_tracklet_dataloader
from LUMIA.models.tracklet_net import TrackletClassifierNet
from LUMIA.training.config import (
    TrackletModelConfig,
    TrackletTrainingConfig,
)
from LUMIA.training.trainer import TrackletTrainer

if TYPE_CHECKING:
    from collections.abc import Sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_tracklets")


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for tracklet model training."""
    parser = argparse.ArgumentParser(
        description="Train Project LUMIA multi-epoch tracklet neural networks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/ground_truth/ztf_sso_tracklets_128x128.h5"),
        help="Path to CSR HDF5 archive containing variable-length tracklets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/models"),
        help="Directory to persist top-K ranked model checkpoints.",
    )
    parser.add_argument(
        "--aggregator",
        type=str,
        default="gru",
        choices=["gru", "bigru", "transformer", "masked_pool"],
        help="Temporal aggregator architecture.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size of tracklets.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Base learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="class_balanced_focal",
        choices=["class_balanced_focal", "focal", "cross_entropy"],
        help="Loss function formulation.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focusing parameter gamma for focal modulation.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val_map",
        choices=["val_map", "val_macro_f1", "val_ap_real"],
        help="Primary validation metric for top-K checkpoint selection.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine_annealing_warm_restarts",
        choices=["cosine_annealing_warm_restarts", "cosine_decay"],
        help="Learning rate decay schedule.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of DataLoader worker subprocesses.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device ('cuda', 'cpu', 'cuda:0', or 'auto').",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Enable Automatic Mixed Precision (AMP).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_false",
        dest="amp",
        help="Disable Automatic Mixed Precision (AMP).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform configuration check and exit without running full training.",
    )
    return parser.parse_args(args)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute training script workflow."""
    args = parse_args(argv)

    # 1. Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 2. Build and validate Pydantic configuration contracts
    model_cfg = TrackletModelConfig(
        spatial_backbone="resnet_lite",
        spatial_embed_dim=128,
        temporal_aggregator=args.aggregator,
        temporal_hidden_dim=128,
        temporal_layers=2,
        dropout_rate=0.25,
        num_classes=4,
    )

    train_cfg = TrackletTrainingConfig(
        dataset_path=args.dataset,
        model_output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        loss_function=args.loss,
        focal_gamma=args.focal_gamma,
        primary_metric=args.metric,
        lr_scheduler_type=args.scheduler,
        use_amp=args.amp,
        num_workers=args.workers,
        seed=args.seed,
        model=model_cfg,
    )

    logger.info("Initializing TrackletClassifierNet with config: %s", model_cfg.model_dump())
    model = TrackletClassifierNet(model_cfg)

    if args.dry_run:
        logger.info("Dry-run requested. Configuration and model successfully verified.")
        return 0

    if not train_cfg.dataset_path.exists():
        logger.error("Dataset archive not found at: %s", train_cfg.dataset_path)
        return 1

    train_loader = create_ztf_tracklet_dataloader(
        h5_path=train_cfg.dataset_path,
        split="train",
        batch_size=train_cfg.batch_size,
        shuffle=True,
        augment=True,
        num_workers=train_cfg.num_workers,
        pin_memory=(args.device != "cpu" and torch.cuda.is_available()),
    )

    val_loader = create_ztf_tracklet_dataloader(
        h5_path=train_cfg.dataset_path,
        split="val",
        batch_size=train_cfg.batch_size,
        shuffle=False,
        augment=False,
        num_workers=train_cfg.num_workers,
        pin_memory=(args.device != "cpu" and torch.cuda.is_available()),
    )

    trainer = TrackletTrainer(
        model=model,
        config=train_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
    )

    results = trainer.train()
    logger.info(
        "Training completed successfully. Best %s: %.4f (Val Macro-F1: %.4f). Saved to %s",
        train_cfg.primary_metric,
        results.get("best_primary_metric", 0.0),
        results.get("best_val_macro_f1", 0.0),
        results["best_model_path"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
