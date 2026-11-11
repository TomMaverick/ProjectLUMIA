"""Command-line coordinator for Project LUMIA solar system data-mining and dataset generation.

Supports safe dry-run mode (--dry-run, --max-cutouts 5) with automated visual QC PNG export.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from LUMIA.data_mining.pipeline import (
    DataMiningPipelineConfig,
    SolarSystemDataMiningPipeline,
)
from LUMIA.data_mining.storage import DEFAULT_HDF5_STORAGE_PATH
from LUMIA.data_mining.taxonomy import (
    HorizonsQueryConfig,
    load_objects_catalog,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from LUMIA.data_mining.taxonomy import TargetObjectSpec

logger = logging.getLogger("LUMIA.data_mining.cli")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line interface arguments for data-mining execution."""
    parser = argparse.ArgumentParser(
        description="Project LUMIA Solar System Data-Mining & Curation Coordinator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("configs/objects_catalog.json"),
        help="Path to solar system objects catalog JSON.",
    )
    parser.add_argument(
        "--output-hdf5",
        type=Path,
        default=DEFAULT_HDF5_STORAGE_PATH,
        help="Destination path for out-of-core CSR HDF5 archive.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Activate safe dry-run mode: fetch <= max-cutouts total and export visual QC PNGs.",
    )
    parser.add_argument(
        "--max-cutouts",
        type=int,
        default=5,
        help="Maximum total cutouts to extract across targets in dry-run mode.",
    )
    parser.add_argument(
        "--qc-dir",
        type=Path,
        default=Path("data/debug_qc"),
        help="Directory to save visual quality control 3-panel PNG images in dry-run mode.",
    )
    parser.add_argument(
        "--stamp-size",
        type=int,
        default=144,
        choices=[63, 128, 144],
        help="Pixel width and height of square cutouts.",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=None,
        help="Filter targets by specific object IDs or names (e.g., --objects 99942 1 67P).",
    )
    parser.add_argument(
        "--taxonomy",
        choices=["NEO", "MBA", "COMET", "TNO", "ISO"],
        default=None,
        help="Filter targets by dynamic solar system taxonomy class.",
    )
    parser.add_argument(
        "--step-minutes",
        type=int,
        default=15,
        help="Cadence time step for JPL Horizons ephemerides query in minutes.",
    )
    parser.add_argument(
        "--obs-days",
        type=float,
        default=1.0,
        help="Ephemeris time window duration in days from current UTC.",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Disable synthetic moving-object injection (only harvest real survey cutouts).",
    )
    parser.add_argument(
        "--survey",
        type=str,
        default="ZTF",
        choices=["ZTF", "Pan-STARRS", "Rubin", "panstarrs", "rubin", "ztf"],
        help="Target survey archive for server-side cutout retrieval (ZTF, Pan-STARRS, Rubin).",
    )
    parser.add_argument(
        "--faint-injection",
        action="store_true",
        help="Enable calibrated sub-5-sigma synthetic moving source injection.",
    )
    parser.add_argument(
        "--reset-ledger",
        action="store_true",
        help="Clear entire crawler checkpoint ledger before running.",
    )
    parser.add_argument(
        "--reset-targets",
        action="store_true",
        help="Clear crawler checkpoint ledger records for selected targets before running.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity level.",
    )
    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Execute mining run based on command line arguments."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Initializing Project LUMIA Data-Mining Subsystem")
    if args.dry_run:
        logger.info(
            "SAFE DRY-RUN ACTIVE: Max cutouts = %d | QC Export Dir = %s",
            args.max_cutouts,
            args.qc_dir,
        )

    # 1. Load target objects catalog
    try:
        entries = load_objects_catalog(args.catalog)
        logger.info("Loaded %d object entries from catalog: %s", len(entries), args.catalog)
    except Exception as exc:
        logger.error("Failed to load catalog from %s: %s", args.catalog, exc)
        return 1

    # 2. Filter targets if specified
    filtered_specs: list[TargetObjectSpec] = []
    object_filter_set = set(args.objects) if args.objects else None

    for entry in entries:
        if args.taxonomy and entry.taxonomy_class != args.taxonomy:
            continue
        if object_filter_set and (
            entry.object_id not in object_filter_set and entry.name not in object_filter_set
        ):
            continue
        filtered_specs.append(entry.to_target_spec())

    if not filtered_specs:
        logger.warning("No targets matched the given filter criteria.")
        return 0

    logger.info("Selected %d targets for harvesting sequence", len(filtered_specs))

    # 3. Configure Horizons and Pipeline
    now_utc = datetime.now(UTC)
    horizons_cfg = HorizonsQueryConfig(
        start_time=now_utc,
        end_time=now_utc + timedelta(days=args.obs_days),
        step_minutes=args.step_minutes,
    )

    if args.reset_ledger and not args.dry_run:
        from LUMIA.data_mining.crawler_ledger import SQLiteCrawlerLedger

        ledger = SQLiteCrawlerLedger()
        ledger.clear()
        logger.info("Cleared entire crawler checkpoint ledger.")
    elif args.reset_targets and not args.dry_run:
        from LUMIA.data_mining.crawler_ledger import SQLiteCrawlerLedger

        ledger = SQLiteCrawlerLedger()
        ledger.delete_for_objects([t.target_id for t in filtered_specs])
        logger.info(
            "Cleared crawler checkpoint ledger for %d selected targets.", len(filtered_specs)
        )

    qc_dir = args.qc_dir if args.dry_run else None

    pipeline_cfg = DataMiningPipelineConfig(
        output_hdf5_path=args.output_hdf5,
        stamp_size=args.stamp_size,
        survey=args.survey,
        faint_injection=args.faint_injection,
        dry_run=args.dry_run,
        max_cutouts=args.max_cutouts,
        qc_output_dir=qc_dir,
        enable_synthetic_injection=not args.no_synthetic,
        horizons_config=horizons_cfg,
    )

    pipeline = SolarSystemDataMiningPipeline(pipeline_cfg)

    # 4. Execute campaign
    campaign_report = pipeline.run_mining_campaign(filtered_specs)

    # 5. Format and display summary
    print("\n" + "=" * 70)
    print("PROJECT LUMIA DATA-MINING CAMPAIGN REPORT")
    print("=" * 70)
    print(f"Start Time (UTC)       : {campaign_report.started_at_utc.isoformat()}")
    print(f"End Time (UTC)         : {campaign_report.completed_at_utc.isoformat()}")
    print(f"Mode                   : {'DRY-RUN (Safe Test)' if args.dry_run else 'PRODUCTION'}")
    print(f"Targets Processed      : {campaign_report.total_targets_processed}")
    print(f"Tracklets Written      : {campaign_report.total_samples_written}")
    print(f"QC PNGs Exported       : {campaign_report.qc_images_exported}")
    if args.dry_run and qc_dir:
        print(f"Visual QC Directory    : {qc_dir.resolve()}")
    print(f"HDF5 Archive File      : {args.output_hdf5.resolve()}")
    print("-" * 70)
    print("Partition Distribution:")
    for split_name, stats in campaign_report.split_counts.items():
        print(
            f"  [{split_name.upper():5s}] Tracklets: {stats.get('total_tracklets', 0):4d} | "
            f"Frames: {stats.get('total_frames', 0):4d} | "
            f"Point: {stats.get('point', 0):3d} | Streak: {stats.get('streak', 0):3d} | "
            f"Coma: {stats.get('coma', 0):3d} | Bogus/HN: {stats.get('bogus', 0):3d}"
        )
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
