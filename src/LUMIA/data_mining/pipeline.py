"""Solar system data-mining, artifact filtering, and HDF5 curation pipeline."""

from __future__ import annotations

import hashlib
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field

from LUMIA.core.schemas import LumiaBaseModel
from LUMIA.data_mining.archive_crawler import (
    ArchiveCrawler,
    CutoutTriplet,
    WCSMetadata,
    compute_fixed_sky_midpoint,
    compute_geometric_midpoint,
    partition_tracklet_sliding_window,
)
from LUMIA.data_mining.artifact_filter import (
    ArtifactFilterConfig,
    DeepStreaksArtifactFilter,
)
from LUMIA.data_mining.augmentation import (
    FaintSyntheticConfig,
    FaintSyntheticInjector,
    apply_spatial_crop_jitter,
)
from LUMIA.data_mining.crawler_ledger import SQLiteCrawlerLedger
from LUMIA.data_mining.qc_export import export_cutout_triplet_qc_png
from LUMIA.data_mining.storage import (
    DEFAULT_HDF5_STORAGE_PATH,
    DEFAULT_SPLITS_MANIFEST_PATH,
    HDF5TrainingStorageManager,
    TrackletSample,
)
from LUMIA.data_mining.taxonomy import (
    HorizonsClient,
    HorizonsQueryConfig,
    TargetObjectSpec,
    compute_optimal_cadence_window,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class DataMiningPipelineConfig(LumiaBaseModel):
    """Configuration for an automated data-mining and training dataset generation run."""

    output_hdf5_path: Path = Field(default=DEFAULT_HDF5_STORAGE_PATH)
    stamp_size: int = Field(default=144, ge=32, le=256)
    survey: str = Field(default="ZTF", description="Survey archive to crawl: ZTF, Pan-STARRS, Rubin/LSST.")
    faint_injection: bool = Field(
        default=False, description="Enable calibrated sub-5-sigma synthetic injection."
    )
    splits_manifest_path: Path | None = Field(default=DEFAULT_SPLITS_MANIFEST_PATH)
    ledger_db_path: Path | None = Field(default=Path("data/ground_truth/crawler_checkpoint.db"))
    enable_crop_jitter: bool = Field(default=True)
    crop_jitter_count: int = Field(default=1, ge=0, le=5)
    crop_jitter_max_offset_px: int = Field(default=4, ge=1, le=10)
    enable_synthetic_injection: bool = Field(default=True)
    synthetic_tracklets_per_target: int = Field(default=2, ge=0, le=50)
    synthetic_tracklet_length: int = Field(default=4, ge=2, le=20)
    max_hard_negative_ratio: float = Field(default=0.35, ge=0.0, le=1.0)
    dry_run: bool = Field(default=False, description="Run in safe test mode without API load.")
    max_cutouts: int = Field(
        default=5, ge=1, le=1000, description="Max total cutouts to harvest in dry-run mode."
    )
    qc_output_dir: Path | None = Field(
        default=None, description="Directory to export visual QA PNG images."
    )
    enable_d4_augmentation: bool = Field(
        default=False,
        description="Deprecated for disk; D4 symmetries are applied in-memory in DataLoader",
    )
    synthetic_samples_per_target: int = Field(
        default=2,
        ge=0,
        le=100,
        description="Alias for synthetic_tracklets_per_target for backward compatibility.",
    )
    horizons_config: HorizonsQueryConfig = Field(default_factory=HorizonsQueryConfig)
    filter_config: ArtifactFilterConfig = Field(default_factory=ArtifactFilterConfig)
    synthetic_config: FaintSyntheticConfig = Field(default_factory=FaintSyntheticConfig)


class TargetProcessingReport(LumiaBaseModel):
    """Execution summary for a single solar system target."""

    target_id: str = Field(...)
    taxonomy_class: str = Field(...)
    cadence_window_minutes: float = Field(...)
    ephemeris_points_found: int = Field(..., ge=0)
    raw_cutouts_extracted: int = Field(default=0, ge=0)
    artifacts_rejected: int = Field(default=0, ge=0)
    artifacts_detected: int = Field(default=0, ge=0)
    hard_negatives_curated: int = Field(default=0, ge=0)
    real_samples_curated: int = Field(default=0, ge=0)
    real_tracklets_curated: int = Field(default=0, ge=0)
    synthetic_samples_injected: int = Field(default=0, ge=0)
    synthetic_tracklets_injected: int = Field(default=0, ge=0)
    tracklets_created: int = Field(default=0, ge=0)
    qc_images_exported: int = Field(default=0, ge=0)


class PipelineCampaignReport(LumiaBaseModel):
    """Aggregate report certifying the completion of a full data-mining campaign."""

    started_at_utc: datetime = Field(...)
    completed_at_utc: datetime = Field(...)
    total_targets_processed: int = Field(..., ge=0)
    total_samples_written: int = Field(..., ge=0)
    qc_images_exported: int = Field(default=0, ge=0)
    split_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    target_reports: list[TargetProcessingReport] = Field(default_factory=list)


class SolarSystemDataMiningPipeline:
    """Production coordinator executing the solar system mining, filtering, and curation chain."""

    def __init__(self, config: DataMiningPipelineConfig | None = None) -> None:
        self.config = config if config is not None else DataMiningPipelineConfig()
        self.horizons_client = HorizonsClient(
            default_location=self.config.horizons_config.observatory_location
        )
        self.crawler = ArchiveCrawler()
        self.artifact_filter = DeepStreaksArtifactFilter(self.config.filter_config)
        self.synthetic_injector = FaintSyntheticInjector(self.config.synthetic_config)
        self.ledger = (
            SQLiteCrawlerLedger(db_path=self.config.ledger_db_path)
            if (self.config.ledger_db_path is not None and not self.config.dry_run)
            else None
        )
        self.storage = HDF5TrainingStorageManager(
            file_path=self.config.output_hdf5_path,
            stamp_size=self.config.stamp_size,
            seed=self.config.synthetic_config.seed,
            splits_manifest_path=self.config.splits_manifest_path,
            ledger=self.ledger,
        )
        self.total_qc_exported = 0

    def process_target(self, target: TargetObjectSpec) -> TargetProcessingReport:
        """Mine, filter, augment, and store variable-length tracklets for a single target."""
        logger.info("Initiating mining for %s (%s)", target.target_id, target.taxonomy_class)

        # 1. Query ephemeris track from NASA JPL Horizons
        eph_points = self.horizons_client.query_ephemeris(target, self.config.horizons_config)
        if len(eph_points) < 2:
            return TargetProcessingReport(
                target_id=target.target_id,
                taxonomy_class=target.taxonomy_class,
                cadence_window_minutes=45.0,
                ephemeris_points_found=len(eph_points),
                raw_cutouts_extracted=len(eph_points),
                artifacts_rejected=0,
                artifacts_detected=0,
                hard_negatives_curated=0,
                real_samples_curated=0,
                real_tracklets_curated=0,
                synthetic_samples_injected=0,
                synthetic_tracklets_injected=0,
                tracklets_created=0,
            )

        # 2. Compute dynamic cadence window based on object speed and class
        avg_speed = float(np.mean([p.total_speed_arcsec_min for p in eph_points]))
        cadence_window = compute_optimal_cadence_window(target.taxonomy_class, avg_speed)

        # 3. Partition track into fixed-sky sliding-window sub-tracklets
        sub_tracklets = partition_tracklet_sliding_window(
            eph_points, stamp_size=self.config.stamp_size, margin_px=12, min_length=2
        )
        if not sub_tracklets:
            sub_tracklets = [eph_points[: min(len(eph_points), 8)]]

        tracklets_to_store: list[TrackletSample] = []
        total_raw_frames = 0
        artifacts_detected = 0
        artifacts_rejected = 0
        hard_negatives_curated = 0
        real_tracklets_curated = 0
        target_qc_exported = 0

        h = w = self.config.stamp_size
        pix_scale_deg = 1.012 / 3600.0  # ~0.0002811 deg/px

        for st_idx, st_points in enumerate(sub_tracklets):
            ra_ref, dec_ref = compute_geometric_midpoint(st_points)
            sub_trk_id = f"TRK_{target.target_id}_{st_idx:03d}"

            # Skip tracklets already recorded in the SQLite ledger
            if self.ledger is not None and self.ledger.is_tracklet_completed(
                target.target_id, sub_trk_id
            ):
                logger.info("Tracklet %s already completed in ledger. Skipping.", sub_trk_id)
                continue

            frames: list[CutoutTriplet] = []
            has_artifact = False
            artifact_taxonomy = "ARTIFACT_UNKNOWN"

            is_streak = target.taxonomy_class == "NEO" and avg_speed > 3.0
            label_val = 3 if target.has_coma else (2 if is_streak else 1)

            # Stable seed per sub-tracklet
            seed_int = int(
                hashlib.md5(f"{target.target_id}:{st_idx}".encode()).hexdigest()[:6], 16
            )

            for ep_idx, pt in enumerate(st_points):
                total_raw_frames += 1

                if not self.config.dry_run:
                    # Production mode: query and fetch real FITS cutouts from IPAC ZTF IBE
                    try:
                        exposures = self.crawler.query_overlapping_exposures(
                            [pt], survey=self.config.survey
                        )
                        if exposures:
                            exp = exposures[0]
                            ref_key = self.crawler.build_reference_cache_key(
                                field=1,
                                filter_band=exp.filter_band,
                                ccd=1,
                                qid=1,
                                ra_0=ra_ref,
                                dec_0=dec_ref,
                            )
                            sci_b = self.crawler.fetch_fits_cutout(exp.science_url)
                            ref_b = self.crawler.fetch_reference_cutout(
                                exp.reference_url, cache_key=ref_key
                            )
                            diff_b = self.crawler.fetch_fits_cutout(exp.difference_url)
                            frame_triplet = self.crawler.extract_triplet_from_fits_buffers(
                                sci_bytes=sci_b,
                                ref_bytes=ref_b,
                                diff_bytes=diff_b,
                                point=pt,
                                ra_ref=ra_ref,
                                dec_ref=dec_ref,
                                epoch_index=ep_idx,
                                tracklet_id=sub_trk_id,
                                stamp_size=self.config.stamp_size,
                            )
                        else:
                            frame_triplet = None
                    except Exception as exc:
                        logger.debug(
                            "Live cutout fetch failed for %s epoch %d: %s",
                            target.target_id,
                            ep_idx,
                            exc,
                        )
                        frame_triplet = None

                    if frame_triplet is not None:
                        frames.append(frame_triplet)
                    continue

                # Safe dry-run mode: synthesize calibrated test triplet without network load
                cos_dec = math.cos(math.radians(dec_ref))
                dx_deg = (pt.ra_deg - ra_ref) * cos_dec
                dy_deg = pt.dec_deg - dec_ref
                tx = float((w / 2.0) + (dx_deg / pix_scale_deg))
                ty = float((h / 2.0) + (dy_deg / pix_scale_deg))

                wcs_params = WCSMetadata(
                    crpix=(float(w / 2.0), float(h / 2.0)),
                    crval=(ra_ref, dec_ref),
                    cd=((-pix_scale_deg, 0.0), (0.0, pix_scale_deg)),
                )

                rng = np.random.default_rng(seed_int + ep_idx)
                sci_bg = rng.normal(100.0, 10.0, (h, w)).astype(np.float32)
                ref_bg = rng.normal(100.0, 5.0, (h, w)).astype(np.float32)
                diff_bg = rng.normal(0.0, 8.0, (h, w)).astype(np.float32)

                # Insert expected moving signal on Difference & Science
                y_grid, x_grid = np.indices((h, w))
                r_sq = (x_grid - tx) ** 2 + (y_grid - ty) ** 2
                signal = 50.0 * np.exp(-r_sq / (2.0 * (1.5**2)))
                sci_bg += signal
                diff_bg += signal
                triplet_data = np.stack([sci_bg, ref_bg, diff_bg], axis=0)

                assessment = self.artifact_filter.evaluate_cutout(diff_bg, sci_bg)
                if assessment.is_artifact:
                    has_artifact = True
                    artifact_taxonomy = assessment.hard_negative_taxonomy

                frame_triplet = CutoutTriplet(
                    target_id=target.target_id,
                    tracklet_id=sub_trk_id,
                    epoch_index=ep_idx,
                    timestamp_utc=pt.timestamp_utc,
                    bjd_tdb=pt.bjd_tdb,
                    ra_deg=pt.ra_deg,
                    dec_deg=pt.dec_deg,
                    target_x=tx,
                    target_y=ty,
                    wcs_params=wcs_params,
                    stamp_size=self.config.stamp_size,
                    filter_band="r",
                    survey=self.config.survey,
                    snr=float(np.max(diff_bg) / 8.0),
                    mag_apparent=pt.apparent_mag,
                    is_synthetic=False,
                    is_artifact=assessment.is_artifact,
                    artifact_type=assessment.artifact_type,
                    data=triplet_data,
                )
                frames.append(frame_triplet)
                if (
                    self.config.dry_run
                    and self.config.qc_output_dir is not None
                    and self.total_qc_exported < self.config.max_cutouts
                ):
                    export_cutout_triplet_qc_png(frame_triplet, self.config.qc_output_dir)
                    self.total_qc_exported += 1
                    target_qc_exported += 1

            if len(frames) < 2:
                continue

            if has_artifact:
                artifacts_detected += 1
                cur_total = hard_negatives_curated + real_tracklets_curated + 1
                if (hard_negatives_curated + 1) / cur_total <= self.config.max_hard_negative_ratio:
                    hard_neg_sample = TrackletSample(
                        tracklet_id=f"BOGUS_{sub_trk_id}",
                        object_id=f"BOGUS_{target.target_id}",
                        taxonomy_class=artifact_taxonomy,
                        label=0,
                        frames=frames,
                    )
                    tracklets_to_store.append(hard_neg_sample)
                    hard_negatives_curated += 1
                else:
                    artifacts_rejected += 1
                continue

            # Valid real tracklet
            clean_sample = TrackletSample(
                tracklet_id=sub_trk_id,
                object_id=target.target_id,
                taxonomy_class=target.taxonomy_class,
                label=label_val,
                frames=frames,
            )
            tracklets_to_store.append(clean_sample)
            real_tracklets_curated += 1

            # Spatial crop jittering (Refinement 6)
            if self.config.enable_crop_jitter and self.config.crop_jitter_count > 0:
                for j in range(self.config.crop_jitter_count):
                    jitter_rng = np.random.default_rng(seed_int + 500 + j)
                    jittered_frames = apply_spatial_crop_jitter(
                        frames,
                        max_offset_px=self.config.crop_jitter_max_offset_px,
                        rng=jitter_rng,
                    )
                    jitter_sample = TrackletSample(
                        tracklet_id=f"{sub_trk_id}_jitter{j + 1}",
                        object_id=target.target_id,
                        taxonomy_class=target.taxonomy_class,
                        label=label_val,
                        frames=jittered_frames,
                    )
                    tracklets_to_store.append(jitter_sample)

        # 4. Faint Synthetic Multi-Epoch Tracklet Injection (Refinement 6)
        synthetic_tracklets_curated = 0
        num_synth = max(
            self.config.synthetic_tracklets_per_target,
            self.config.synthetic_samples_per_target,
        )
        if self.config.enable_synthetic_injection and num_synth > 0 and len(eph_points) >= 2:
            syn_len = min(len(eph_points), self.config.synthetic_tracklet_length)
            if syn_len < 2:
                syn_len = 2

            for s_idx in range(num_synth):
                syn_trk_id = f"SYNTH_{target.target_id}_{s_idx:02d}"
                if (
                    self.ledger is not None
                    and not self.config.dry_run
                    and self.ledger.is_tracklet_completed(target.target_id, syn_trk_id)
                ):
                    logger.info(
                        "Synthetic tracklet %s already completed in ledger. Skipping.", syn_trk_id
                    )
                    continue

                syn_rng = np.random.default_rng(s_idx + 200)
                clean_bg_triplets = [
                    np.stack(
                        [
                            syn_rng.normal(100.0, 10.0, (h, w)).astype(np.float32),
                            syn_rng.normal(100.0, 5.0, (h, w)).astype(np.float32),
                            syn_rng.normal(0.0, 8.0, (h, w)).astype(np.float32),
                        ],
                        axis=0,
                    )
                    for _ in range(syn_len)
                ]
                inj_triplets, syn_metas = self.synthetic_injector.inject_synthetic_tracklet(
                    clean_background_triplets=clean_bg_triplets,
                    target_id=syn_trk_id,
                    speed_px_per_frame=float(np.clip(avg_speed * 1.5, 1.0, 8.0)),
                    position_angle_deg=float(s_idx * 45.0),
                    mag=float(20.0 + (s_idx * 0.4)),
                )

                syn_frames: list[CutoutTriplet] = []
                base_pt = eph_points[0]
                for f_idx, (in_trip, smeta) in enumerate(zip(inj_triplets, syn_metas, strict=True)):
                    syn_frame = CutoutTriplet(
                        target_id=f"SYNTH_{target.target_id}",
                        tracklet_id=syn_trk_id,
                        epoch_index=f_idx,
                        timestamp_utc=base_pt.timestamp_utc,
                        bjd_tdb=base_pt.bjd_tdb + (f_idx * (cadence_window / 1440.0)),
                        ra_deg=base_pt.ra_deg,
                        dec_deg=base_pt.dec_deg,
                        target_x=smeta.centroid_x_px,
                        target_y=smeta.centroid_y_px,
                        wcs_params=WCSMetadata(
                            crpix=(float(w / 2.0), float(h / 2.0)),
                            crval=(base_pt.ra_deg, base_pt.dec_deg),
                            cd=((-pix_scale_deg, 0.0), (0.0, pix_scale_deg)),
                        ),
                        stamp_size=self.config.stamp_size,
                        filter_band="r",
                        survey="ZTF",
                        snr=smeta.snr,
                        mag_apparent=smeta.apparent_mag,
                        is_synthetic=True,
                        is_artifact=False,
                        artifact_type="NONE",
                        data=in_trip,
                    )
                    syn_frames.append(syn_frame)

                syn_label = 2 if syn_metas[0].is_streak else 1
                syn_sample = TrackletSample(
                    tracklet_id=syn_trk_id,
                    object_id=target.target_id,  # Keeps target_id for group-based split routing
                    taxonomy_class=f"SYNTH_{target.taxonomy_class}",
                    label=syn_label,
                    frames=syn_frames,
                )
                tracklets_to_store.append(syn_sample)
                synthetic_tracklets_curated += 1

        # 5. Persist variable-length tracklets to HDF5 via CSR indexing
        if tracklets_to_store:
            self.storage.append_tracklets(tracklets_to_store)

        return TargetProcessingReport(
            target_id=target.target_id,
            taxonomy_class=target.taxonomy_class,
            cadence_window_minutes=cadence_window,
            ephemeris_points_found=len(eph_points),
            raw_cutouts_extracted=total_raw_frames,
            artifacts_rejected=artifacts_rejected,
            artifacts_detected=artifacts_detected,
            hard_negatives_curated=hard_negatives_curated,
            real_samples_curated=real_tracklets_curated,
            real_tracklets_curated=real_tracklets_curated,
            synthetic_samples_injected=synthetic_tracklets_curated,
            synthetic_tracklets_injected=synthetic_tracklets_curated,
            tracklets_created=len(tracklets_to_store),
            qc_images_exported=target_qc_exported,
        )

    def run_mining_campaign(
        self,
        targets: Sequence[TargetObjectSpec],
    ) -> PipelineCampaignReport:
        """Execute a full mining campaign across a diverse catalog of solar system targets."""
        start_time = datetime.now(UTC)
        reports: list[TargetProcessingReport] = []

        for target in targets:
            rep = self.process_target(target)
            reports.append(rep)
            if self.config.dry_run and self.total_qc_exported >= self.config.max_cutouts:
                logger.info(
                    "Dry-run maximum cutout threshold reached (%d). Halting campaign.",
                    self.config.max_cutouts,
                )
                break

        summary = self.storage.get_summary()
        total_samples = sum(s["total_tracklets"] for s in summary.values())

        return PipelineCampaignReport(
            started_at_utc=start_time,
            completed_at_utc=datetime.now(UTC),
            total_targets_processed=len(reports),
            total_samples_written=total_samples,
            qc_images_exported=self.total_qc_exported,
            split_counts=summary,
            target_reports=reports,
        )
