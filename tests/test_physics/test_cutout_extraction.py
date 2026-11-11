"""Unit and integration tests for Module 4 -> Module 7 SNR peak & triplet cutout bridge."""

from __future__ import annotations

import ast
import builtins
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

import LUMIA.core.schemas as core_schemas
from LUMIA.ml.classifier import (
    ClassificationResult,
    CutoutClassificationBatchResult,
    RealBogusClassifier,
    classify_cutout_batch,
)
from LUMIA.ml.config import CNNClassifierConfig
from LUMIA.physics.cutout_extraction import (
    CutoutBatchResult,
    CutoutConfig,
    PeakExtractionConfig,
    SNRPeakCandidate,
    extract_snr_peaks,
    extract_zogy_cutout_batch,
    slice_cutout_triplets,
)
from LUMIA.physics.gpu_processing import ZOGYConfig, ZOGYResult, zogy_subtract

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def _make_gaussian_psf(size: int, sigma_px: float) -> np.ndarray:
    """Create a normalized 2D circular Gaussian PSF centered at (size // 2, size // 2)."""
    center: int = size // 2
    y_grid, x_grid = np.mgrid[0:size, 0:size]
    r2: np.ndarray = (x_grid - center) ** 2 + (y_grid - center) ** 2
    psf: np.ndarray = np.exp(-0.5 * r2 / (sigma_px**2))
    psf /= np.sum(psf)
    return psf.astype(np.float32)


def _inject_gaussian_source(
    image: np.ndarray,
    x0: float,
    y0: float,
    flux: float,
    sigma_px: float,
) -> None:
    """Inject an analytical 2D Gaussian source into a 2D image array."""
    ny, nx = image.shape
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]
    norm: float = flux / (2.0 * math.pi * sigma_px**2)
    r2: np.ndarray = (x_grid - x0) ** 2 + (y_grid - y0) ** 2
    image += (norm * np.exp(-0.5 * r2 / (sigma_px**2))).astype(np.float32)


def _make_synthetic_zogy_result(
    height: int = 128,
    width: int = 128,
) -> tuple[np.ndarray, np.ndarray, ZOGYResult]:
    """Build deterministic science, reference, and ZOGYResult frames with known peaks."""
    rng = np.random.default_rng(2026)
    science = rng.normal(100.0, 1.0, size=(height, width)).astype(np.float32)
    reference = rng.normal(100.0, 1.0, size=(height, width)).astype(np.float32)

    # Static field star present in both science and reference (cancels in ZOGY)
    _inject_gaussian_source(science, 64.0, 64.0, flux=800.0, sigma_px=1.6)
    _inject_gaussian_source(reference, 64.0, 64.0, flux=800.0, sigma_px=1.6)

    # Moving-object transients present only in science epoch
    _inject_gaussian_source(science, 40.0, 50.0, flux=180.0, sigma_px=1.6)
    _inject_gaussian_source(science, 92.0, 84.0, flux=110.0, sigma_px=1.6)
    # Faint sub-threshold bump
    _inject_gaussian_source(science, 25.0, 95.0, flux=12.0, sigma_px=1.6)

    psf = _make_gaussian_psf(15, 1.6)
    zogy_cfg = ZOGYConfig(
        flux_scale_science=1.0,
        flux_scale_reference=1.0,
        noise_std_science=1.0,
        noise_std_reference=1.0,
    )
    zogy_res = zogy_subtract(science, reference, psf, psf, zogy_cfg)
    return science, reference, zogy_res


class TestDeterministicSNRPeakExtraction:
    """Verify SNR peak finding, non-maximum suppression, tie-breaking, and determinism."""

    def test_extracts_supra_threshold_peaks_sorted_and_deterministic(self) -> None:
        """Verify peaks above threshold are extracted in descending SNR order deterministically."""
        _, _, zogy_res = _make_synthetic_zogy_result()
        cfg = PeakExtractionConfig(
            snr_threshold=6.0,
            nms_radius_px=3,
            id_prefix="ztf-cand",
        )

        peaks_run1: list[SNRPeakCandidate] = extract_snr_peaks(zogy_res, cfg)
        peaks_run2: list[SNRPeakCandidate] = extract_snr_peaks(
            zogy_res.significance_map,
            cfg,
        )

        assert peaks_run1 == peaks_run2
        assert len(peaks_run1) == 2

        # Brighter transient at (40, 50) must rank first, fainter at (92, 84) second
        first, second = peaks_run1[0], peaks_run1[1]
        assert first.snr > second.snr >= 6.0
        assert first.candidate_id == "ztf-cand-0000"
        assert second.candidate_id == "ztf-cand-0001"

        assert abs(first.x_px - 40) <= 1 and abs(first.y_px - 50) <= 1
        assert abs(second.x_px - 92) <= 1 and abs(second.y_px - 84) <= 1
        assert first.x_centroid_px == pytest.approx(40.0, abs=0.35)
        assert first.y_centroid_px == pytest.approx(50.0, abs=0.35)

    def test_nms_breaks_plateau_ties_and_respects_max_peaks(self) -> None:
        """Verify flat-top plateaus within nms_radius_px collapse to one deterministic peak."""
        sig_map = torch.zeros((48, 48), dtype=torch.float32)
        # 2x2 exact tie plateau at SNR = 9.5
        sig_map[20:22, 15:17] = 9.5
        # Distinct peak at SNR = 12.0
        sig_map[35, 30] = 12.0

        dummy_psf = torch.zeros((48, 48), dtype=torch.float32)
        dummy_psf[24, 24] = 1.0
        zogy_res = ZOGYResult(
            difference_image=sig_map.clone(),
            significance_map=sig_map,
            difference_psf=dummy_psf,
            theoretical_noise_std=1.0,
            flux_scale_diff=1.0,
        )

        peaks = extract_snr_peaks(
            zogy_res,
            PeakExtractionConfig(snr_threshold=5.0, nms_radius_px=2),
        )
        assert len(peaks) == 2
        assert peaks[0].x_px == 30 and peaks[0].y_px == 35
        # Plateau tie breaks deterministically to top-leftmost coordinate (y=20, x=15)
        assert peaks[1].x_px == 15 and peaks[1].y_px == 20

        capped = extract_snr_peaks(
            zogy_res,
            PeakExtractionConfig(snr_threshold=5.0, nms_radius_px=2, max_peaks=1),
        )
        assert len(capped) == 1
        assert capped[0].x_px == 30 and capped[0].y_px == 35

    def test_nms_reduces_broad_source_cluster_to_single_peak(self) -> None:
        """A broad asteroid where >20 pixels exceed SNR threshold collapses to 1 peak."""
        h, w = 64, 64
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        r2 = (x_grid - 32.0) ** 2 + (y_grid - 32.0) ** 2
        sig_map = (15.0 * np.exp(-0.5 * r2 / (2.5**2))).astype(np.float32)

        n_pixels_above = int(np.sum(sig_map >= 5.0))
        assert n_pixels_above >= 20, f"Expected >= 20 pixels > 5.0, got {n_pixels_above}"

        cfg = PeakExtractionConfig(snr_threshold=5.0, nms_radius_px=3)
        peaks = extract_snr_peaks(sig_map, cfg)

        assert len(peaks) == 1
        peak = peaks[0]
        assert peak.x_px == 32
        assert peak.y_px == 32
        assert peak.snr == pytest.approx(15.0, rel=1e-3)
        assert peak.x_centroid_px == pytest.approx(32.0, abs=0.1)
        assert peak.y_centroid_px == pytest.approx(32.0, abs=0.1)

    def test_zogy_result_convenience_methods(self) -> None:
        """ZOGYResult.extract_peaks() and extract_cutouts() delegate seamlessly in-memory."""
        science, reference, zogy_res = _make_synthetic_zogy_result(64, 64)

        peaks = zogy_res.extract_peaks(PeakExtractionConfig(snr_threshold=6.0))
        direct_peaks = extract_snr_peaks(zogy_res, PeakExtractionConfig(snr_threshold=6.0))
        assert peaks == direct_peaks

        batch = zogy_res.extract_cutouts(
            science,
            reference,
            peak_config=PeakExtractionConfig(snr_threshold=6.0),
            cutout_config=CutoutConfig(stamp_size=31, boundary_mode="pad"),
        )
        direct_batch = extract_zogy_cutout_batch(
            science,
            reference,
            zogy_res,
            peak_config=PeakExtractionConfig(snr_threshold=6.0),
            cutout_config=CutoutConfig(stamp_size=31, boundary_mode="pad"),
        )
        assert batch.triplets.shape == direct_batch.triplets.shape
        assert batch.candidate_ids == direct_batch.candidate_ids


class TestVectorizedCutoutTripletSlicing:
    """Verify vectorized (B, 3, H, W) odd-dimension slicing and boundary modes."""

    @pytest.mark.parametrize("stamp_size", [31, 51])
    def test_slices_exact_aligned_three_channel_odd_triplets(
        self,
        stamp_size: int,
    ) -> None:
        """Verify Channel 0/1/2 pixel alignment and center-pixel symmetry for 31x31 and 51x51."""
        height, width = 120, 130
        y_grid, x_grid = np.mgrid[0:height, 0:width]
        # Distinct deterministic encodings per channel so any channel/axis swap is caught
        science = (1000.0 + y_grid * 10.0 + x_grid * 0.1).astype(np.float32)
        reference = (2000.0 + y_grid * 10.0 + x_grid * 0.1).astype(np.float32)
        difference = (3000.0 + y_grid * 10.0 + x_grid * 0.1).astype(np.float32)

        science_orig = science.copy()
        peaks = [
            SNRPeakCandidate(
                candidate_id="cand-a",
                x_px=55,
                y_px=45,
                x_centroid_px=55.0,
                y_centroid_px=45.0,
                snr=11.2,
            ),
            SNRPeakCandidate(
                candidate_id="cand-b",
                x_px=80,
                y_px=70,
                x_centroid_px=80.0,
                y_centroid_px=70.0,
                snr=7.9,
            ),
        ]

        batch: CutoutBatchResult = slice_cutout_triplets(
            science,
            reference,
            difference,
            peaks,
            CutoutConfig(stamp_size=stamp_size),  # type: ignore[arg-type]
        )

        assert batch.triplets.dtype == torch.float32
        assert tuple(batch.triplets.shape) == (2, 3, stamp_size, stamp_size)
        assert batch.stamp_size == stamp_size
        assert batch.candidate_ids == ["cand-a", "cand-b"]
        assert batch.rejected_boundary_count == 0
        # Pure non-mutating check
        np.testing.assert_array_equal(science, science_orig)

        half = stamp_size // 2
        for idx, peak in enumerate(peaks):
            assert float(batch.triplets[idx, 0, half, half].item()) == pytest.approx(
                float(science[peak.y_px, peak.x_px])
            )
            assert float(batch.triplets[idx, 1, half, half].item()) == pytest.approx(
                float(reference[peak.y_px, peak.x_px])
            )
            assert float(batch.triplets[idx, 2, half, half].item()) == pytest.approx(
                float(difference[peak.y_px, peak.x_px])
            )
            # Top-left corner of stamp matches (y - half, x - half)
            assert float(batch.triplets[idx, 0, 0, 0].item()) == pytest.approx(
                float(science[peak.y_px - half, peak.x_px - half])
            )

    def test_boundary_modes_pad_edge_and_reject_near_frame_borders(self) -> None:
        """Verify peaks near image edges are padded, edge-clamped, or rejected cleanly."""
        science = np.ones((64, 64), dtype=np.float32) * 5.0
        reference = np.ones((64, 64), dtype=np.float32) * 3.0
        difference = np.ones((64, 64), dtype=np.float32) * 2.0
        science[2, 3] = 99.0  # Near top-left corner (within 15px of border)
        science[32, 32] = 77.0  # Safe interior center

        peaks = [
            SNRPeakCandidate(
                candidate_id="edge-peak",
                x_px=3,
                y_px=2,
                x_centroid_px=3.0,
                y_centroid_px=2.0,
                snr=14.0,
            ),
            SNRPeakCandidate(
                candidate_id="interior-peak",
                x_px=32,
                y_px=32,
                x_centroid_px=32.0,
                y_centroid_px=32.0,
                snr=9.0,
            ),
        ]

        # 1. Constant padding mode preserves peak at stamp center (15, 15) and
        #    fills out-of-frame pixels with pad_value.
        padded_batch = slice_cutout_triplets(
            science,
            reference,
            difference,
            peaks,
            CutoutConfig(stamp_size=31, boundary_mode="pad", pad_value=-1.0),
        )
        assert tuple(padded_batch.triplets.shape) == (2, 3, 31, 31)
        assert padded_batch.rejected_boundary_count == 0
        # Stamp (0, 0) corresponds to frame (2 - 15, 3 - 15) = (-13, -12) -> padded with -1.0
        assert float(padded_batch.triplets[0, 0, 0, 0].item()) == pytest.approx(-1.0)
        assert float(padded_batch.triplets[0, 1, 0, 0].item()) == pytest.approx(-1.0)
        assert float(padded_batch.triplets[0, 2, 0, 0].item()) == pytest.approx(-1.0)
        # Center of stamp (15, 15) is the peak at frame[2, 3] == 99.0
        assert float(padded_batch.triplets[0, 0, 15, 15].item()) == pytest.approx(99.0)

        # 2. Edge-clamping mode replicates border values without wrap-around
        edge_batch = slice_cutout_triplets(
            science,
            reference,
            difference,
            peaks,
            CutoutConfig(stamp_size=31, boundary_mode="edge"),
        )
        assert tuple(edge_batch.triplets.shape) == (2, 3, 31, 31)
        assert float(edge_batch.triplets[0, 0, 0, 0].item()) == pytest.approx(5.0)
        assert float(edge_batch.triplets[0, 0, 15, 15].item()) == pytest.approx(99.0)

        # 3. Reject mode drops boundary-straddling candidates and keeps interior candidates
        rejected_batch = slice_cutout_triplets(
            science,
            reference,
            difference,
            peaks,
            CutoutConfig(stamp_size=31, boundary_mode="reject"),
        )
        assert tuple(rejected_batch.triplets.shape) == (1, 3, 31, 31)
        assert rejected_batch.rejected_boundary_count == 1
        assert rejected_batch.candidate_ids == ["interior-peak"]
        assert float(rejected_batch.triplets[0, 0, 15, 15].item()) == pytest.approx(77.0)

    def test_empty_peaks_and_strict_validation_errors(self) -> None:
        """Verify empty peak list returns (0, 3, S, S) and invalid configs/inputs fail fast."""
        frame = np.zeros((64, 64), dtype=np.float32)
        empty_batch = slice_cutout_triplets(
            frame,
            frame,
            frame,
            [],
            CutoutConfig(stamp_size=51),
        )
        assert tuple(empty_batch.triplets.shape) == (0, 3, 51, 51)
        assert empty_batch.candidates == []

        # Even stamp sizes (e.g., 32) must be rejected by Pydantic schema
        with pytest.raises(ValidationError):
            CutoutConfig(stamp_size=32)  # type: ignore[arg-type]

        # NaN pad_value or extra fields must be rejected
        with pytest.raises(ValidationError):
            CutoutConfig(stamp_size=31, pad_value=float("nan"))

        with pytest.raises(ValidationError):
            PeakExtractionConfig(snr_threshold=float("inf"))

        # Mismatched frame shapes must raise ValueError
        with pytest.raises(ValueError, match="shape"):
            slice_cutout_triplets(
                frame,
                np.zeros((64, 60), dtype=np.float32),
                frame,
                [],
            )

    def test_extreme_corner_and_border_peaks_with_pad_and_reject(self) -> None:
        """Verify peaks located precisely on the 4 frame corners (0,0), (W-1, 0), etc."""
        h, w = 64, 64
        science = np.full((h, w), 10.0, dtype=np.float32)
        reference = np.full((h, w), 5.0, dtype=np.float32)
        difference = np.full((h, w), 1.0, dtype=np.float32)

        # Four extreme corners
        corner_peaks = [
            SNRPeakCandidate(candidate_id="tl", x_px=0, y_px=0, x_centroid_px=0.0, y_centroid_px=0.0, snr=10.0),
            SNRPeakCandidate(candidate_id="tr", x_px=w - 1, y_px=0, x_centroid_px=float(w - 1), y_centroid_px=0.0, snr=10.0),
            SNRPeakCandidate(candidate_id="bl", x_px=0, y_px=h - 1, x_centroid_px=0.0, y_centroid_px=float(h - 1), snr=10.0),
            SNRPeakCandidate(candidate_id="br", x_px=w - 1, y_px=h - 1, x_centroid_px=float(w - 1), y_centroid_px=float(h - 1), snr=10.0),
        ]

        # 1. Pad mode
        batch_pad = slice_cutout_triplets(
            science,
            reference,
            difference,
            corner_peaks,
            CutoutConfig(stamp_size=31, boundary_mode="pad", pad_value=-99.0),
        )
        assert tuple(batch_pad.triplets.shape) == (4, 3, 31, 31)
        assert batch_pad.rejected_boundary_count == 0
        assert float(batch_pad.triplets[0, 0, 15, 15].item()) == pytest.approx(10.0)
        assert float(batch_pad.triplets[0, 0, 0, 0].item()) == pytest.approx(-99.0)

        # 2. Reject mode
        batch_rej = slice_cutout_triplets(
            science,
            reference,
            difference,
            corner_peaks,
            CutoutConfig(stamp_size=31, boundary_mode="reject"),
        )
        assert tuple(batch_rej.triplets.shape) == (0, 3, 31, 31)
        assert batch_rej.rejected_boundary_count == 4
        assert batch_rej.candidates == []


class TestModule4ToModule7ClassifierBridge:
    """Verify end-to-end ZOGY -> Cutout Extraction -> RealBogusClassifier.predict() flow."""

    @pytest.mark.parametrize(
        ("backbone", "stamp_size"),
        [
            ("vgg", 31),
            ("resnet_lite", 51),
        ],
    )
    def test_extracted_zogy_triplets_feed_directly_into_real_bogus_classifier(
        self,
        backbone: str,
        stamp_size: int,
    ) -> None:
        """Verify extracted ZOGY triplets classify cleanly through VGG and ResNet-Lite."""
        science, reference, zogy_res = _make_synthetic_zogy_result(128, 128)

        batch = extract_zogy_cutout_batch(
            science,
            reference,
            zogy_res,
            peak_config=PeakExtractionConfig(snr_threshold=6.0, id_prefix="m4-m7"),
            cutout_config=CutoutConfig(
                stamp_size=stamp_size,  # type: ignore[arg-type]
                boundary_mode="pad",
            ),
        )
        assert len(batch.candidates) == 2
        assert tuple(batch.triplets.shape) == (2, 3, stamp_size, stamp_size)

        clf_cfg = CNNClassifierConfig(
            backbone=backbone,  # type: ignore[arg-type]
            stamp_size=stamp_size,  # type: ignore[arg-type]
            device="cpu",
            deterministic=True,
            seed=123,
            model_version="lumia-rb-bridge-test",
        )
        classifier = RealBogusClassifier(clf_cfg)
        predictions: list[ClassificationResult] = classifier.predict(
            batch.triplets,
            batch.candidate_ids,
        )

        assert len(predictions) == 2
        for pred, expected_id in zip(predictions, batch.candidate_ids, strict=True):
            assert pred.candidate_id == expected_id
            assert 0.0 <= pred.score <= 1.0
            assert pred.model_version == "lumia-rb-bridge-test"

        # Direct CutoutBatchResult contract call and pure classify_cutout_batch bridge
        direct_predictions = classifier.predict(batch)
        assert direct_predictions == predictions

        bridge_result: CutoutClassificationBatchResult = classify_cutout_batch(
            classifier,
            batch,
        )
        assert list(bridge_result.classifications) == predictions
        assert bridge_result.stamp_size == stamp_size
        assert bridge_result.model_version == "lumia-rb-bridge-test"

        # Deterministic repeat run check
        repeat_predictions = classifier.predict(batch.triplets, batch.candidate_ids)
        assert predictions == repeat_predictions

    def test_configurable_snr_threshold_negative_dips_and_border_margin(self) -> None:
        """Verify threshold filtering, negative reference dip rejection, and border margins."""
        sig_map = torch.zeros((64, 64), dtype=torch.float32)
        sig_map[3, 4] = 15.0  # Near detector border
        sig_map[30, 30] = -18.0  # Strong negative reference-star subtraction dip
        sig_map[22, 40] = 6.2  # Moderate transient
        sig_map[45, 18] = 11.4  # Bright transient

        low_thresh_peaks = extract_snr_peaks(
            sig_map,
            PeakExtractionConfig(snr_threshold=5.5, nms_radius_px=2, border_margin_px=0),
        )
        assert [(p.x_px, p.y_px) for p in low_thresh_peaks] == [
            (4, 3),
            (18, 45),
            (40, 22),
        ]

        # Raising threshold excludes 6.2-sigma peak; border_margin_px=6 excludes (4, 3)
        filtered_peaks = extract_snr_peaks(
            sig_map,
            PeakExtractionConfig(snr_threshold=8.0, nms_radius_px=2, border_margin_px=6),
        )
        assert len(filtered_peaks) == 1
        assert (filtered_peaks[0].x_px, filtered_peaks[0].y_px) == (18, 45)

        # Oversized border margin exceeding half-frame returns empty list cleanly
        assert (
            extract_snr_peaks(
                sig_map,
                PeakExtractionConfig(snr_threshold=5.0, border_margin_px=32),
            )
            == []
        )

    def test_difference_channel_selection_duplicate_ids_and_non_aliasing(self) -> None:
        """Verify significance_map channel selection, duplicate ID guard, and tensor isolation."""
        science = torch.full((64, 64), 10.0, dtype=torch.float32)
        reference = torch.full((64, 64), 4.0, dtype=torch.float32)
        diff_img = torch.full((64, 64), 6.0, dtype=torch.float32)
        sig_map = torch.zeros((64, 64), dtype=torch.float32)
        sig_map[32, 32] = 9.5
        psf = torch.zeros((64, 64), dtype=torch.float32)
        psf[32, 32] = 1.0

        zogy_res = ZOGYResult(
            difference_image=diff_img,
            significance_map=sig_map,
            difference_psf=psf,
            theoretical_noise_std=1.0,
            flux_scale_diff=1.0,
        )

        batch_sig = extract_zogy_cutout_batch(
            science,
            reference,
            zogy_res,
            peak_config=PeakExtractionConfig(snr_threshold=5.0),
            cutout_config=CutoutConfig(
                stamp_size=31,
                difference_channel="significance_map",
            ),
        )
        assert float(batch_sig.triplets[0, 2, 15, 15].item()) == pytest.approx(9.5)
        assert batch_sig.triplets.is_contiguous()
        assert batch_sig.triplets.data_ptr() != science.data_ptr()

        # Mutating returned triplets never mutates input frames
        batch_sig.triplets[0, 0, 15, 15] = -999.0
        assert float(science[32, 32].item()) == pytest.approx(10.0)

        # Duplicate candidate IDs in CutoutBatchResult must be rejected
        dup_peak = SNRPeakCandidate(
            candidate_id="dup-0001",
            x_px=20,
            y_px=20,
            x_centroid_px=20.0,
            y_centroid_px=20.0,
            snr=8.0,
        )
        with pytest.raises(ValidationError, match="Duplicate|unique"):
            CutoutBatchResult(
                triplets=torch.zeros((2, 3, 31, 31), dtype=torch.float32),
                candidates=[dup_peak, dup_peak],
                stamp_size=31,
            )

        # Out-of-frame peak coordinates fail fast
        oob_peak = SNRPeakCandidate(
            candidate_id="oob-0001",
            x_px=64,
            y_px=20,
            x_centroid_px=64.0,
            y_centroid_px=20.0,
            snr=8.0,
        )
        with pytest.raises(ValueError, match="outside frame"):
            slice_cutout_triplets(science, reference, diff_img, [oob_peak])

    def test_in_memory_bridge_performs_zero_filesystem_io_and_shares_core_contracts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify extraction + classification executes with zero disk I/O and shared contracts."""
        assert CutoutBatchResult is core_schemas.CutoutBatchResult
        assert SNRPeakCandidate is core_schemas.SNRPeakCandidate
        assert ClassificationResult is core_schemas.ClassificationResult

        science, reference, zogy_res = _make_synthetic_zogy_result(96, 96)
        classifier = RealBogusClassifier(
            CNNClassifierConfig(backbone="vgg", stamp_size=31, device="cpu", seed=7)
        )

        def _forbid_open(*args: object, **kwargs: object) -> object:
            msg = f"Unexpected filesystem open during in-memory bridge: {args!r}"
            raise AssertionError(msg)

        monkeypatch.setattr(builtins, "open", _forbid_open)
        monkeypatch.setattr(Path, "open", _forbid_open)

        batch = extract_zogy_cutout_batch(
            science,
            reference,
            zogy_res,
            peak_config=PeakExtractionConfig(snr_threshold=6.0, id_prefix="no-io"),
            cutout_config=CutoutConfig(stamp_size=31, boundary_mode="pad"),
        )
        classified = classify_cutout_batch(classifier, batch)
        assert len(classified.classifications) == len(batch.candidates)

    def test_physics_package_never_imports_ml_modules_or_weights(self) -> None:
        """AST guard: src/LUMIA/physics must never import LUMIA.ml."""
        physics_dir = _REPO_ROOT / "src" / "LUMIA" / "physics"
        violations: list[str] = []
        for py_file in sorted(physics_dir.rglob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "LUMIA.ml" or alias.name.startswith("LUMIA.ml."):
                            violations.append(f"{py_file.name}:{node.lineno} -> {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if mod == "LUMIA.ml" or mod.startswith("LUMIA.ml."):
                        violations.append(f"{py_file.name}:{node.lineno} -> {mod}")
        assert violations == []

