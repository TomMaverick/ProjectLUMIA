"""Advanced artifact and false-positive filtering pipeline inspired by DeepStreaks."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import Field
from scipy import ndimage

from LUMIA.core.schemas import LumiaBaseModel

ArtifactType = Literal[
    "NONE",
    "SATELLITE_TRAIL",
    "AIRPLANE_PATH",
    "COSMIC_RAY",
    "OPTICAL_GHOST",
    "DIPOLE",
    "BLEEDING_COLUMN",
    "DIFFRACTION_SPIKE",
]

ARTIFACT_TAXONOMY_MAP: dict[ArtifactType, str] = {
    "SATELLITE_TRAIL": "ARTIFACT_SATELLITE",
    "AIRPLANE_PATH": "ARTIFACT_AIRPLANE",
    "COSMIC_RAY": "ARTIFACT_COSMIC_RAY",
    "OPTICAL_GHOST": "ARTIFACT_GHOST",
    "DIPOLE": "ARTIFACT_DIPOLE",
    "BLEEDING_COLUMN": "ARTIFACT_BLEEDING_COLUMN",
    "DIFFRACTION_SPIKE": "ARTIFACT_DIFFRACTION_SPIKE",
    "NONE": "REAL_SOURCE",
}


class ArtifactAssessment(LumiaBaseModel):
    """Validation report and categorization of instrumental or anthropogenic artifacts."""

    is_artifact: bool = Field(
        ..., description="True if cutout should be quarantined as a false-positive."
    )
    artifact_type: ArtifactType = Field(default="NONE")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the artifact classification."
    )
    streak_score: float = Field(
        default=0.0, ge=0.0, description="Hough/Radon linear streak metric."
    )
    laplacian_contrast: float = Field(
        default=0.0, ge=0.0, description="L.A.Cosmic sub-PSF edge sharpness."
    )
    strobe_periodicity_score: float = Field(
        default=0.0, ge=0.0, description="Aircraft strobe metric."
    )
    ghost_radial_score: float = Field(
        default=0.0, ge=0.0, description="Optical reflection halo metric."
    )
    dipole_score: float = Field(
        default=0.0, ge=0.0, description="Difference image dipole lobe ratio."
    )
    bleeding_column_score: float = Field(
        default=0.0, ge=0.0, description="CCD bleeding charge column excess."
    )
    diffraction_spike_score: float = Field(
        default=0.0, ge=0.0, description="Linear diffraction spike prominence."
    )
    reasons: list[str] = Field(default_factory=list)

    @property
    def hard_negative_taxonomy(self) -> str:
        """Return the specific taxonomy string for HDF5 hard-negative labeling."""
        return ARTIFACT_TAXONOMY_MAP.get(self.artifact_type, "ARTIFACT_UNKNOWN")


class ArtifactFilterConfig(LumiaBaseModel):
    """Configuration thresholds for the multi-stage artifact firewall."""

    streak_threshold: float = Field(
        default=4.5, gt=0.0, description="Radon/Hough peak-to-background ratio."
    )
    min_streak_length_px: int = Field(
        default=18, ge=5, description="Minimum continuous trail length in pixels."
    )
    border_margin_px: int = Field(
        default=4, ge=1, le=16, description="Border margin for unbounded satellite track detection."
    )
    max_bounded_streak_px: int = Field(
        default=75, ge=20, le=120, description="Max allowed length for bounded NEO streak."
    )
    laplacian_cr_threshold: float = Field(
        default=4.8, gt=0.0, description="L.A.Cosmic edge ratio threshold."
    )
    min_cosmic_ray_snr: float = Field(default=5.0, gt=0.0)
    strobe_peak_min_count: int = Field(default=3, ge=2)
    ghost_ring_ratio_threshold: float = Field(default=2.5, gt=0.0)
    dipole_threshold: float = Field(
        default=0.45, gt=0.0, description="Minimum lobe ratio for dipole detection."
    )
    bleeding_column_threshold: float = Field(
        default=4.0, gt=0.0, description="Column excess signal-to-noise ratio."
    )
    diffraction_spike_threshold: float = Field(
        default=3.5, gt=0.0, description="Diffraction spike linear prominence."
    )
    psf_fwhm_px: float = Field(
        default=2.5, gt=0.0, description="Nominal seeing PSF FWHM in pixels."
    )


class DeepStreaksArtifactFilter:
    """Multi-stage machine vision artifact filter for difference and science cutouts."""

    def __init__(self, config: ArtifactFilterConfig | None = None) -> None:
        self.config = config if config is not None else ArtifactFilterConfig()

    def evaluate_cutout(
        self,
        difference_stamp: np.ndarray,
        science_stamp: np.ndarray | None = None,
    ) -> ArtifactAssessment:
        """Execute full artifact detection cascade on 2D image stamps.

        Args:
            difference_stamp: 2D difference or significance cutout (H, W).
            science_stamp: Optional 2D calibrated science cutout (H, W).

        Returns:
            ArtifactAssessment containing rejection decision and artifact type.
        """
        diff = difference_stamp.astype(np.float32)
        reasons: list[str] = []

        # 1. Saturated Bleeding Column Detection (CCD charge leakage)
        bleed_score, bleed_detected = self._detect_bleeding_column(diff)
        if bleed_detected:
            reasons.append(f"Detected CCD bleeding charge column (Excess SNR: {bleed_score:.2f})")
            return ArtifactAssessment(
                is_artifact=True,
                artifact_type="BLEEDING_COLUMN",
                confidence=0.96,
                bleeding_column_score=bleed_score,
                reasons=reasons,
            )

        # 2. Difference Image Dipole Detection (paired positive/negative subtraction residuals)
        dipole_score, dipole_detected = self._detect_dipole(diff)
        if dipole_detected:
            reasons.append(f"Detected subtraction dipole (Lobe ratio: {dipole_score:.2f})")
            return ArtifactAssessment(
                is_artifact=True,
                artifact_type="DIPOLE",
                confidence=min(0.99, float(dipole_score * 0.9)),
                dipole_score=dipole_score,
                reasons=reasons,
            )

        # 3. Satellite Trail Detection (Radon / Hough Linear Transform)
        streak_score, streak_detected = self._detect_satellite_streak(diff)
        if streak_detected:
            reasons.append(f"Detected linear satellite trail (Radon score: {streak_score:.2f})")
            return ArtifactAssessment(
                is_artifact=True,
                artifact_type="SATELLITE_TRAIL",
                confidence=min(0.99, float(streak_score / self.config.streak_threshold * 0.8)),
                streak_score=streak_score,
                reasons=reasons,
            )

        # 4. Airplane Strobe Path Detection
        strobe_score, strobe_detected = self._detect_airplane_strobe(diff)
        if strobe_detected:
            reasons.append(f"Detected periodic strobe flight path (Score: {strobe_score:.2f})")
            return ArtifactAssessment(
                is_artifact=True,
                artifact_type="AIRPLANE_PATH",
                confidence=0.92,
                strobe_periodicity_score=strobe_score,
                reasons=reasons,
            )

        # 5. Cosmic Ray Detection (L.A.Cosmic Sub-PSF Laplacian)
        cr_score, cr_detected = self._detect_cosmic_ray(diff)
        if cr_detected:
            reasons.append(f"Detected sharp sub-PSF cosmic ray spike (Laplacian: {cr_score:.2f})")
            return ArtifactAssessment(
                is_artifact=True,
                artifact_type="COSMIC_RAY",
                confidence=min(0.99, float(cr_score / self.config.laplacian_cr_threshold * 0.8)),
                laplacian_contrast=cr_score,
                reasons=reasons,
            )

        # 6. Optical Ghost and Diffraction Spike Halo Detection
        sci = science_stamp if science_stamp is not None else diff
        ghost_score, ghost_detected = self._detect_optical_ghost(sci)
        if ghost_detected:
            reasons.append(
                f"Detected diffuse optical ghost / bright star halo (Ratio: {ghost_score:.2f})"
            )
            return ArtifactAssessment(
                is_artifact=True,
                artifact_type="OPTICAL_GHOST",
                confidence=0.88,
                ghost_radial_score=ghost_score,
                reasons=reasons,
            )

        # 7. Diffraction Spike Detection
        spike_score, spike_detected = self._detect_diffraction_spike(diff)
        if spike_detected:
            reasons.append(f"Detected linear diffraction spike (Score: {spike_score:.2f})")
            return ArtifactAssessment(
                is_artifact=True,
                artifact_type="DIFFRACTION_SPIKE",
                confidence=0.91,
                diffraction_spike_score=spike_score,
                reasons=reasons,
            )

        return ArtifactAssessment(
            is_artifact=False,
            artifact_type="NONE",
            confidence=0.95,
            streak_score=streak_score,
            laplacian_contrast=cr_score,
            strobe_periodicity_score=strobe_score,
            ghost_radial_score=ghost_score,
            dipole_score=dipole_score,
            bleeding_column_score=bleed_score,
            diffraction_spike_score=spike_score,
            reasons=["Clear of detectable instrumental/anthropogenic artifacts."],
        )

    def _detect_satellite_streak(self, image_2d: np.ndarray) -> tuple[float, bool]:
        """Detect bright, continuous linear streaks using a discrete Radon projection."""
        med = float(np.median(image_2d))
        mad = float(np.median(np.abs(image_2d - med)))
        std = max(1.4826 * mad, 1e-6)

        # Binary thresholded edge mask for linear feature extraction
        bright_mask = (image_2d - med) > (2.5 * std)
        if np.sum(bright_mask) < self.config.min_streak_length_px:
            return 0.0, False

        # Compute Radon-like linear projections across angles 0 to 180 degrees
        n_angles = 36
        angles_deg = np.linspace(0, 180, n_angles, endpoint=False)
        projections: list[float] = []

        h, w = image_2d.shape
        cy, cx = h / 2.0, w / 2.0
        y_coords, x_coords = np.indices((h, w))
        x_rel = x_coords - cx
        y_rel = y_coords - cy

        norm_img = np.clip((image_2d - med) / std, 0.0, 50.0)

        for ang in angles_deg:
            rad = math.radians(ang)
            # Distance of each pixel to line along angle
            proj_dist = x_rel * math.cos(rad) + y_rel * math.sin(rad)
            # Bin into 1D profile
            hist, _ = np.histogram(proj_dist, bins=h // 2, weights=norm_img)
            projections.append(float(np.max(hist)))

        max_peak = max(projections) if projections else 0.0
        median_proj = float(np.median(projections)) if projections else 1.0
        streak_ratio = max_peak / max(median_proj, 1.0)

        # Verify minimum physical length of continuous streak and unbounded geometry
        labeled, n_features = ndimage.label(bright_mask)
        max_extent = 0
        is_unbounded = False
        margin = self.config.border_margin_px

        if n_features > 0:
            for feat_idx in range(1, n_features + 1):
                feat_pts = np.argwhere(labeled == feat_idx)
                if len(feat_pts) == 0:
                    continue
                span = np.max(feat_pts, axis=0) - np.min(feat_pts, axis=0)
                extent = int(np.hypot(span[0], span[1]))
                if extent > max_extent:
                    max_extent = extent

                # Check if this streak crosses outer borders (LEO satellite signature)
                if extent >= self.config.min_streak_length_px:
                    touch_top = bool(np.any(feat_pts[:, 0] <= margin))
                    touch_bottom = bool(np.any(feat_pts[:, 0] >= h - 1 - margin))
                    touch_left = bool(np.any(feat_pts[:, 1] <= margin))
                    touch_right = bool(np.any(feat_pts[:, 1] >= w - 1 - margin))
                    n_borders = sum([touch_top, touch_bottom, touch_left, touch_right])

                    # Satellite tracks cross at least 2 borders or span past max_bounded_streak_px
                    if (
                        n_borders >= 2
                        or extent >= self.config.max_bounded_streak_px
                        or streak_ratio >= 15.0
                    ):
                        is_unbounded = True

        streak_detected = (
            (streak_ratio >= self.config.streak_threshold)
            and (max_extent >= self.config.min_streak_length_px)
            and is_unbounded
        )
        return streak_ratio, streak_detected

    def _detect_airplane_strobe(self, image_2d: np.ndarray) -> tuple[float, bool]:
        """Detect collinear periodic strobe flash sequences characteristic of aircraft."""
        med = float(np.median(image_2d))
        mad = float(np.median(np.abs(image_2d - med)))
        std = max(1.4826 * mad, 1e-6)

        peaks_y, peaks_x = np.where((image_2d - med) > (4.0 * std))
        if len(peaks_x) < self.config.strobe_peak_min_count:
            return 0.0, False

        # Check collinearity and spacing regularity of top peaks
        intensities = image_2d[peaks_y, peaks_x]
        sorted_indices = np.argsort(intensities)[::-1][:10]
        pts_x = peaks_x[sorted_indices]
        pts_y = peaks_y[sorted_indices]

        if len(pts_x) < 3:
            return 0.0, False

        # Fit line through peaks
        try:
            poly = np.polyfit(pts_x, pts_y, 1, full=True)
            residuals = poly[1]
            residual_val = float(residuals[0]) if residuals.size > 0 else 100.0

            # Linearity check
            if residual_val < 5.0 and len(pts_x) >= 3:
                # Compute adjacent pairwise distances
                dists = np.sqrt(np.diff(pts_x) ** 2 + np.diff(pts_y) ** 2)
                if len(dists) >= 2:
                    dist_std = float(np.std(dists))
                    dist_mean = float(np.mean(dists))
                    if dist_mean > 3.0 and (dist_std / dist_mean) < 0.35:
                        return float(dist_mean), True
        except Exception:
            pass

        return 0.0, False

    def _detect_cosmic_ray(self, image_2d: np.ndarray) -> tuple[float, bool]:
        """L.A.Cosmic algorithm: evaluates Laplacian edge sharpness to reject single-pixel hits."""
        laplacian_kernel = np.array(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float32
        )
        lap = ndimage.convolve(image_2d, laplacian_kernel, mode="nearest")

        med = float(np.median(image_2d))
        mad = float(np.median(np.abs(image_2d - med)))
        noise_model = max(1.4826 * mad, 1e-6)

        # Contrast of Laplacian to noise
        contrast_map = lap / (2.0 * noise_model)
        max_contrast = float(np.max(contrast_map))

        # Check sharpness/FWHM around the peak
        py, px = np.unravel_index(np.argmax(contrast_map), image_2d.shape)
        y0 = max(0, py - 2)
        y1 = min(image_2d.shape[0], py + 3)
        x0 = max(0, px - 2)
        x1 = min(image_2d.shape[1], px + 3)
        local_stamp = image_2d[y0:y1, x0:x1] - med

        sub_psf = False
        if local_stamp.size > 0:
            center_val = image_2d[py, px] - med
            surroundings = np.delete(local_stamp.flatten(), local_stamp.size // 2)
            surrounding_mean = float(np.mean(surroundings)) if surroundings.size > 0 else 0.0
            if center_val > 0 and (surrounding_mean / max(center_val, 1e-3)) < 0.15:
                sub_psf = True

        is_significant = (image_2d[py, px] - med) >= (self.config.min_cosmic_ray_snr * noise_model)
        cr_detected = (
            (max_contrast >= self.config.laplacian_cr_threshold) and sub_psf and is_significant
        )
        return max_contrast, cr_detected

    def _detect_optical_ghost(self, image_2d: np.ndarray) -> tuple[float, bool]:
        """Detect broad diffuse reflection halos or circular ghost rings."""
        h, w = image_2d.shape
        cy, cx = h / 2.0, w / 2.0
        y_indices, x_indices = np.indices((h, w))
        r = np.hypot(x_indices - cx, y_indices - cy)

        # Radial profile bins
        inner_mask = r < (h * 0.2)
        mid_mask = (r >= (h * 0.2)) & (r < (h * 0.45))

        med_inner = float(np.median(image_2d[inner_mask])) if np.any(inner_mask) else 0.0
        med_mid = float(np.median(image_2d[mid_mask])) if np.any(mid_mask) else 0.0
        mad_mid = (
            float(np.median(np.abs(image_2d[mid_mask] - med_mid))) if np.any(mid_mask) else 1e-3
        )

        ratio = (med_mid - med_inner) / max(mad_mid, 1e-3)
        ghost_detected = ratio >= self.config.ghost_ring_ratio_threshold
        return max(0.0, ratio), ghost_detected

    def _detect_dipole(self, diff_2d: np.ndarray) -> tuple[float, bool]:
        """Detect paired positive and negative lobes characteristic of subtraction dipoles."""
        med = float(np.median(diff_2d))
        mad = float(np.median(np.abs(diff_2d - med)))
        noise = max(1.4826 * mad, 1e-6)

        pos_max = float(np.max(diff_2d)) - med
        neg_min = med - float(np.min(diff_2d))

        if pos_max < 3.5 * noise or neg_min < 2.5 * noise:
            return 0.0, False

        py_pos, px_pos = np.unravel_index(np.argmax(diff_2d), diff_2d.shape)
        py_neg, px_neg = np.unravel_index(np.argmin(diff_2d), diff_2d.shape)

        dist = math.hypot(px_pos - px_neg, py_pos - py_neg)
        max_dist = 4.0 * self.config.psf_fwhm_px
        if 1.0 <= dist <= max_dist:
            ratio = min(pos_max, neg_min) / max(pos_max, neg_min, 1e-4)
            if ratio >= self.config.dipole_threshold:
                return float(ratio), True

        return 0.0, False

    def _detect_bleeding_column(self, diff_2d: np.ndarray) -> tuple[float, bool]:
        """Detect saturated CCD bleed columns spanning vertically across the cutout."""
        med = float(np.median(diff_2d))
        mad = float(np.median(np.abs(diff_2d - med)))
        noise = max(1.4826 * mad, 1e-6)

        col_medians = np.median(diff_2d, axis=0) - med
        max_col_excess = float(np.max(col_medians))
        if (max_col_excess / noise) >= self.config.bleeding_column_threshold:
            peak_col = int(np.argmax(col_medians))
            col_data = diff_2d[:, peak_col] - med
            span_ratio = float(np.sum(col_data > 2.0 * noise)) / float(diff_2d.shape[0])
            if span_ratio >= 0.4:
                return float(max_col_excess / noise), True

        return 0.0, False

    def _detect_diffraction_spike(self, image_2d: np.ndarray) -> tuple[float, bool]:
        """Detect narrow linear diffraction spikes extending across cardinal axes."""
        h, w = image_2d.shape
        cy, cx = h // 2, w // 2
        med = float(np.median(image_2d))
        mad = float(np.median(np.abs(image_2d - med)))
        noise = max(1.4826 * mad, 1e-6)

        row_slice = np.abs(image_2d[cy, :] - med)
        col_slice = np.abs(image_2d[:, cx] - med)

        row_spike = np.sum(row_slice > 3.0 * noise) / w
        col_spike = np.sum(col_slice > 3.0 * noise) / h

        max_prominence = max(row_spike, col_spike) * 10.0
        if max_prominence >= self.config.diffraction_spike_threshold:
            return float(max_prominence), True

        return 0.0, False


class DIAArtifactMiner:
    """Active miner extracting optical subtraction artifacts (dipoles, bleeds, spikes) from frames."""

    def __init__(self, filter_engine: DeepStreaksArtifactFilter | None = None) -> None:
        self.filter = filter_engine if filter_engine is not None else DeepStreaksArtifactFilter()

    def mine_artifacts_from_difference_frame(
        self,
        difference_frame: np.ndarray,
        science_frame: np.ndarray | None = None,
        stamp_size: int = 48,
        detection_threshold_sigma: float = 4.0,
        max_artifacts: int = 50,
    ) -> list[dict[str, Any]]:
        """Scans a difference frame to crop and classify DIA false positives."""
        diff = np.asarray(difference_frame, dtype=np.float32)
        med = float(np.median(diff))
        mad = float(np.median(np.abs(diff - med)))
        noise = max(1.4826 * mad, 1e-6)

        abs_diff = np.abs(diff - med)
        threshold = detection_threshold_sigma * noise

        local_max = (abs_diff == ndimage.maximum_filter(abs_diff, size=stamp_size // 2)) & (
            abs_diff >= threshold
        )
        peak_y, peak_x = np.where(local_max)

        h, w = diff.shape
        half = stamp_size // 2
        mined_artifacts: list[dict[str, Any]] = []

        for y, x in zip(peak_y, peak_x, strict=False):
            if len(mined_artifacts) >= max_artifacts:
                break
            if y < half or y + half > h or x < half or x + half > w:
                continue
            if any(math.hypot(x - a["x"], y - a["y"]) < half for a in mined_artifacts):
                continue

            diff_stamp = diff[y - half : y + half, x - half : x + half]
            sci_stamp = (
                science_frame[y - half : y + half, x - half : x + half]
                if science_frame is not None
                else None
            )

            assessment = self.filter.evaluate_cutout(diff_stamp, sci_stamp)
            if assessment.is_artifact:
                mined_artifacts.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "stamp": diff_stamp,
                        "artifact_type": assessment.artifact_type,
                        "taxonomy": assessment.hard_negative_taxonomy,
                        "confidence": assessment.confidence,
                        "reasons": assessment.reasons,
                    }
                )

        return mined_artifacts

