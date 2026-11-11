"""Tests for synthetic moving-object (asteroid) injection engine."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch

from LUMIA.physics.synthetic_injection import (
    GAUSSIAN_FWHM_TO_SIGMA,
    SyntheticInjectionConfig,
    SyntheticInjector,
    inject_synthetic_asteroid,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# 1. Flux Conservation & Calibration Tests
# ---------------------------------------------------------------------------


class TestFluxConservation:
    """Verify that total injected instrumental ADU matches astrophysical equations."""

    def test_flux_conservation_noiseless(self) -> None:
        """Verify sum of pixels strictly equals F_ADU = 10^(-0.4*(m-ZP)) * t_exp."""
        h, w = 256, 256
        science = torch.zeros((h, w), dtype=torch.float32)

        x0, y0 = 128.0, 128.0
        magnitude = 18.5
        zero_point = 25.0
        exposure_time = 45.0
        fwhm_pixels = 3.2

        expected_flux_adu = (10.0 ** (-0.4 * (magnitude - zero_point))) * exposure_time

        injected = inject_synthetic_asteroid(
            science_tensor=science,
            x0=x0,
            y0=y0,
            magnitude=magnitude,
            zero_point=zero_point,
            fwhm_pixels=fwhm_pixels,
            dx=0.0,
            dy=0.0,
            exposure_time=exposure_time,
            include_poisson_noise=False,
            in_place=False,
        )

        measured_flux = float(injected.sum().item())
        relative_error = abs(measured_flux - expected_flux_adu) / expected_flux_adu

        assert relative_error < 0.001, (
            f"Measured flux {measured_flux:.2f} deviates from expected {expected_flux_adu:.2f} "
            f"by {relative_error:.4%}"
        )

    def test_flux_conservation_with_poisson_noise(self) -> None:
        """Verify Poisson noisy flux is statistically consistent within 3-sigma."""
        h, w = 256, 256
        science = torch.zeros((h, w), dtype=torch.float32)

        x0, y0 = 128.0, 128.0
        magnitude = 17.0
        zero_point = 25.0
        exposure_time = 30.0
        fwhm_pixels = 3.0
        gain = 1.8

        expected_flux_adu = (10.0 ** (-0.4 * (magnitude - zero_point))) * exposure_time
        expected_electrons = expected_flux_adu * gain

        # Standard deviation of the summed Poisson flux in ADU is sqrt(N_e) / gain = sqrt(F_ADU / gain)
        sigma_flux_adu = math.sqrt(expected_electrons) / gain

        gen = torch.Generator().manual_seed(42)
        injected = inject_synthetic_asteroid(
            science_tensor=science,
            x0=x0,
            y0=y0,
            magnitude=magnitude,
            zero_point=zero_point,
            fwhm_pixels=fwhm_pixels,
            dx=0.0,
            dy=0.0,
            exposure_time=exposure_time,
            gain=gain,
            include_poisson_noise=True,
            generator=gen,
        )

        measured_flux = float(injected.sum().item())
        difference = abs(measured_flux - expected_flux_adu)

        assert difference <= 3.5 * sigma_flux_adu, (
            f"Measured noisy flux {measured_flux:.2f} deviates from {expected_flux_adu:.2f} "
            f"by {difference:.2f} ADU (allowed 3.5-sigma = {3.5 * sigma_flux_adu:.2f})"
        )

    def test_trailed_flux_conservation(self) -> None:
        """Verify that motion trailing (dx, dy > 0) preserves total integrated flux."""
        h, w = 512, 512
        science = torch.zeros((h, w), dtype=torch.float32)

        x0, y0 = 256.0, 256.0
        magnitude = 19.0
        zero_point = 25.0
        exposure_time = 60.0
        fwhm_pixels = 3.5
        dx, dy = 25.0, -18.0

        expected_flux_adu = (10.0 ** (-0.4 * (magnitude - zero_point))) * exposure_time

        injected = inject_synthetic_asteroid(
            science_tensor=science,
            x0=x0,
            y0=y0,
            magnitude=magnitude,
            zero_point=zero_point,
            fwhm_pixels=fwhm_pixels,
            dx=dx,
            dy=dy,
            exposure_time=exposure_time,
            include_poisson_noise=False,
        )

        measured_flux = float(injected.sum().item())
        relative_error = abs(measured_flux - expected_flux_adu) / expected_flux_adu

        assert relative_error < 0.001


# ---------------------------------------------------------------------------
# 2. Motion Trailing & Morphology Tests
# ---------------------------------------------------------------------------


class TestMotionTrailingMorphology:
    """Verify that intra-exposure motion smearing stretches emission along trajectory."""

    @staticmethod
    def _compute_second_moments(image: torch.Tensor) -> tuple[float, float, float]:
        """Compute intensity-weighted second moments M_xx, M_yy, M_xy."""
        total = image.sum().item()
        assert total > 0, "Image sum must be strictly positive."

        h, w = image.shape
        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )

        x_mean = (image * x_grid).sum().item() / total
        y_mean = (image * y_grid).sum().item() / total

        m_xx = (image * (x_grid - x_mean) ** 2).sum().item() / total
        m_yy = (image * (y_grid - y_mean) ** 2).sum().item() / total
        m_xy = (image * (x_grid - x_mean) * (y_grid - y_mean)).sum().item() / total

        return m_xx, m_yy, m_xy

    def test_trailing_elongates_moments_along_motion_axis(self) -> None:
        """Verify M_xx >> M_yy when dx is large and dy = 0."""
        h, w = 256, 256
        x0, y0 = 128.0, 128.0
        fwhm = 3.0

        # Stationary source (dx = 0, dy = 0)
        science_stat = torch.zeros((h, w), dtype=torch.float32)
        stat_img = inject_synthetic_asteroid(
            science_stat,
            x0=x0,
            y0=y0,
            magnitude=16.0,
            zero_point=25.0,
            fwhm_pixels=fwhm,
            dx=0.0,
            dy=0.0,
            include_poisson_noise=False,
        )
        stat_mxx, stat_myy, _ = self._compute_second_moments(stat_img)

        # For circular stationary Gaussian, M_xx should approximately equal M_yy
        expected_sigma2 = (fwhm * GAUSSIAN_FWHM_TO_SIGMA) ** 2
        assert abs(stat_mxx - expected_sigma2) / expected_sigma2 < 0.05
        assert abs(stat_myy - expected_sigma2) / expected_sigma2 < 0.05

        # Trailed source along column direction (dx = 20.0, dy = 0.0)
        science_trailed = torch.zeros((h, w), dtype=torch.float32)
        trail_img = inject_synthetic_asteroid(
            science_trailed,
            x0=x0,
            y0=y0,
            magnitude=16.0,
            zero_point=25.0,
            fwhm_pixels=fwhm,
            dx=20.0,
            dy=0.0,
            include_poisson_noise=False,
        )
        trail_mxx, trail_myy, _ = self._compute_second_moments(trail_img)

        # Column dispersion M_xx must be vastly larger than row dispersion M_yy
        assert trail_mxx > 10.0 * trail_myy
        assert trail_mxx > 10.0 * stat_mxx
        # Row dispersion M_yy should remain essentially unchanged
        assert abs(trail_myy - stat_myy) / stat_myy < 0.05

    def test_diagonal_trailing_correlation(self) -> None:
        """Verify diagonal motion (dx = 15, dy = 15) produces positive covariance M_xy."""
        h, w = 256, 256
        science = torch.zeros((h, w), dtype=torch.float32)

        diag_img = inject_synthetic_asteroid(
            science,
            x0=128.0,
            y0=128.0,
            magnitude=16.0,
            zero_point=25.0,
            fwhm_pixels=3.0,
            dx=15.0,
            dy=15.0,
            include_poisson_noise=False,
        )
        m_xx, m_yy, m_xy = self._compute_second_moments(diag_img)

        assert m_xx > 5.0
        assert m_yy > 5.0
        assert m_xy > 0.0, "Diagonal motion along (+x, +y) must exhibit positive spatial covariance."
        # Ratio of m_xy to (m_xx + m_yy)/2 should indicate strong diagonal elongation
        correlation = m_xy / math.sqrt(m_xx * m_yy)
        assert correlation > 0.85


# ---------------------------------------------------------------------------
# 3. Saturation, Boundaries, and In-Place Behavior Tests
# ---------------------------------------------------------------------------


class TestSaturationAndBoundaries:
    """Verify detector saturation clipping, boundary clamping, and immutability."""

    def test_saturation_clipping_ceiling(self) -> None:
        """Verify bright source flux clips cleanly at saturation_limit_adu."""
        h, w = 128, 128
        science = torch.zeros((h, w), dtype=torch.float32)

        sat_limit = 1000.0
        injected = inject_synthetic_asteroid(
            science,
            x0=64.0,
            y0=64.0,
            magnitude=10.0,  # Extremely bright source
            zero_point=25.0,
            fwhm_pixels=2.5,
            saturation_limit_adu=sat_limit,
            include_poisson_noise=False,
        )

        assert injected.max().item() <= sat_limit
        # Peak core pixels should be saturated exactly at ceiling
        assert injected.max().item() == pytest.approx(sat_limit)

    def test_in_place_flag_behavior(self) -> None:
        """Verify in_place=False preserves input; in_place=True modifies input."""
        h, w = 64, 64
        original = torch.zeros((h, w), dtype=torch.float32)

        # Default in_place=False
        out1 = inject_synthetic_asteroid(
            original,
            x0=32.0,
            y0=32.0,
            magnitude=18.0,
            zero_point=25.0,
            fwhm_pixels=3.0,
            in_place=False,
        )
        assert original.sum().item() == 0.0
        assert out1.sum().item() > 0.0

        # in_place=True
        out2 = inject_synthetic_asteroid(
            original,
            x0=32.0,
            y0=32.0,
            magnitude=18.0,
            zero_point=25.0,
            fwhm_pixels=3.0,
            in_place=True,
        )
        assert out2 is original
        assert original.sum().item() > 0.0

    def test_source_near_edge_does_not_crash(self) -> None:
        """Verify injection centered near sensor boundary (x0=1, y0=1) handles clipping."""
        h, w = 128, 128
        science = torch.zeros((h, w), dtype=torch.float32)

        injected = inject_synthetic_asteroid(
            science,
            x0=1.0,
            y0=1.0,
            magnitude=18.0,
            zero_point=25.0,
            fwhm_pixels=3.0,
            dx=5.0,
            dy=-5.0,
        )

        assert injected.shape == (h, w)
        assert injected.sum().item() > 0.0

    def test_source_completely_off_detector(self) -> None:
        """Verify source entirely outside pixel coordinates returns unaltered tensor."""
        h, w = 128, 128
        science = torch.zeros((h, w), dtype=torch.float32)

        injected = inject_synthetic_asteroid(
            science,
            x0=-500.0,
            y0=-500.0,
            magnitude=18.0,
            zero_point=25.0,
            fwhm_pixels=3.0,
        )

        assert injected.sum().item() == 0.0


# ---------------------------------------------------------------------------
# 4. Device Agnosticism & SyntheticInjector Service Tests
# ---------------------------------------------------------------------------


class TestDeviceAgnosticismAndClassAPI:
    """Verify CPU/GPU execution compatibility and SyntheticInjector class interface."""

    def test_device_agnosticism(self) -> None:
        """Verify execution succeeds and preserves tensor device and float precision."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        science = torch.zeros((128, 128), dtype=torch.float32, device=device)
        injected = inject_synthetic_asteroid(
            science,
            x0=64.0,
            y0=64.0,
            magnitude=19.0,
            zero_point=25.0,
            fwhm_pixels=2.8,
            dx=4.0,
            dy=2.0,
        )

        assert injected.device == science.device
        assert injected.dtype == torch.float32
        assert injected.shape == (128, 128)

    def test_synthetic_injector_service(self) -> None:
        """Verify SyntheticInjector service properties and injection execution."""
        injector = SyntheticInjector(
            zero_point=25.0,
            gain=1.5,
            fwhm_pixels=3.2,
            saturation_limit_adu=60000.0,
        )

        assert injector.zero_point == 25.0
        assert injector.gain == 1.5
        assert injector.fwhm_pixels == 3.2
        assert injector.sigma_pixels == pytest.approx(3.2 * GAUSSIAN_FWHM_TO_SIGMA)
        assert injector.saturation_limit_adu == 60000.0

        # Compute flux
        expected_flux = (10.0 ** (-0.4 * (18.0 - 25.0))) * 30.0
        assert injector.compute_flux_adu(magnitude=18.0, exposure_time=30.0) == pytest.approx(
            expected_flux
        )

        science = torch.zeros((128, 128), dtype=torch.float32)
        injected = injector.inject(
            science_tensor=science,
            x0=64.0,
            y0=64.0,
            magnitude=18.0,
            exposure_time=30.0,
            dx=5.0,
            dy=-3.0,
            include_poisson_noise=False,
        )

        assert injected.sum().item() == pytest.approx(expected_flux, rel=1e-3)

    def test_invalid_input_types_raise_proper_exceptions(self) -> None:
        """Verify invalid science_tensor types and shapes raise descriptive domain errors."""
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            inject_synthetic_asteroid(
                science_tensor=[[1.0, 2.0]],  # type: ignore[arg-type]
                x0=10.0,
                y0=10.0,
                magnitude=18.0,
                zero_point=25.0,
                fwhm_pixels=3.0,
            )

        with pytest.raises(ValueError, match="must be 2-dimensional"):
            inject_synthetic_asteroid(
                science_tensor=torch.zeros((10, 10, 10)),
                x0=5.0,
                y0=5.0,
                magnitude=18.0,
                zero_point=25.0,
                fwhm_pixels=3.0,
            )

        with pytest.raises(ValueError, match="fwhm_pixels must be positive"):
            inject_synthetic_asteroid(
                science_tensor=torch.zeros((32, 32)),
                x0=16.0,
                y0=16.0,
                magnitude=18.0,
                zero_point=25.0,
                fwhm_pixels=-1.0,
            )

    def test_synthetic_injection_config_validation(self) -> None:
        """Verify SyntheticInjectionConfig Pydantic model validation."""
        cfg = SyntheticInjectionConfig(
            magnitude=19.2,
            zero_point=25.1,
            fwhm_pixels=3.0,
            exposure_time=45.0,
            gain=1.2,
            dx=10.0,
            dy=5.0,
        )
        assert cfg.magnitude == 19.2
        assert cfg.exposure_time == 45.0
        assert cfg.gain == 1.2
        assert cfg.include_poisson_noise is True
