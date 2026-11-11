"""Unit tests for Module 4 PyTorch shift-and-stack, determinism, and ZOGY subtraction."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch
from pydantic import BaseModel, ValidationError

from LUMIA.physics.gpu_processing import (
    GPUProcessingConfig,
    PsiPhiStackResult,
    ShiftAndStackResult,
    VRAMLimitExceededError,
    ZOGYConfig,
    ZOGYResult,
    compute_psi_phi_stack,
    configure_deterministic_torch,
    shift_and_stack,
    zogy_subtract,
)


def _make_gaussian_psf(size: int, sigma_px: float) -> np.ndarray:
    """Create a normalized 2D circular Gaussian PSF centered at (size // 2, size // 2)."""
    center: int = size // 2
    y_grid, x_grid = np.mgrid[0:size, 0:size]
    r2: np.ndarray = (x_grid - center) ** 2 + (y_grid - center) ** 2
    psf: np.ndarray = np.exp(-0.5 * r2 / (sigma_px**2))
    psf /= np.sum(psf)
    return psf


def _inject_psf_source(
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
    image += norm * np.exp(-0.5 * r2 / (sigma_px**2))


def test_deterministic_execution_flags_enforced() -> None:
    """Verify deterministic PyTorch execution flags are enabled and outputs are bitwise reproducible."""
    torch.use_deterministic_algorithms(False)
    assert not torch.are_deterministic_algorithms_enabled()

    configure_deterministic_torch(seed=1234, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None

    rng = np.random.default_rng(99)
    frames: np.ndarray = rng.normal(0.0, 1.0, size=(4, 32, 32)).astype(np.float32)
    timestamps: np.ndarray = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    velocities: np.ndarray = np.array([[1.25, -0.75], [-0.5, 1.5]], dtype=np.float32)

    torch.use_deterministic_algorithms(False)
    cfg = GPUProcessingConfig(device="cpu", deterministic=True, seed=1234)
    res_a: ShiftAndStackResult = shift_and_stack(frames, timestamps, velocities, config=cfg)
    assert torch.are_deterministic_algorithms_enabled()

    res_b: ShiftAndStackResult = shift_and_stack(frames, timestamps, velocities, config=cfg)
    assert torch.equal(res_a.stacked_sum, res_b.stacked_sum)
    assert torch.equal(res_a.stacked_mean, res_b.stacked_mean)


def test_shift_and_stack_along_velocity_vectors_and_psi_phi() -> None:
    """Verify shift-and-stack coherently co-adds moving sources and boosts Psi/Phi SNR."""
    n_frames: int = 5
    height: int = 48
    width: int = 48
    x_start: float = 12.0
    y_start: float = 15.0
    true_vx: float = 2.0
    true_vy: float = 1.5
    source_flux: float = 400.0
    sigma_psf: float = 1.3

    timestamps: np.ndarray = np.arange(n_frames, dtype=np.float32)
    frames: np.ndarray = np.zeros((n_frames, height, width), dtype=np.float32)
    for k, dt in enumerate(timestamps):
        frame_2d = np.zeros((height, width), dtype=np.float64)
        _inject_psf_source(
            frame_2d,
            x0=x_start + true_vx * float(dt),
            y0=y_start + true_vy * float(dt),
            flux=source_flux,
            sigma_px=sigma_psf,
        )
        frames[k] = frame_2d.astype(np.float32)

    velocities = np.array(
        [
            [true_vx, true_vy],
            [0.0, 0.0],
            [-2.0, 1.5],
            [1.0, -1.0],
        ],
        dtype=np.float32,
    )

    cfg = GPUProcessingConfig(device="cpu", deterministic=True, velocity_batch_size=2)
    stack_res: ShiftAndStackResult = shift_and_stack(
        frames=frames,
        timestamps=timestamps,
        velocities=velocities,
        config=cfg,
    )

    assert stack_res.stacked_sum.shape == (4, height, width)
    assert stack_res.stacked_mean.shape == (4, height, width)

    single_epoch_peak: float = float(frames[0, int(y_start), int(x_start)])
    true_vel_peak: float = float(stack_res.stacked_sum[0, int(y_start), int(x_start)])
    stationary_vel_peak: float = float(stack_res.stacked_sum[1, int(y_start), int(x_start)])

    # Co-added peak along true velocity matches ~5x single-frame peak; wrong velocities do not
    assert true_vel_peak == pytest.approx(n_frames * single_epoch_peak, rel=3e-2)
    assert true_vel_peak > 4.0 * stationary_vel_peak

    # Verify Psi/Phi likelihood stacking boosts SNR by sqrt(n_frames)
    bg_sigma: float = 5.0
    inv_var: float = 1.0 / (bg_sigma**2)
    psi_frames: np.ndarray = frames * inv_var
    phi_frames: np.ndarray = np.full_like(frames, inv_var)

    single_frame_snr: float = single_epoch_peak / bg_sigma
    psi_phi_res: PsiPhiStackResult = compute_psi_phi_stack(
        psi_frames=psi_frames,
        phi_frames=phi_frames,
        timestamps=timestamps,
        velocities=velocities[:2],
        config=cfg,
    )
    stacked_snr_peak: float = float(psi_phi_res.snr_stack[0, int(y_start), int(x_start)])
    assert stacked_snr_peak == pytest.approx(
        math.sqrt(n_frames) * single_frame_snr,
        rel=3e-2,
    )


def test_zogy_optimal_subtraction_and_noise_scaling() -> None:
    """Verify ZOGY cancels static stars across differing PSFs and matches theoretical noise."""
    size: int = 64
    psf_size: int = 21
    sigma_psf_ref: float = 1.2
    sigma_psf_sci: float = 2.0
    f_ref: float = 1.0
    f_sci: float = 2.0
    noise_ref: float = 3.0
    noise_sci: float = 4.0

    psf_ref: np.ndarray = _make_gaussian_psf(psf_size, sigma_psf_ref)
    psf_sci: np.ndarray = _make_gaussian_psf(psf_size, sigma_psf_sci)

    ref_img = np.zeros((size, size), dtype=np.float64)
    sci_img = np.zeros((size, size), dtype=np.float64)

    # Inject an identical static bright star at (28, 28) scaled by each frame's flux factor
    static_flux: float = 50000.0
    _inject_psf_source(ref_img, x0=28.0, y0=28.0, flux=static_flux * f_ref, sigma_px=sigma_psf_ref)
    _inject_psf_source(sci_img, x0=28.0, y0=28.0, flux=static_flux * f_sci, sigma_px=sigma_psf_sci)

    # Inject a moving/transient asteroid ONLY in the science frame at (x=46, y=20)
    transient_flux: float = 2500.0
    _inject_psf_source(
        sci_img,
        x0=46.0,
        y0=20.0,
        flux=transient_flux * f_sci,
        sigma_px=sigma_psf_sci,
    )

    zogy_cfg = ZOGYConfig(
        flux_scale_science=f_sci,
        flux_scale_reference=f_ref,
        noise_std_science=noise_sci,
        noise_std_reference=noise_ref,
    )
    gpu_cfg = GPUProcessingConfig(device="cpu", deterministic=True)

    result: ZOGYResult = zogy_subtract(
        science_image=sci_img,
        reference_image=ref_img,
        science_psf=psf_sci,
        reference_psf=psf_ref,
        zogy_config=zogy_cfg,
        gpu_config=gpu_cfg,
    )

    assert isinstance(zogy_cfg, BaseModel)
    assert result.difference_image.shape == (size, size)
    assert result.significance_map.shape == (size, size)
    assert result.difference_psf.shape == (size, size)

    # Theoretical calibrated noise std: sqrt((noise_sci / f_sci)^2 + (noise_ref / f_ref)^2)
    expected_noise_std: float = math.hypot(noise_sci / f_sci, noise_ref / f_ref)
    assert result.theoretical_noise_std == pytest.approx(expected_noise_std, rel=1e-6)

    # Static star at (28, 28) cancels cleanly despite different seeing PSFs and flux scales
    static_residual: float = float(torch.abs(result.difference_image[24:33, 24:33]).max())
    transient_peak_sig: float = float(result.significance_map[20, 46])
    peak_flat_idx: int = int(torch.argmax(result.significance_map).item())
    peak_y, peak_x = divmod(peak_flat_idx, size)

    assert static_residual < 5.0
    assert (peak_x, peak_y) == (46, 20)
    assert transient_peak_sig > 25.0

    # Verify empirical unit-variance scaling of significance_map on pure Gaussian noise
    rng = np.random.default_rng(2026)
    noisy_ref: np.ndarray = rng.normal(0.0, noise_ref, size=(128, 128))
    noisy_sci: np.ndarray = rng.normal(0.0, noise_sci, size=(128, 128))
    noise_res: ZOGYResult = zogy_subtract(
        science_image=noisy_sci,
        reference_image=noisy_ref,
        science_psf=psf_sci,
        reference_psf=psf_ref,
        zogy_config=zogy_cfg,
        gpu_config=gpu_cfg,
    )
    assert float(noise_res.difference_image.std()) == pytest.approx(
        expected_noise_std,
        rel=5e-2,
    )
    assert float(noise_res.significance_map.std()) == pytest.approx(1.0, rel=5e-2)


def test_shape_mismatches_and_vram_resource_limits() -> None:
    """Verify shape mismatch errors, CUDA availability checks, and VRAM memory budget limits."""
    frames = np.zeros((4, 32, 32), dtype=np.float32)
    bad_timestamps = np.zeros((3,), dtype=np.float32)
    velocities = np.zeros((5, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="timestamps"):
        shift_and_stack(frames, bad_timestamps, velocities)

    with pytest.raises(ValueError, match="velocities"):
        shift_and_stack(frames, np.zeros((4,), dtype=np.float32), np.zeros((5, 3), dtype=np.float32))

    # Mismatched science and reference shapes in ZOGY
    zogy_cfg = ZOGYConfig(
        flux_scale_science=1.0,
        flux_scale_reference=1.0,
        noise_std_science=1.0,
        noise_std_reference=1.0,
    )
    psf = _make_gaussian_psf(11, 1.5)
    with pytest.raises(ValueError, match="shape"):
        zogy_subtract(
            np.zeros((32, 32)),
            np.zeros((32, 40)),
            psf,
            psf,
            zogy_cfg,
        )

    # Tight VRAM limit that cannot fit even a single velocity slice -> VRAMLimitExceededError
    tiny_vram_cfg = GPUProcessingConfig(device="cpu", max_vram_bytes=64)
    with pytest.raises(VRAMLimitExceededError):
        shift_and_stack(
            frames,
            np.arange(4, dtype=np.float32),
            velocities,
            config=tiny_vram_cfg,
        )

    if not torch.cuda.is_available():
        cuda_cfg = GPUProcessingConfig(device="cuda")
        with pytest.raises(RuntimeError, match="CUDA"):
            shift_and_stack(
                frames,
                np.arange(4, dtype=np.float32),
                velocities,
                config=cuda_cfg,
            )


# ---------------------------------------------------------------------------
# P0-3: strict Pydantic result models (replacing @dataclass outputs)
# ---------------------------------------------------------------------------


def _zeros(*shape: int) -> torch.Tensor:
    return torch.zeros(*shape, dtype=torch.float32)


def _stack_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "stacked_sum": _zeros(3, 8, 8),
        "stacked_mean": _zeros(3, 8, 8),
        "velocities": _zeros(3, 2),
    }
    base.update(overrides)
    return base


def _psi_phi_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "psi_stack": _zeros(3, 8, 8),
        "phi_stack": _zeros(3, 8, 8),
        "flux_stack": _zeros(3, 8, 8),
        "snr_stack": _zeros(3, 8, 8),
        "velocities": _zeros(3, 2),
    }
    base.update(overrides)
    return base


def _zogy_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "difference_image": _zeros(16, 16),
        "significance_map": _zeros(16, 16),
        "difference_psf": _zeros(16, 16),
        "theoretical_noise_std": 1.5,
        "flux_scale_diff": 0.8,
    }
    base.update(overrides)
    return base


def _with_value(shape: tuple[int, ...], value: float) -> torch.Tensor:
    tensor = torch.zeros(*shape, dtype=torch.float32)
    tensor.view(-1)[0] = value
    return tensor


def test_module4_results_are_pydantic_models_returned_by_pipeline() -> None:
    frames = np.zeros((3, 16, 16), dtype=np.float32)
    times = np.arange(3, dtype=np.float32)
    vels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cfg = GPUProcessingConfig(device="cpu")

    stack = shift_and_stack(frames, times, vels, config=cfg)
    psi_phi = compute_psi_phi_stack(frames + 1.0, frames + 1.0, times, vels, config=cfg)

    assert isinstance(stack, BaseModel)
    assert isinstance(psi_phi, BaseModel)
    assert stack.model_config["frozen"] is True
    with pytest.raises(ValidationError):
        stack.stacked_sum = torch.zeros(1)  # type: ignore[misc]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_shift_and_stack_result_rejects_non_finite_tensors(bad: float) -> None:
    for field in ("stacked_sum", "stacked_mean", "velocities"):
        shape = (3, 2) if field == "velocities" else (3, 8, 8)
        with pytest.raises(ValidationError):
            ShiftAndStackResult(**_stack_kwargs(**{field: _with_value(shape, bad)}))  # type: ignore[arg-type]


def test_shift_and_stack_result_rejects_bad_shapes_types_and_extras() -> None:
    ShiftAndStackResult(**_stack_kwargs())  # type: ignore[arg-type]  # baseline is valid

    with pytest.raises(ValidationError):
        ShiftAndStackResult(**_stack_kwargs(stacked_sum=_zeros(8, 8)))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ShiftAndStackResult(**_stack_kwargs(stacked_mean=_zeros(3, 8, 9)))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ShiftAndStackResult(**_stack_kwargs(velocities=_zeros(3, 3)))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ShiftAndStackResult(**_stack_kwargs(velocities=_zeros(4, 2)))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ShiftAndStackResult(**_stack_kwargs(stacked_sum=np.zeros((3, 8, 8))))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ShiftAndStackResult(**_stack_kwargs(stacked_sum=torch.zeros(3, 8, 8, dtype=torch.int64)))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ShiftAndStackResult(**_stack_kwargs(extra_field=1))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_psi_phi_result_rejects_non_finite_tensors(bad: float) -> None:
    for field in ("psi_stack", "phi_stack", "flux_stack", "snr_stack"):
        with pytest.raises(ValidationError):
            PsiPhiStackResult(**_psi_phi_kwargs(**{field: _with_value((3, 8, 8), bad)}))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        PsiPhiStackResult(**_psi_phi_kwargs(velocities=_with_value((3, 2), bad)))  # type: ignore[arg-type]


def test_psi_phi_result_rejects_inconsistent_shapes_and_extras() -> None:
    PsiPhiStackResult(**_psi_phi_kwargs())  # type: ignore[arg-type]  # baseline is valid

    with pytest.raises(ValidationError):
        PsiPhiStackResult(**_psi_phi_kwargs(snr_stack=_zeros(3, 8, 7)))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        PsiPhiStackResult(**_psi_phi_kwargs(psi_stack=_zeros(8, 8)))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        PsiPhiStackResult(**_psi_phi_kwargs(velocities=_zeros(2, 2)))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        PsiPhiStackResult(**_psi_phi_kwargs(unexpected=True))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_zogy_result_rejects_non_finite_values(bad: float) -> None:
    for field in ("difference_image", "significance_map", "difference_psf"):
        with pytest.raises(ValidationError):
            ZOGYResult(**_zogy_kwargs(**{field: _with_value((16, 16), bad)}))  # type: ignore[arg-type]
    for field in ("theoretical_noise_std", "flux_scale_diff"):
        with pytest.raises(ValidationError):
            ZOGYResult(**_zogy_kwargs(**{field: bad}))  # type: ignore[arg-type]


def test_zogy_result_rejects_bad_shapes_non_positive_scales_and_extras() -> None:
    ZOGYResult(**_zogy_kwargs())  # type: ignore[arg-type]  # baseline is valid

    with pytest.raises(ValidationError):
        ZOGYResult(**_zogy_kwargs(difference_image=_zeros(3, 16, 16)))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ZOGYResult(**_zogy_kwargs(significance_map=_zeros(16, 15)))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ZOGYResult(**_zogy_kwargs(theoretical_noise_std=0.0))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ZOGYResult(**_zogy_kwargs(flux_scale_diff=-1.0))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ZOGYResult(**_zogy_kwargs(note="x"))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_zogy_config_rejects_non_finite_scales(bad: float) -> None:
    good: dict[str, float] = {
        "flux_scale_science": 1.0,
        "flux_scale_reference": 1.0,
        "noise_std_science": 1.0,
        "noise_std_reference": 1.0,
    }
    for field in good:
        with pytest.raises(ValidationError):
            ZOGYConfig(**{**good, field: bad})
    with pytest.raises(ValidationError):
        ZOGYConfig(**good, bogus=1.0)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        GPUProcessingConfig(device="cpu", bogus=1)  # type: ignore[call-arg]
