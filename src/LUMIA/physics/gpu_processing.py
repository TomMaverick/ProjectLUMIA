"""GPU-accelerated PyTorch shift-and-stack, Psi/Phi stacking, and ZOGY image subtraction."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import ConfigDict, Field, model_validator

from LUMIA.core.runtime import (
    DEVICE_SPEC_PATTERN,
    configure_deterministic_torch,
    resolve_torch_device,
)
from LUMIA.core.schemas import LumiaBaseModel

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from LUMIA.core.schemas import CutoutBatchResult, SNRPeakCandidate
    from LUMIA.physics.cutout_extraction import CutoutConfig, PeakExtractionConfig

_EPSILON: float = 1e-12


class VRAMLimitExceededError(MemoryError):
    """Raised when tensor working memory exceeds configured or hardware VRAM limits."""


class GPUProcessingConfig(LumiaBaseModel):
    """Validated compute, determinism, and VRAM chunking configuration."""

    device: str = Field(
        default="auto",
        pattern=DEVICE_SPEC_PATTERN,
        description="Target PyTorch device: 'cpu', 'cuda', 'cuda:N' or 'auto'.",
    )
    deterministic: bool = Field(
        default=True,
        description="Enforce torch.use_deterministic_algorithms(True).",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Deterministic random seed for reproducible execution.",
    )
    velocity_batch_size: int = Field(
        default=32,
        ge=1,
        description="Maximum number of velocity vectors evaluated per VRAM chunk.",
    )
    max_vram_bytes: int = Field(
        default=2 * 1024**3,
        gt=0,
        description="Maximum working tensor memory budget in bytes per batch.",
    )
    num_cpu_threads: int | None = Field(
        default=None,
        ge=1,
        description="Optional limit on intra-op CPU threads.",
    )


class ZOGYConfig(LumiaBaseModel):
    """Validated photometric flux scales and background RMS noise parameters for ZOGY."""

    flux_scale_science: float = Field(
        ...,
        gt=0.0,
        description="Flux-based zero-point scaling factor F_n of the science frame.",
    )
    flux_scale_reference: float = Field(
        ...,
        gt=0.0,
        description="Flux-based zero-point scaling factor F_r of the reference frame.",
    )
    noise_std_science: float = Field(
        ...,
        gt=0.0,
        description="Background pixel RMS noise sigma_n of the science frame.",
    )
    noise_std_reference: float = Field(
        ...,
        gt=0.0,
        description="Background pixel RMS noise sigma_r of the reference frame.",
    )


def _require_finite_float_tensor(name: str, tensor: torch.Tensor, ndim: int) -> None:
    """Validate that a tensor is floating point, has the given rank, and is finite."""
    if not tensor.is_floating_point():
        msg = f"{name} must be a floating-point tensor, got dtype {tensor.dtype}."
        raise ValueError(msg)
    if tensor.ndim != ndim:
        msg = f"{name} must be a {ndim}D tensor, got shape {tuple(tensor.shape)}."
        raise ValueError(msg)
    if not bool(torch.all(torch.isfinite(tensor)).item()):
        msg = f"{name} contains non-finite (NaN or Inf) values."
        raise ValueError(msg)


def _require_velocity_table(name: str, velocities: torch.Tensor, n_velocities: int) -> None:
    """Validate a (V, 2) trial-velocity table matching the stack's velocity count."""
    _require_finite_float_tensor(name, velocities, 2)
    if velocities.shape[1] != 2 or velocities.shape[0] != n_velocities:
        msg = (
            f"{name} must have shape ({n_velocities}, 2), "
            f"got {tuple(velocities.shape)}."
        )
        raise ValueError(msg)


class ShiftAndStackResult(LumiaBaseModel):
    """Co-added sum and mean tensor stacks across trial velocity vectors."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
    )

    stacked_sum: torch.Tensor
    stacked_mean: torch.Tensor
    velocities: torch.Tensor

    @model_validator(mode="after")
    def _validate_tensors(self) -> ShiftAndStackResult:
        _require_finite_float_tensor("stacked_sum", self.stacked_sum, 3)
        _require_finite_float_tensor("stacked_mean", self.stacked_mean, 3)
        if self.stacked_mean.shape != self.stacked_sum.shape:
            msg = (
                f"stacked_mean shape {tuple(self.stacked_mean.shape)} must match "
                f"stacked_sum shape {tuple(self.stacked_sum.shape)}."
            )
            raise ValueError(msg)
        _require_velocity_table("velocities", self.velocities, self.stacked_sum.shape[0])
        return self


class PsiPhiStackResult(LumiaBaseModel):
    """Maximum-likelihood Psi, Phi, flux, and SNR tensor stacks across velocities."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
    )

    psi_stack: torch.Tensor
    phi_stack: torch.Tensor
    flux_stack: torch.Tensor
    snr_stack: torch.Tensor
    velocities: torch.Tensor

    @model_validator(mode="after")
    def _validate_tensors(self) -> PsiPhiStackResult:
        reference_shape = tuple(self.psi_stack.shape)
        for name in ("psi_stack", "phi_stack", "flux_stack", "snr_stack"):
            tensor: torch.Tensor = getattr(self, name)
            _require_finite_float_tensor(name, tensor, 3)
            if tuple(tensor.shape) != reference_shape:
                msg = (
                    f"{name} shape {tuple(tensor.shape)} must match "
                    f"psi_stack shape {reference_shape}."
                )
                raise ValueError(msg)
        _require_velocity_table("velocities", self.velocities, reference_shape[0])
        return self


class ZOGYResult(LumiaBaseModel):
    """Proper ZOGY difference image, matched-filter significance map, and effective PSF."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
    )

    difference_image: torch.Tensor
    significance_map: torch.Tensor
    difference_psf: torch.Tensor
    theoretical_noise_std: float = Field(..., gt=0.0)
    flux_scale_diff: float = Field(..., gt=0.0)

    @model_validator(mode="after")
    def _validate_tensors(self) -> ZOGYResult:
        reference_shape = tuple(self.difference_image.shape)
        for name in ("difference_image", "significance_map", "difference_psf"):
            tensor: torch.Tensor = getattr(self, name)
            _require_finite_float_tensor(name, tensor, 2)
            if tuple(tensor.shape) != reference_shape:
                msg = (
                    f"{name} shape {tuple(tensor.shape)} must match "
                    f"difference_image shape {reference_shape}."
                )
                raise ValueError(msg)
        return self

    def extract_peaks(
        self,
        config: PeakExtractionConfig | None = None,
    ) -> list[SNRPeakCandidate]:
        """Deterministically extract local-maximum SNR peak candidates from significance_map."""
        from LUMIA.physics.cutout_extraction import extract_snr_peaks

        return extract_snr_peaks(self, config=config)

    def extract_cutouts(
        self,
        science_frame: torch.Tensor | ArrayLike,
        reference_frame: torch.Tensor | ArrayLike,
        *,
        peak_config: PeakExtractionConfig | None = None,
        cutout_config: CutoutConfig | None = None,
    ) -> CutoutBatchResult:
        """Extract SNR peaks and slice aligned 3-channel (Sci, Ref, Diff) cutouts in memory."""
        from LUMIA.physics.cutout_extraction import extract_zogy_cutout_batch

        return extract_zogy_cutout_batch(
            science_frame=science_frame,
            reference_frame=reference_frame,
            zogy_result=self,
            peak_config=peak_config,
            cutout_config=cutout_config,
        )


def _to_float_tensor(
    data: ArrayLike | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Convert an array-like or Tensor input to a finite tensor on the target device."""
    tensor: torch.Tensor = (
        data.to(device=device, dtype=dtype)
        if isinstance(data, torch.Tensor)
        else torch.as_tensor(np.asarray(data), dtype=dtype, device=device)
    )
    if not bool(torch.all(torch.isfinite(tensor)).item()):
        msg = f"Input tensor '{name}' contains non-finite (NaN or Inf) values."
        raise ValueError(msg)
    return tensor


def shift_and_stack(
    frames: ArrayLike | torch.Tensor,
    timestamps: ArrayLike | torch.Tensor,
    velocities: ArrayLike | torch.Tensor,
    config: GPUProcessingConfig | None = None,
) -> ShiftAndStackResult:
    """Shift and co-add sequential frames along candidate 2D velocity vectors."""
    cfg: GPUProcessingConfig = config if config is not None else GPUProcessingConfig()
    device: torch.device = resolve_torch_device(cfg.device)
    configure_deterministic_torch(
        seed=cfg.seed,
        deterministic=cfg.deterministic,
        num_cpu_threads=cfg.num_cpu_threads,
    )

    frames_cpu_check: tuple[int, ...] = (
        tuple(frames.shape) if isinstance(frames, torch.Tensor) else np.asarray(frames).shape
    )
    if len(frames_cpu_check) != 3:
        msg = f"frames must be a 3D tensor of shape (T, H, W), got {frames_cpu_check}."
        raise ValueError(msg)
    n_frames, height, width = frames_cpu_check
    if n_frames < 1 or height < 2 or width < 2:
        msg = (
            f"Invalid frame dimensions (T={n_frames}, H={height}, W={width}); "
            "need T>=1, H>=2, W>=2."
        )
        raise ValueError(msg)

    # Pre-check VRAM working set before allocating tensors on device
    elem_bytes: int = 4  # float32
    base_bytes: int = n_frames * height * width * elem_bytes
    bytes_per_velocity: int = n_frames * height * width * 3 * elem_bytes
    min_required_bytes: int = base_bytes + bytes_per_velocity
    if min_required_bytes > cfg.max_vram_bytes:
        msg = (
            f"Minimum working memory ({min_required_bytes} bytes) exceeds "
            f"configured max_vram_bytes ({cfg.max_vram_bytes} bytes)."
        )
        raise VRAMLimitExceededError(msg)

    try:
        with torch.no_grad():
            frames_t: torch.Tensor = _to_float_tensor(
                frames,
                device=device,
                dtype=torch.float32,
                name="frames",
            )
            times_t: torch.Tensor = _to_float_tensor(
                timestamps,
                device=device,
                dtype=torch.float32,
                name="timestamps",
            )
            vel_t: torch.Tensor = _to_float_tensor(
                velocities,
                device=device,
                dtype=torch.float32,
                name="velocities",
            )

            if times_t.ndim != 1 or times_t.shape[0] != n_frames:
                msg = (
                    f"timestamps must be a 1D tensor of length {n_frames}, "
                    f"got shape {tuple(times_t.shape)}."
                )
                raise ValueError(msg)
            if vel_t.ndim != 2 or vel_t.shape[1] != 2 or vel_t.shape[0] == 0:
                msg = (
                    "velocities must be a non-empty 2D tensor of shape (V, 2), "
                    f"got shape {tuple(vel_t.shape)}."
                )
                raise ValueError(msg)

            n_vel: int = int(vel_t.shape[0])
            dt: torch.Tensor = times_t - times_t[0]  # (T,)

            # Base pixel coordinate meshgrid (H, W)
            y_coords: torch.Tensor = torch.arange(height, device=device, dtype=torch.float32)
            x_coords: torch.Tensor = torch.arange(width, device=device, dtype=torch.float32)
            base_y, base_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

            inv_w_scale: float = 2.0 / float(width - 1)
            inv_h_scale: float = 2.0 / float(height - 1)

            available_chunk_bytes: int = max(
                bytes_per_velocity,
                cfg.max_vram_bytes - base_bytes,
            )
            chunk_size: int = min(
                cfg.velocity_batch_size,
                max(1, available_chunk_bytes // bytes_per_velocity),
            )

            stacked_sum_out: torch.Tensor = torch.empty(
                (n_vel, height, width),
                dtype=torch.float32,
                device=device,
            )

            for start_idx in range(0, n_vel, chunk_size):
                end_idx: int = min(start_idx + chunk_size, n_vel)
                batch_len: int = end_idx - start_idx
                vel_chunk: torch.Tensor = vel_t[start_idx:end_idx]  # (B, 2)

                # Shift offsets dx, dy of shape (B, T, 1, 1)
                dx: torch.Tensor = (
                    vel_chunk[:, 0:1] * dt.unsqueeze(0)
                ).view(batch_len, n_frames, 1, 1)
                dy: torch.Tensor = (
                    vel_chunk[:, 1:2] * dt.unsqueeze(0)
                ).view(batch_len, n_frames, 1, 1)

                sample_x_norm: torch.Tensor = (
                    (base_x.view(1, 1, height, width) + dx) * inv_w_scale - 1.0
                )
                sample_y_norm: torch.Tensor = (
                    (base_y.view(1, 1, height, width) + dy) * inv_h_scale - 1.0
                )
                grid: torch.Tensor = torch.stack(
                    (sample_x_norm, sample_y_norm),
                    dim=-1,
                ).view(batch_len * n_frames, height, width, 2)

                expanded_frames: torch.Tensor = (
                    frames_t.unsqueeze(0)
                    .expand(batch_len, n_frames, height, width)
                    .reshape(batch_len * n_frames, 1, height, width)
                )

                sampled: torch.Tensor = F.grid_sample(
                    expanded_frames,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                ).view(batch_len, n_frames, height, width)

                stacked_sum_out[start_idx:end_idx] = torch.sum(sampled, dim=1)

            stacked_mean_out: torch.Tensor = stacked_sum_out / float(n_frames)
            return ShiftAndStackResult(
                stacked_sum=stacked_sum_out,
                stacked_mean=stacked_mean_out,
                velocities=vel_t,
            )
    except torch.OutOfMemoryError as exc:
        msg = f"Hardware GPU out-of-memory encountered during shift_and_stack: {exc}"
        raise VRAMLimitExceededError(msg) from exc


def compute_psi_phi_stack(
    psi_frames: ArrayLike | torch.Tensor,
    phi_frames: ArrayLike | torch.Tensor,
    timestamps: ArrayLike | torch.Tensor,
    velocities: ArrayLike | torch.Tensor,
    config: GPUProcessingConfig | None = None,
) -> PsiPhiStackResult:
    """Compute likelihood-weighted Psi and Phi stacks and moving-object SNR maps."""
    psi_shape: tuple[int, ...] = (
        tuple(psi_frames.shape)
        if isinstance(psi_frames, torch.Tensor)
        else np.asarray(psi_frames).shape
    )
    phi_shape: tuple[int, ...] = (
        tuple(phi_frames.shape)
        if isinstance(phi_frames, torch.Tensor)
        else np.asarray(phi_frames).shape
    )
    if psi_shape != phi_shape:
        msg = f"psi_frames shape {psi_shape} must match phi_frames shape {phi_shape}."
        raise ValueError(msg)

    psi_res: ShiftAndStackResult = shift_and_stack(
        frames=psi_frames,
        timestamps=timestamps,
        velocities=velocities,
        config=config,
    )
    phi_res: ShiftAndStackResult = shift_and_stack(
        frames=phi_frames,
        timestamps=timestamps,
        velocities=velocities,
        config=config,
    )

    with torch.no_grad():
        safe_phi: torch.Tensor = torch.clamp(phi_res.stacked_sum, min=0.0)
        valid: torch.Tensor = safe_phi > _EPSILON
        denom_flux: torch.Tensor = torch.where(
            valid,
            safe_phi,
            torch.ones_like(safe_phi),
        )
        denom_snr: torch.Tensor = torch.sqrt(denom_flux)
        flux_stack: torch.Tensor = torch.where(
            valid,
            psi_res.stacked_sum / denom_flux,
            torch.zeros_like(psi_res.stacked_sum),
        )
        snr_stack: torch.Tensor = torch.where(
            valid,
            psi_res.stacked_sum / denom_snr,
            torch.zeros_like(psi_res.stacked_sum),
        )

    return PsiPhiStackResult(
        psi_stack=psi_res.stacked_sum,
        phi_stack=phi_res.stacked_sum,
        flux_stack=flux_stack,
        snr_stack=snr_stack,
        velocities=psi_res.velocities,
    )


def _embed_psf_zero_phase(
    psf: torch.Tensor,
    image_shape: tuple[int, int],
    *,
    name: str,
) -> torch.Tensor:
    """Normalize a 2D PSF kernel and shift its center to index (0, 0) for FFT."""
    if psf.ndim != 2:
        msg = f"{name} must be a 2D PSF kernel, got shape {tuple(psf.shape)}."
        raise ValueError(msg)
    kh, kw = psf.shape
    ih, iw = image_shape
    if kh < 1 or kw < 1 or kh > ih or kw > iw:
        msg = (
            f"{name} shape {(kh, kw)} must be non-empty and not exceed "
            f"image shape {(ih, iw)}."
        )
        raise ValueError(msg)

    psf_sum: float = float(torch.sum(psf).item())
    if not math.isfinite(psf_sum) or psf_sum <= 0.0:
        msg = f"{name} must have a finite strictly positive sum, got {psf_sum}."
        raise ValueError(msg)

    norm_psf: torch.Tensor = psf / psf_sum
    canvas: torch.Tensor = torch.zeros(image_shape, dtype=psf.dtype, device=psf.device)
    y0: int = (ih // 2) - (kh // 2)
    x0: int = (iw // 2) - (kw // 2)
    canvas[y0 : y0 + kh, x0 : x0 + kw] = norm_psf
    return torch.fft.ifftshift(canvas)


def zogy_subtract(
    science_image: ArrayLike | torch.Tensor,
    reference_image: ArrayLike | torch.Tensor,
    science_psf: ArrayLike | torch.Tensor,
    reference_psf: ArrayLike | torch.Tensor,
    zogy_config: ZOGYConfig,
    gpu_config: GPUProcessingConfig | None = None,
) -> ZOGYResult:
    """Compute the optimal ZOGY difference image, significance map, and effective PSF."""
    cfg: GPUProcessingConfig = (
        gpu_config if gpu_config is not None else GPUProcessingConfig()
    )
    device: torch.device = resolve_torch_device(cfg.device)
    configure_deterministic_torch(
        seed=cfg.seed,
        deterministic=cfg.deterministic,
        num_cpu_threads=cfg.num_cpu_threads,
    )

    sci_shape: tuple[int, ...] = (
        tuple(science_image.shape)
        if isinstance(science_image, torch.Tensor)
        else np.asarray(science_image).shape
    )
    ref_shape: tuple[int, ...] = (
        tuple(reference_image.shape)
        if isinstance(reference_image, torch.Tensor)
        else np.asarray(reference_image).shape
    )
    if len(sci_shape) != 2 or len(ref_shape) != 2 or sci_shape != ref_shape:
        msg = (
            f"science_image shape {sci_shape} and reference_image shape {ref_shape} "
            "must be matching 2D arrays."
        )
        raise ValueError(msg)

    ih, iw = sci_shape
    # Complex64/float64 FFT working memory estimate (~6 full-size 2D buffers)
    est_bytes: int = ih * iw * 8 * 8
    if est_bytes > cfg.max_vram_bytes:
        msg = (
            f"Estimated ZOGY FFT memory ({est_bytes} bytes) exceeds "
            f"max_vram_bytes ({cfg.max_vram_bytes} bytes)."
        )
        raise VRAMLimitExceededError(msg)

    f_n: float = zogy_config.flux_scale_science
    f_r: float = zogy_config.flux_scale_reference
    sig_n: float = zogy_config.noise_std_science
    sig_r: float = zogy_config.noise_std_reference

    # ZOGY Eq. 15: effective difference zero-point scale and calibrated noise RMS
    denom_zero_freq: float = math.hypot(sig_n * f_r, sig_r * f_n)
    f_d: float = (f_r * f_n) / denom_zero_freq
    theoretical_noise_std: float = 1.0 / f_d

    try:
        with torch.no_grad():
            n_img: torch.Tensor = _to_float_tensor(
                science_image,
                device=device,
                dtype=torch.float64,
                name="science_image",
            )
            r_img: torch.Tensor = _to_float_tensor(
                reference_image,
                device=device,
                dtype=torch.float64,
                name="reference_image",
            )
            p_n_raw: torch.Tensor = _to_float_tensor(
                science_psf,
                device=device,
                dtype=torch.float64,
                name="science_psf",
            )
            p_r_raw: torch.Tensor = _to_float_tensor(
                reference_psf,
                device=device,
                dtype=torch.float64,
                name="reference_psf",
            )

            p_n_shifted: torch.Tensor = _embed_psf_zero_phase(
                p_n_raw,
                (ih, iw),
                name="science_psf",
            )
            p_r_shifted: torch.Tensor = _embed_psf_zero_phase(
                p_r_raw,
                (ih, iw),
                name="reference_psf",
            )

            n_hat: torch.Tensor = torch.fft.fft2(n_img)
            r_hat: torch.Tensor = torch.fft.fft2(r_img)
            p_n_hat: torch.Tensor = torch.fft.fft2(p_n_shifted)
            p_r_hat: torch.Tensor = torch.fft.fft2(p_r_shifted)

            denom_sq: torch.Tensor = (
                (sig_n**2) * (f_r**2) * (p_r_hat.abs() ** 2)
                + (sig_r**2) * (f_n**2) * (p_n_hat.abs() ** 2)
            )
            denom: torch.Tensor = torch.sqrt(torch.clamp(denom_sq, min=_EPSILON**2))

            # Unit-variance white-noise difference spectrum D_unit_hat (ZOGY Eq. 13 / F_D)
            d_unit_hat: torch.Tensor = (f_r * p_r_hat * n_hat - f_n * p_n_hat * r_hat) / denom

            # Difference PSF spectrum P_D_hat (ZOGY Eq. 14)
            p_d_hat: torch.Tensor = (f_r * f_n * p_r_hat * p_n_hat) / (f_d * denom)

            # Calibrated flux-scaled difference image D (noise std == 1 / F_D)
            d_unit_spatial: torch.Tensor = torch.fft.ifft2(d_unit_hat).real
            diff_image: torch.Tensor = (d_unit_spatial / f_d).to(dtype=torch.float32)

            # Spatial difference PSF centered at (ih // 2, iw // 2)
            p_d_zero_phase: torch.Tensor = torch.fft.ifft2(p_d_hat).real
            p_d_centered: torch.Tensor = torch.fft.fftshift(p_d_zero_phase).to(
                dtype=torch.float32
            )

            # Matched-filter SNR significance map S_corr in Gaussian sigma units (ZOGY Eq. 16-17)
            p_d_norm_l2: float = math.sqrt(
                max(float(torch.sum(p_d_zero_phase**2).item()), _EPSILON**2)
            )
            s_raw: torch.Tensor = torch.fft.ifft2(torch.conj(p_d_hat) * d_unit_hat).real
            sig_map: torch.Tensor = (s_raw / p_d_norm_l2).to(dtype=torch.float32)

            return ZOGYResult(
                difference_image=diff_image,
                significance_map=sig_map,
                difference_psf=p_d_centered,
                theoretical_noise_std=theoretical_noise_std,
                flux_scale_diff=f_d,
            )
    except torch.OutOfMemoryError as exc:
        msg = f"Hardware GPU out-of-memory encountered during zogy_subtract: {exc}"
        raise VRAMLimitExceededError(msg) from exc
