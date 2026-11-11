"""Pure PyTorch engine for non-destructive synthetic moving-object (asteroid) injection.

Operates fully vectorized on CPU and CUDA torch.Tensor instances with intra-exposure
PSF motion trailing, flux calibration, and Poisson detector shot noise.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch
from pydantic import ConfigDict, Field

from LUMIA.core.schemas import LumiaBaseModel

if TYPE_CHECKING:
    pass

__all__ = [
    "GAUSSIAN_FWHM_TO_SIGMA",
    "SyntheticInjectionConfig",
    "SyntheticInjector",
    "inject_faint_moving_source",
    "inject_synthetic_asteroid",
]

GAUSSIAN_FWHM_TO_SIGMA: float = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
"""Conversion factor from Full Width at Half Maximum (FWHM) to Gaussian standard deviation sigma."""


class SyntheticInjectionConfig(LumiaBaseModel):
    """Immutable parameters defining a synthetic source injection campaign."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    magnitude: float = Field(..., description="Apparent top-of-atmosphere visual magnitude m.")
    zero_point: float = Field(..., description="Photometric zero-point ZP in instrumental mag.")
    fwhm_pixels: float = Field(
        ..., gt=0.0, description="Full Width at Half Maximum of seeing PSF in pixels."
    )
    exposure_time: float = Field(
        default=30.0, gt=0.0, description="Exposure shutter duration t_exp in seconds."
    )
    gain: float = Field(
        default=1.0, gt=0.0, description="Detector conversion gain in electrons per ADU (e-/ADU)."
    )
    dx: float = Field(
        default=0.0, description="Intra-exposure column displacement in pixels during t_exp."
    )
    dy: float = Field(
        default=0.0, description="Intra-exposure row displacement in pixels during t_exp."
    )
    saturation_limit_adu: float = Field(
        default=65535.0, gt=0.0, description="Detector full-well saturation ceiling in ADU."
    )
    include_poisson_noise: bool = Field(
        default=True, description="Whether to apply Poisson photoelectron shot noise to signal."
    )


def inject_synthetic_asteroid(
    science_tensor: torch.Tensor,
    x0: float,
    y0: float,
    magnitude: float,
    zero_point: float,
    fwhm_pixels: float,
    dx: float = 0.0,
    dy: float = 0.0,
    exposure_time: float = 30.0,
    gain: float = 1.0,
    *,
    saturation_limit_adu: float = 65535.0,
    include_poisson_noise: bool = True,
    generator: torch.Generator | None = None,
    trail_anchor: Literal["center", "start"] = "center",
    in_place: bool = False,
) -> torch.Tensor:
    """Inject a synthetic moving point source into a calibrated 2D science image tensor.

    Physical Formulation
    --------------------
    1. Instrumental Flux Integration:
       $$F_{\\text{ADU}} = 10^{-0.4(m - ZP)} \\cdot t_{\\text{exp}}$$

    2. Stationary 2D Gaussian PSF Profile:
       $$\\sigma = \\frac{\\text{FWHM}}{2\\sqrt{2\\ln 2}} \\approx \\frac{\\text{FWHM}}{2.35482}$$
       $$A = \\frac{F_{\\text{ADU}}}{2\\pi \\sigma^2}$$

    3. Intra-Exposure Motion Trailing:
       Evaluates $M$ discrete quadrature sub-steps along trajectory $(\\Delta x, \\Delta y)$:
       $$M = \\max(1, \\lceil 4 \\sqrt{\\Delta x^2 + \\Delta y^2} \\rceil)$$
       $$I_{\\text{trailed}}(x, y) = \\frac{A}{M} \\sum_{k=0}^{M-1} \\exp\\left(-\\frac{(x - x_k)^2 + (y - y_k)^2}{2\\sigma^2}\\right)$$

    4. Poisson Shot Noise Realization:
       $$\\lambda_e(x, y) = I_{\\text{trailed}}(x, y) \\cdot g$$
       $$P_e(x, y) \\sim \\text{Poisson}(\\lambda_e(x, y))$$
       $$I_{\\text{noisy}}(x, y) = \\frac{P_e(x, y)}{g}$$

    5. Saturation Clipping:
       $$S_{\\text{mock}} = \\min(S_{\\text{raw}} + I_{\\text{noisy}}, \\; \\text{saturation\\_limit\\_adu})$$

    Parameters
    ----------
    science_tensor : torch.Tensor
        Input 2D calibrated science image tensor of shape (H, W).
    x0 : float
        Centroid column position in zero-indexed pixel coordinates.
    y0 : float
        Centroid row position in zero-indexed pixel coordinates.
    magnitude : float
        Apparent visual magnitude m.
    zero_point : float
        Photometric zero-point ZP.
    fwhm_pixels : float
        PSF Full Width at Half Maximum in pixels (must be > 0).
    dx : float, optional
        Intra-exposure displacement along column axis in pixels (default 0.0).
    dy : float, optional
        Intra-exposure displacement along row axis in pixels (default 0.0).
    exposure_time : float, optional
        Exposure duration in seconds (default 30.0).
    gain : float, optional
        Detector conversion gain in electrons per ADU (default 1.0).
    saturation_limit_adu : float, optional
        Saturation clipping ceiling in ADU (default 65535.0).
    include_poisson_noise : bool, optional
        If True, applies Poisson shot noise to injected flux (default True).
    generator : torch.Generator | None, optional
        PyTorch random generator for deterministic noise sampling.
    trail_anchor : Literal["center", "start"], optional
        Whether (x0, y0) represents the exposure midpoint ("center") or start point ("start").
    in_place : bool, optional
        If True, modifies science_tensor in-place; otherwise returns a cloned tensor (default False).

    Returns
    -------
    torch.Tensor
        The injected science tensor on the same device and dtype as science_tensor.
    """
    if not isinstance(science_tensor, torch.Tensor):
        msg = f"science_tensor must be a torch.Tensor, got {type(science_tensor)}."
        raise TypeError(msg)
    if science_tensor.ndim != 2:
        msg = f"science_tensor must be 2-dimensional (H, W), got shape {tuple(science_tensor.shape)}."
        raise ValueError(msg)
    if fwhm_pixels <= 0.0:
        msg = f"fwhm_pixels must be positive, got {fwhm_pixels}."
        raise ValueError(msg)
    if gain <= 0.0:
        msg = f"gain must be positive, got {gain}."
        raise ValueError(msg)
    if exposure_time <= 0.0:
        msg = f"exposure_time must be positive, got {exposure_time}."
        raise ValueError(msg)

    device = science_tensor.device
    dtype = science_tensor.dtype if science_tensor.is_floating_point() else torch.float32
    height, width = int(science_tensor.shape[0]), int(science_tensor.shape[1])

    # 1. Total Instrumental Flux Calculation
    flux_adu = (10.0 ** (-0.4 * (magnitude - zero_point))) * exposure_time
    sigma = fwhm_pixels * GAUSSIAN_FWHM_TO_SIGMA
    amplitude = flux_adu / (2.0 * math.pi * (sigma**2))

    # 2. Motion Trailing Quadrature Sub-Steps
    trail_length = math.sqrt(dx**2 + dy**2)
    m_steps = max(1, math.ceil(4.0 * trail_length))

    sub_s = (torch.arange(m_steps, device=device, dtype=dtype) + 0.5) / float(m_steps)
    if trail_anchor == "center":
        xk = (x0 + (sub_s - 0.5) * dx).view(m_steps, 1, 1)
        yk = (y0 + (sub_s - 0.5) * dy).view(m_steps, 1, 1)
    else:
        xk = (x0 + sub_s * dx).view(m_steps, 1, 1)
        yk = (y0 + sub_s * dy).view(m_steps, 1, 1)

    # 3. Local Bounding Box Extraction (Minimum 31x31 footprint)
    rx = max(15, math.ceil(4.0 * sigma + abs(dx) / 2.0))
    ry = max(15, math.ceil(4.0 * sigma + abs(dy) / 2.0))

    xmin = max(0, int(math.floor(x0 - rx)))
    xmax = min(width, int(math.ceil(x0 + rx)) + 1)
    ymin = max(0, int(math.floor(y0 - ry)))
    ymax = min(height, int(math.ceil(y0 + ry)) + 1)

    if xmin >= xmax or ymin >= ymax:
        # Injection patch completely off-detector
        return science_tensor if in_place else science_tensor.clone()

    # 4. Trailed 2D Gaussian Surface Brightness Evaluation
    y_coords = torch.arange(ymin, ymax, device=device, dtype=dtype).view(1, -1, 1)
    x_coords = torch.arange(xmin, xmax, device=device, dtype=dtype).view(1, 1, -1)

    r2 = (x_coords - xk) ** 2 + (y_coords - yk) ** 2
    gaussian_substeps = (amplitude / float(m_steps)) * torch.exp(-r2 / (2.0 * (sigma**2)))
    trailed_patch = torch.sum(gaussian_substeps, dim=0)

    # 5. Poisson Detector Noise Realization
    if include_poisson_noise:
        signal_electrons = torch.clamp(trailed_patch * gain, min=0.0)
        if generator is not None:
            noisy_electrons = torch.poisson(signal_electrons, generator=generator)
        else:
            noisy_electrons = torch.poisson(signal_electrons)
        injected_patch = noisy_electrons / gain
    else:
        injected_patch = trailed_patch

    # 6. Additive Composite & Saturation Ceiling
    output = science_tensor if in_place else science_tensor.clone()
    combined_patch = output[ymin:ymax, xmin:xmax] + injected_patch
    output[ymin:ymax, xmin:xmax] = torch.clamp(combined_patch, min=0.0, max=saturation_limit_adu)

    return output


class SyntheticInjector:
    """Configurable domain service for injecting synthetic moving objects into science frames."""

    def __init__(
        self,
        zero_point: float,
        gain: float,
        fwhm_pixels: float,
        saturation_limit_adu: float = 65535.0,
    ) -> None:
        """Initialize synthetic injector with telescope characteristics.

        Parameters
        ----------
        zero_point : float
            Photometric zero point ZP.
        gain : float
            Detector conversion gain in e-/ADU.
        fwhm_pixels : float
            Seeing PSF Full Width at Half Maximum in pixels.
        saturation_limit_adu : float, optional
            Full-well detector saturation limit in ADU (default 65535.0).
        """
        if fwhm_pixels <= 0.0:
            msg = f"fwhm_pixels must be positive, got {fwhm_pixels}."
            raise ValueError(msg)
        if gain <= 0.0:
            msg = f"gain must be positive, got {gain}."
            raise ValueError(msg)

        self._zero_point: float = float(zero_point)
        self._gain: float = float(gain)
        self._fwhm_pixels: float = float(fwhm_pixels)
        self._saturation_limit_adu: float = float(saturation_limit_adu)

    @property
    def zero_point(self) -> float:
        """Return photometric zero-point ZP."""
        return self._zero_point

    @property
    def gain(self) -> float:
        """Return detector gain in e-/ADU."""
        return self._gain

    @property
    def fwhm_pixels(self) -> float:
        """Return seeing FWHM in pixels."""
        return self._fwhm_pixels

    @property
    def sigma_pixels(self) -> float:
        """Return Gaussian standard deviation sigma in pixels."""
        return self._fwhm_pixels * GAUSSIAN_FWHM_TO_SIGMA

    @property
    def saturation_limit_adu(self) -> float:
        """Return saturation clipping limit in ADU."""
        return self._saturation_limit_adu

    def compute_flux_adu(self, magnitude: float, exposure_time: float) -> float:
        """Calculate total integrated instrumental flux in ADU.

        $$F_{\\text{ADU}} = 10^{-0.4(m - ZP)} \\cdot t_{\\text{exp}}$$
        """
        if exposure_time <= 0.0:
            msg = f"exposure_time must be positive, got {exposure_time}."
            raise ValueError(msg)
        return (10.0 ** (-0.4 * (magnitude - self._zero_point))) * exposure_time

    def inject(
        self,
        science_tensor: torch.Tensor,
        x0: float,
        y0: float,
        magnitude: float,
        exposure_time: float = 30.0,
        dx: float = 0.0,
        dy: float = 0.0,
        *,
        include_poisson_noise: bool = True,
        generator: torch.Generator | None = None,
        trail_anchor: Literal["center", "start"] = "center",
        in_place: bool = False,
    ) -> torch.Tensor:
        """Inject synthetic moving object into science image using configured parameters."""
        return inject_synthetic_asteroid(
            science_tensor=science_tensor,
            x0=x0,
            y0=y0,
            magnitude=magnitude,
            zero_point=self._zero_point,
            fwhm_pixels=self._fwhm_pixels,
            dx=dx,
            dy=dy,
            exposure_time=exposure_time,
            gain=self._gain,
            saturation_limit_adu=self._saturation_limit_adu,
            include_poisson_noise=include_poisson_noise,
            generator=generator,
            trail_anchor=trail_anchor,
            in_place=in_place,
        )


def inject_faint_moving_source(
    science_tensor: torch.Tensor,
    x0: float,
    y0: float,
    target_snr: float,
    background_rms: float,
    zero_point: float = 26.0,
    fwhm_pixels: float = 2.0,
    dx: float = 0.0,
    dy: float = 0.0,
    exposure_time: float = 30.0,
    gain: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, float]:
    """Injects a synthetic moving point source calibrated to a specific low SNR (1.5 <= SNR < 5.0).

    Computes the visual magnitude required to achieve target_snr given the local
    background RMS noise level, and injects intra-exposure motion trailing.

    Parameters
    ----------
    science_tensor : torch.Tensor
        Input 2D calibrated science image tensor.
    x0, y0 : float
        Centroid coordinates in pixels.
    target_snr : float
        Target signal-to-noise ratio in sigma (e.g. 2.5).
    background_rms : float
        Background noise standard deviation in ADU.
    zero_point : float
        Photometric zero-point magnitude.
    fwhm_pixels : float
        Point Spread Function FWHM in pixels.
    dx, dy : float
        Intra-exposure trail displacement in pixels.
    exposure_time : float
        Exposure time in seconds.
    gain : float
        Detector gain in e-/ADU.
    generator : torch.Generator | None
        Optional random number generator.

    Returns
    -------
    injected_tensor : torch.Tensor
        Image tensor with synthetic source injected.
    magnitude : float
        Calculated apparent visual magnitude.
    """
    sigma_psf = fwhm_pixels * GAUSSIAN_FWHM_TO_SIGMA
    n_eff_pixels = 4.0 * math.pi * (sigma_psf**2)
    trail_length = math.sqrt(dx**2 + dy**2)
    trail_loss_factor = math.sqrt(1.0 + (trail_length / (2.0 * fwhm_pixels)) ** 2)

    noise_adu = background_rms * math.sqrt(n_eff_pixels) * trail_loss_factor
    target_flux_adu = target_snr * noise_adu

    if target_flux_adu <= 0.0:
        target_flux_adu = 1e-4

    mag = float(zero_point - 2.5 * math.log10(target_flux_adu / exposure_time))

    injected = inject_synthetic_asteroid(
        science_tensor=science_tensor,
        x0=x0,
        y0=y0,
        magnitude=mag,
        zero_point=zero_point,
        fwhm_pixels=fwhm_pixels,
        dx=dx,
        dy=dy,
        exposure_time=exposure_time,
        gain=gain,
        generator=generator,
    )
    return injected, mag

