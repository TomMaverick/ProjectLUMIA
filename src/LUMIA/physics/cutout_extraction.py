"""Deterministic SNR peak extraction and vectorized 3-channel cutout slicing bridge (P2-2).

Pure in-memory domain bridge connecting Module 4 (``ZOGYResult``) to Module 7
(``RealBogusClassifier``) with zero database, orchestration, or HTTP dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import Field

from LUMIA.core.schemas import (
    CUTOUT_TRIPLET_CHANNELS,
    CutoutBatchResult,
    CutoutStampSize,
    LumiaBaseModel,
    SNRPeakCandidate,
)
from LUMIA.physics.gpu_processing import ZOGYResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

BoundaryMode = Literal["pad", "edge", "reject"]
DifferenceChannelSource = Literal["difference_image", "significance_map"]

__all__ = [
    "BoundaryMode",
    "CutoutBatchResult",
    "CutoutConfig",
    "CutoutStampSize",
    "DifferenceChannelSource",
    "PeakExtractionConfig",
    "SNRPeakCandidate",
    "extract_snr_peaks",
    "extract_zogy_cutout_batch",
    "slice_cutout_triplets",
]

_TRIPLET_CHANNELS: int = CUTOUT_TRIPLET_CHANNELS


class PeakExtractionConfig(LumiaBaseModel):
    """Validated parameters for deterministic local-maximum SNR peak extraction."""

    snr_threshold: float = Field(
        default=5.0,
        gt=0.0,
        description="Minimum matched-filter significance (sigma) required for a peak.",
    )
    nms_radius_px: int = Field(
        default=2,
        ge=1,
        description="Chebyshev radius R (window 2R+1) for non-maximum suppression.",
    )
    border_margin_px: int = Field(
        default=0,
        ge=0,
        description="Optional detector border exclusion margin in pixels.",
    )
    max_peaks: int | None = Field(
        default=None,
        ge=1,
        description="Optional cap on the number of returned peaks (highest SNR first).",
    )
    id_prefix: str = Field(
        default="cand",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.-]+$",
        description="Identifier prefix assigned to extracted peak candidates.",
    )


class CutoutConfig(LumiaBaseModel):
    """Validated configuration for vectorized 3-channel odd-dimension stamp slicing."""

    stamp_size: CutoutStampSize = Field(
        default=31,
        description="Odd square stamp side length in pixels (31 or 51).",
    )
    boundary_mode: BoundaryMode = Field(
        default="pad",
        description="Strategy when a stamp window extends past frame boundaries.",
    )
    pad_value: float = Field(
        default=0.0,
        description="Constant fill value applied outside frame edges when boundary_mode='pad'.",
    )
    difference_channel: DifferenceChannelSource = Field(
        default="difference_image",
        description="ZOGY tensor used for channel 2 of the extracted triplet.",
    )


def _to_2d_float32_tensor(name: str, array_or_tensor: torch.Tensor | ArrayLike) -> torch.Tensor:
    """Convert input to a non-aliased finite 2D float32 tensor without mutating the source."""
    if isinstance(array_or_tensor, torch.Tensor):
        tensor = array_or_tensor.detach().to(dtype=torch.float32)
    else:
        arr = np.asarray(array_or_tensor)
        if arr.dtype.kind not in {"f", "i", "u"}:
            msg = f"{name} must be numeric, got dtype {arr.dtype}."
            raise ValueError(msg)
        tensor = torch.as_tensor(arr, dtype=torch.float32)

    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        msg = f"{name} must be a non-empty 2D tensor, got shape {tuple(tensor.shape)}."
        raise ValueError(msg)
    if not bool(torch.all(torch.isfinite(tensor)).item()):
        msg = f"{name} contains non-finite (NaN or Inf) pixel values."
        raise ValueError(msg)
    return tensor


def _compute_3x3_centroid(
    sig_map: torch.Tensor,
    y_px: int,
    x_px: int,
) -> tuple[float, float]:
    """Compute positive-significance-weighted 3x3 sub-pixel centroid around (y_px, x_px)."""
    height, width = int(sig_map.shape[0]), int(sig_map.shape[1])
    y0 = max(0, y_px - 1)
    y1 = min(height, y_px + 2)
    x0 = max(0, x_px - 1)
    x1 = min(width, x_px + 2)

    patch = torch.clamp(sig_map[y0:y1, x0:x1], min=0.0)
    weight_sum = float(patch.sum().item())
    if weight_sum <= 0.0:
        return float(x_px), float(y_px)

    ys = torch.arange(y0, y1, dtype=torch.float32, device=sig_map.device).unsqueeze(1)
    xs = torch.arange(x0, x1, dtype=torch.float32, device=sig_map.device).unsqueeze(0)

    cx = float((patch * xs).sum().item()) / weight_sum
    cy = float((patch * ys).sum().item()) / weight_sum
    cx = min(max(cx, 0.0), float(width - 1))
    cy = min(max(cy, 0.0), float(height - 1))
    return cx, cy


def extract_snr_peaks(
    source: ZOGYResult | torch.Tensor | ArrayLike,
    config: PeakExtractionConfig | None = None,
) -> list[SNRPeakCandidate]:
    """Deterministically extract local-maximum SNR peaks from a ZOGY significance map.

    Peaks are filtered by ``snr_threshold`` and ``border_margin_px``, suppressed within
    a ``(2 * nms_radius_px + 1)`` Chebyshev window using vectorized 2D max-pooling plus
    deterministic plateau tie-breaking, and returned ordered by ``(-snr, y_px, x_px)``.
    """
    cfg = config if config is not None else PeakExtractionConfig()
    raw_map: torch.Tensor | ArrayLike = (
        source.significance_map if isinstance(source, ZOGYResult) else source
    )
    sig_map = _to_2d_float32_tensor("significance_map", raw_map)
    height, width = int(sig_map.shape[0]), int(sig_map.shape[1])

    margin = cfg.border_margin_px
    if 2 * margin >= height or 2 * margin >= width:
        return []

    radius = cfg.nms_radius_px
    kernel = 2 * radius + 1
    with torch.no_grad():
        pooled = F.max_pool2d(
            sig_map.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel,
            stride=1,
            padding=radius,
        ).squeeze(0).squeeze(0)

        candidate_mask = (sig_map >= cfg.snr_threshold) & (sig_map == pooled)
        if margin > 0:
            border_mask = torch.zeros_like(candidate_mask)
            border_mask[margin : height - margin, margin : width - margin] = True
            candidate_mask = candidate_mask & border_mask

        coords = torch.nonzero(candidate_mask, as_tuple=False).cpu()
        if coords.numel() == 0:
            return []

        sig_cpu = sig_map.cpu()
        raw_entries: list[tuple[float, int, int]] = [
            (
                float(sig_cpu[int(row[0].item()), int(row[1].item())].item()),
                int(row[0].item()),
                int(row[1].item()),
            )
            for row in coords
        ]

    # Deterministic ordering: highest SNR first, ties broken by ascending (y_px, x_px)
    raw_entries.sort(key=lambda item: (-item[0], item[1], item[2]))

    # Deterministic plateau suppression within Chebyshev radius
    accepted: list[tuple[float, int, int]] = []
    for snr_val, y_px, x_px in raw_entries:
        if any(
            max(abs(y_px - ay), abs(x_px - ax)) <= radius
            for _, ay, ax in accepted
        ):
            continue
        accepted.append((snr_val, y_px, x_px))
        if cfg.max_peaks is not None and len(accepted) >= cfg.max_peaks:
            break

    sig_cpu = sig_map.cpu()
    peaks: list[SNRPeakCandidate] = []
    for idx, (snr_val, y_px, x_px) in enumerate(accepted):
        cx, cy = _compute_3x3_centroid(sig_cpu, y_px, x_px)
        peaks.append(
            SNRPeakCandidate(
                candidate_id=f"{cfg.id_prefix}-{idx:04d}",
                x_px=x_px,
                y_px=y_px,
                x_centroid_px=cx,
                y_centroid_px=cy,
                snr=snr_val,
            )
        )
    return peaks


def slice_cutout_triplets(
    science_frame: torch.Tensor | ArrayLike,
    reference_frame: torch.Tensor | ArrayLike,
    difference_frame: torch.Tensor | ArrayLike,
    peaks: Sequence[SNRPeakCandidate],
    config: CutoutConfig | None = None,
) -> CutoutBatchResult:
    """Slice aligned 3-channel (Science, Reference, Difference) stamps in memory.

    Uses vectorized PyTorch tensor indexing with zero Python pixel loops and never
    mutates input frames. Center pixel ``(stamp_size // 2, stamp_size // 2)`` of each
    extracted stamp corresponds exactly to ``(peak.y_px, peak.x_px)``.
    """
    cfg = config if config is not None else CutoutConfig()
    sci = _to_2d_float32_tensor("science_frame", science_frame)
    ref = _to_2d_float32_tensor("reference_frame", reference_frame).to(device=sci.device)
    diff = _to_2d_float32_tensor("difference_frame", difference_frame).to(device=sci.device)

    if sci.shape != ref.shape or sci.shape != diff.shape:
        msg = (
            "science_frame, reference_frame, and difference_frame must share identical "
            f"2D shapes; got {tuple(sci.shape)}, {tuple(ref.shape)}, {tuple(diff.shape)}."
        )
        raise ValueError(msg)

    height, width = int(sci.shape[0]), int(sci.shape[1])
    stamp_size: int = cfg.stamp_size
    half: int = stamp_size // 2

    for peak in peaks:
        if peak.x_px >= width or peak.y_px >= height:
            msg = (
                f"Peak '{peak.candidate_id}' coordinate ({peak.x_px}, {peak.y_px}) lies "
                f"outside frame dimensions (width={width}, height={height})."
            )
            raise ValueError(msg)

    retained_peaks: list[SNRPeakCandidate]
    rejected_count: int = 0
    if cfg.boundary_mode == "reject":
        retained_peaks = []
        for peak in peaks:
            if (
                half <= peak.x_px < width - half
                and half <= peak.y_px < height - half
            ):
                retained_peaks.append(peak)
            else:
                rejected_count += 1
    else:
        retained_peaks = list(peaks)

    if not retained_peaks:
        empty_triplets = torch.empty(
            (0, _TRIPLET_CHANNELS, stamp_size, stamp_size),
            dtype=torch.float32,
            device="cpu",
        )
        return CutoutBatchResult(
            triplets=empty_triplets,
            candidates=[],
            stamp_size=cfg.stamp_size,
            rejected_boundary_count=rejected_count,
        )

    with torch.no_grad():
        stack_3ch = torch.stack((sci, ref, diff), dim=0)  # (3, H, W)
        y_centers = torch.tensor(
            [p.y_px for p in retained_peaks],
            dtype=torch.long,
            device=sci.device,
        )
        x_centers = torch.tensor(
            [p.x_px for p in retained_peaks],
            dtype=torch.long,
            device=sci.device,
        )

        if cfg.boundary_mode == "edge":
            rel = torch.arange(-half, half + 1, dtype=torch.long, device=sci.device)
            rows = torch.clamp(
                y_centers[:, None, None] + rel[None, :, None],
                min=0,
                max=height - 1,
            )
            cols = torch.clamp(
                x_centers[:, None, None] + rel[None, None, :],
                min=0,
                max=width - 1,
            )
            triplets = stack_3ch[:, rows, cols].permute(1, 0, 2, 3).contiguous()
        else:
            # Used for both 'pad' and 'reject' (interior-only windows never touch padding)
            padded = F.pad(
                stack_3ch.unsqueeze(0),
                (half, half, half, half),
                mode="constant",
                value=cfg.pad_value,
            ).squeeze(0)
            offsets = torch.arange(stamp_size, dtype=torch.long, device=sci.device)
            rows = y_centers[:, None, None] + offsets[None, :, None]
            cols = x_centers[:, None, None] + offsets[None, None, :]
            triplets = padded[:, rows, cols].permute(1, 0, 2, 3).contiguous()

    return CutoutBatchResult(
        triplets=triplets.cpu(),
        candidates=retained_peaks,
        stamp_size=cfg.stamp_size,
        rejected_boundary_count=rejected_count,
    )


def extract_zogy_cutout_batch(
    science_frame: torch.Tensor | ArrayLike,
    reference_frame: torch.Tensor | ArrayLike,
    zogy_result: ZOGYResult,
    *,
    peak_config: PeakExtractionConfig | None = None,
    cutout_config: CutoutConfig | None = None,
) -> CutoutBatchResult:
    """Extract SNR peaks from ZOGYResult and slice 3-channel CNN triplets in one step."""
    c_cfg = cutout_config if cutout_config is not None else CutoutConfig()
    peaks = extract_snr_peaks(zogy_result, config=peak_config)
    diff_frame: torch.Tensor = (
        zogy_result.difference_image
        if c_cfg.difference_channel == "difference_image"
        else zogy_result.significance_map
    )
    return slice_cutout_triplets(
        science_frame=science_frame,
        reference_frame=reference_frame,
        difference_frame=diff_frame,
        peaks=peaks,
        config=c_cfg,
    )
