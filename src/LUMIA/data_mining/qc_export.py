"""Visual Quality Control (QC) exporter for multi-channel astronomical cutouts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if TYPE_CHECKING:
    from LUMIA.data_mining.archive_crawler import CutoutTriplet

logger = logging.getLogger(__name__)


def export_cutout_triplet_qc_png(
    triplet: CutoutTriplet,
    output_dir: Path | str,
    prefix: str = "qc",
    cmap: str = "viridis",
) -> Path:
    """Export a high-resolution 3-panel PNG visualizing Science, Reference, and Difference cutouts.

    Annotates exact sub-pixel target coordinates (target_x, target_y) with a reticle/marker
    and records epoch timestamps, BJD_TDB, astrometric, and photometric metadata.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    channel_names = [
        "Science (Single Epoch)",
        "Reference (Deep Coadd)",
        "Difference (Subtraction)",
    ]

    for ch_idx, (ax, title) in enumerate(zip(axes, channel_names, strict=True)):
        img_data = triplet.data[ch_idx]

        finite_data = img_data[np.isfinite(img_data)]
        if finite_data.size > 0:
            vmin = float(np.percentile(finite_data, 1.0))
            vmax = float(np.percentile(finite_data, 99.0))
            if np.isclose(vmin, vmax):
                vmin -= 1.0
                vmax += 1.0
        else:
            vmin, vmax = 0.0, 1.0

        im = ax.imshow(
            img_data,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Draw red reticle on target coordinates
        tx, ty = triplet.target_x, triplet.target_y
        ax.plot([tx - 8, tx - 2], [ty, ty], color="red", lw=1.2)
        ax.plot([tx + 2, tx + 8], [ty, ty], color="red", lw=1.2)
        ax.plot([tx, tx], [ty - 8, ty - 2], color="red", lw=1.2)
        ax.plot([tx, tx], [ty + 2, ty + 8], color="red", lw=1.2)
        circle = plt.Circle((tx, ty), radius=5.0, color="red", fill=False, lw=1.2, ls="--")
        ax.add_patch(circle)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Detector X [px]", fontsize=10)
        ax.set_ylabel("Detector Y [px]", fontsize=10)
        ax.grid(True, color="gray", alpha=0.3, ls=":")

    iso_time = triplet.timestamp_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    header_text = (
        f"Target: {triplet.target_id} | Tracklet: {triplet.tracklet_id} | "
        f"Epoch: #{triplet.epoch_index}\n"
        f"BJD_TDB: {triplet.bjd_tdb:.5f} | Timestamp: {iso_time} | "
        f"Survey: {triplet.survey} ({triplet.filter_band})\n"
        f"RA: {triplet.ra_deg:.5f} deg | Dec: {triplet.dec_deg:.5f} deg | "
        f"Target Position: (X={triplet.target_x:.2f}, Y={triplet.target_y:.2f}) px\n"
        f"Apparent Mag V: {triplet.mag_apparent:.2f} | SNR: {triplet.snr:.1f} | "
        f"Artifact: {triplet.artifact_type} ({'QUARANTINED' if triplet.is_artifact else 'PASSED'})"
    )
    fig.suptitle(header_text, fontsize=11, y=1.06, va="bottom")

    safe_target = triplet.target_id.replace("/", "_").replace(" ", "_")
    ts_str = triplet.timestamp_utc.strftime("%Y%m%d_%H%M%S")
    dest_file = out_path / f"{prefix}_{safe_target}_ep{triplet.epoch_index:02d}_{ts_str}.png"

    fig.savefig(dest_file, dpi=120, bbox_inches="tight")
    plt.close(fig)

    logger.info("Exported QC cutout image: %s", dest_file)
    return dest_file
