"""Export multi-epoch astronomical tracklet cutouts for visual quality inspection with object ID annotations."""

from __future__ import annotations

import random
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def stretch_image(
    img: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.0,
) -> np.ndarray:
    """Robust astronomical percentile stretch for clear visibility."""
    valid_pixels = img[np.isfinite(img)]
    if len(valid_pixels) == 0:
        return img
    p_low, p_high = np.percentile(valid_pixels, [lower, upper])
    denom = p_high - p_low + 1e-5
    img_stretched = np.clip((img - p_low) / denom, 0.0, 1.0)
    return img_stretched


def main() -> None:
    """Inspect and render sample multi-epoch tracklets with object identifiers."""
    h5_path = Path("data/ground_truth/ztf_sso_tracklets_128x128.h5")
    out_dir = Path("data/visual_inspection")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening HDF5 database: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        split = f["splits/train"]
        labels = split["tracklet_labels"][:]
        offsets = split["tracklet_offsets"][:]
        lengths = split["tracklet_lengths"][:]
        triplets = split["triplets"]

        # Objekt-IDs einlesen (falls vorhanden, ansonsten Fallback auf Index)
        if "tracklet_object_ids" in split:
            object_ids = [
                val.decode("utf-8") if isinstance(val, bytes) else str(val)
                for val in split["tracklet_object_ids"][:]
            ]
        else:
            object_ids = [f"unknown_{i}" for i in range(len(labels))]

        class_names = {
            0: "BOGUS_ARTIFACT",
            1: "POINT_SOURCE",
            2: "STREAK",
            3: "COMET",
        }

        total_exported = 0
        for cls_idx, cls_name in class_names.items():
            valid_indices = np.where(labels == cls_idx)[0]
            if len(valid_indices) == 0:
                print(f"No tracklets found for class {cls_name} ({cls_idx}).")
                continue

            # Draw up to 3 random tracklets per class
            num_samples = min(3, len(valid_indices))
            sampled_indices = random.sample(list(valid_indices), num_samples)

            for tracklet_idx in sampled_indices:
                offset = int(offsets[tracklet_idx])
                length = int(lengths[tracklet_idx])
                obj_id = object_ids[tracklet_idx]

                # O(1) Slicing of sequence from HDF5 (N epochs)
                tracklet_frames = triplets[offset : offset + length]  # (N, 3, 128, 128)
                n_epochs = length

                fig, axes = plt.subplots(n_epochs, 3, figsize=(10, 3.5 * n_epochs))
                title = (
                    f"Class: {cls_name} | Tracklet ID: {tracklet_idx} | "
                    f"Obj: {obj_id} | Epochs: {n_epochs}"
                )
                fig.suptitle(title, fontsize=12, y=1.01)

                channel_names = ["Science", "Reference", "Difference"]

                for epoch in range(n_epochs):
                    for ch in range(3):
                        ax = axes[epoch, ch] if n_epochs > 1 else axes[ch]
                        img = tracklet_frames[epoch, ch]
                        img_stretched = stretch_image(img)

                        ax.imshow(img_stretched, origin="lower", cmap="viridis")

                        # Red crosshair at cutout center (64, 64)
                        ax.axhline(64, color="red", linestyle="--", alpha=0.6, linewidth=1)
                        ax.axvline(64, color="red", linestyle="--", alpha=0.6, linewidth=1)

                        if epoch == 0:
                            ax.set_title(channel_names[ch], fontsize=12, fontweight="bold")
                        if ch == 0:
                            ax.set_ylabel(f"Epoch {epoch}", fontsize=12, fontweight="bold")

                        ax.set_xticks([])
                        ax.set_yticks([])

                plt.tight_layout()
                # Bereinigter Dateiname mit Objekt-ID
                safe_obj_id = obj_id.replace("/", "_").replace(" ", "_")
                out_name = out_dir / f"class_{cls_idx}_{cls_name}_obj_{safe_obj_id}_idx_{tracklet_idx}.png"
                plt.savefig(out_name, dpi=150, bbox_inches="tight")
                plt.close()
                total_exported += 1
                print(f"Exported: {out_name}")

    print(f"\nDone! Exported {total_exported} images to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()