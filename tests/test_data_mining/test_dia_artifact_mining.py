"""Unit and integration tests for DIA optical subtraction artifact mining."""

from __future__ import annotations

import numpy as np
import pytest

from LUMIA.data_mining.artifact_filter import (
    DeepStreaksArtifactFilter,
    DIAArtifactMiner,
)


def test_dipole_detection_on_subtraction_residuals() -> None:
    """Validate detection of positive/negative paired dipole subtraction artifacts."""
    h, w = 48, 48
    stamp = np.zeros((h, w), dtype=np.float32)

    # Positive lobe at (24, 22)
    stamp[24, 22] = 50.0
    stamp[23:26, 21:24] += 25.0

    # Negative lobe at (24, 26) - 4 pixels separation
    stamp[24, 26] = -45.0
    stamp[23:26, 25:28] -= 22.0

    filt = DeepStreaksArtifactFilter()
    assessment = filt.evaluate_cutout(stamp)

    assert assessment.is_artifact is True
    assert assessment.artifact_type == "DIPOLE"
    assert assessment.hard_negative_taxonomy == "ARTIFACT_DIPOLE"
    assert assessment.dipole_score > 0.5


def test_isolated_source_is_not_dipole() -> None:
    """Single positive point source is not flagged as a dipole."""
    h, w = 48, 48
    stamp = np.zeros((h, w), dtype=np.float32)
    stamp[24, 24] = 60.0
    stamp[23:26, 23:26] += 30.0

    filt = DeepStreaksArtifactFilter()
    assessment = filt.evaluate_cutout(stamp)
    assert assessment.artifact_type != "DIPOLE"


def test_bleeding_column_detection() -> None:
    """Validate detection of saturated CCD bleeding column charge leakage."""
    h, w = 48, 48
    stamp = np.zeros((h, w), dtype=np.float32)
    # Bleed column along x=24 spanning full height
    stamp[:, 24] = 80.0
    stamp[:, 23] = 20.0
    stamp[:, 25] = 20.0

    filt = DeepStreaksArtifactFilter()
    assessment = filt.evaluate_cutout(stamp)

    assert assessment.is_artifact is True
    assert assessment.artifact_type == "BLEEDING_COLUMN"
    assert assessment.hard_negative_taxonomy == "ARTIFACT_BLEEDING_COLUMN"


def test_diffraction_spike_detection() -> None:
    """Validate detection of thin linear diffraction spikes from bright stars."""
    h, w = 48, 48
    stamp = np.zeros((h, w), dtype=np.float32)
    stamp[24, :] = 40.0  # Horizontal spike
    stamp[:, 24] = 40.0  # Vertical spike

    filt = DeepStreaksArtifactFilter()
    assessment = filt.evaluate_cutout(stamp)

    assert assessment.is_artifact is True
    assert assessment.artifact_type in ("DIFFRACTION_SPIKE", "SATELLITE_TRAIL", "BLEEDING_COLUMN")


def test_dia_artifact_miner_scans_and_extracts_frame() -> None:
    """Validate DIAArtifactMiner scanning a 256x256 difference frame."""
    frame = np.zeros((256, 256), dtype=np.float32)

    # 1. Place a dipole at (64, 64)
    frame[64, 62] = 60.0
    frame[64, 66] = -55.0

    # 2. Place a bleed column at (180, 180)
    frame[150:210, 180] = 70.0

    miner = DIAArtifactMiner()
    mined = miner.mine_artifacts_from_difference_frame(
        difference_frame=frame,
        stamp_size=48,
        detection_threshold_sigma=3.5,
        max_artifacts=10,
    )

    assert len(mined) >= 2
    types = {m["artifact_type"] for m in mined}
    assert "DIPOLE" in types
    assert any(m["artifact_type"] in ("BLEEDING_COLUMN", "SATELLITE_TRAIL") for m in mined)
