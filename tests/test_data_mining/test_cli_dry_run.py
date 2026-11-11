"""Tests for the data-mining CLI, objects catalog, and safe dry-run visual QC export."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003

import numpy as np
from scripts.run_mining import run_cli

from LUMIA.data_mining.archive_crawler import CutoutTriplet, WCSMetadata
from LUMIA.data_mining.qc_export import export_cutout_triplet_qc_png
from LUMIA.data_mining.taxonomy import load_objects_catalog


def test_objects_catalog_loading_and_diversity() -> None:
    """Verify loading and taxonomy diversity of expanded configs/objects_catalog.json."""
    catalog_path = Path("configs/objects_catalog.json")
    assert catalog_path.exists(), "configs/objects_catalog.json must exist"

    entries = load_objects_catalog(catalog_path)
    assert len(entries) >= 40, f"Catalog must contain at least 40 objects, found {len(entries)}"

    # Verify all expected classes are present
    classes = {e.taxonomy_class for e in entries}
    assert {"NEO", "MBA", "COMET", "TNO", "ISO"}.issubset(classes)

    # Check specific required objects across all categories
    obj_ids = {e.object_id for e in entries}
    expected_ids = {
        # Diverse NEOs
        "99942",  # Apophis
        "101955",  # Bennu
        "2024 YR4",
        "65803",  # Didymos
        "2023 DW",
        "4179",  # Toutatis
        "1620",  # Geographos
        "3200",  # Phaethon
        "433",  # Eros
        "162173",  # Ryugu
        "25143",  # Itokawa
        # Prominent MBAs
        "1",  # Ceres
        "2",  # Pallas
        "3",  # Juno
        "4",  # Vesta
        "5",  # Astraea
        "6",  # Hebe
        "7",  # Iris
        "10",  # Hygiea
        "16",  # Psyche
        # Comets
        "C/2020 F3",  # NEOWISE
        "67P",  # Churyumov-Gerasimenko
        "1P",  # Halley
        "103P",  # Hartley 2
        "9P",  # Tempel 1
        "17P",  # Holmes
        # TNOs / Centaurs / SDOs
        "134340",  # Pluto
        "50000",  # Quaoar
        "90377",  # Sedna
        "2060",  # Chiron
        "136108",  # Haumea
        "136472",  # Makemake
        "136199",  # Eris
        "10199",  # Chariklo
        # ISOs
        "2I",  # Borisov
        "3I",  # ATLAS
    }
    assert expected_ids.issubset(obj_ids)

    # Verify unambiguous SPK IDs for key targets
    # Verify unambiguous numerical SPK/Horizons IDs for key targets
    entry_dict = {e.object_id: e for e in entries}
    assert entry_dict["67P"].spk_id in {"90000703", "1000145"}
    assert entry_dict["1"].spk_id == "2000001", "Ceres SPK ID must be 2000001"
    assert entry_dict["3"].spk_id == "2000003", "Juno SPK ID must be 2000003"
    assert entry_dict["4"].spk_id == "2000004", "Vesta SPK ID must be 2000004"
    assert entry_dict["9P"].spk_id == "90000192", "9P Tempel 1 must use unique numerical ID"
    assert entry_dict["17P"].spk_id == "90000287", "17P Holmes must use unique numerical ID"
    assert entry_dict["109P"].spk_id == "90000985", "109P Swift-Tuttle must use unique numerical ID"
    assert entry_dict["19P"].spk_id == "90000306", "19P Borrelly must use unique numerical ID"
    assert entry_dict["81P"].spk_id == "90000863", "81P Wild 2 must use unique numerical ID"

    # Verify fields for each entry
    for entry in entries:
        assert entry.name
        assert len(entry.magnitude_range) == 2
        assert entry.magnitude_range[0] <= entry.magnitude_range[1]
        target_spec = entry.to_target_spec()
        assert target_spec.target_id == entry.object_id
        assert target_spec.target_name == entry.name
        assert target_spec.spk_id == entry.spk_id


def test_ephemeris_point_robust_parsing_and_safeguards() -> None:
    """Verify EphemerisPoint gracefully handles missing magnitudes and invalid distances."""
    from LUMIA.data_mining.taxonomy import EphemerisPoint, _safe_float

    # 1. Test _safe_float helper directly
    assert _safe_float(None, default=20.0) == 20.0
    assert _safe_float(float("nan"), default=20.0) == 20.0
    assert _safe_float(np.nan, default=20.0) == 20.0
    assert _safe_float(np.ma.masked, default=20.0) == 20.0
    assert _safe_float(-5.0, default=1.0, min_val=1e-4) == 1e-4
    assert _safe_float(0.0, default=1.0, min_val=1e-4) == 1e-4

    # 2. Test EphemerisPoint validation when apparent_mag is NaN or None
    now = datetime.now(UTC)
    ep1 = EphemerisPoint(
        target_id="67P",
        timestamp_utc=now,
        bjd_tdb=2459000.5,
        ra_deg=180.0,
        dec_deg=15.0,
        ra_rate_arcsec_min=0.5,
        dec_rate_arcsec_min=0.2,
        total_speed_arcsec_min=0.54,
        apparent_mag=float("nan"),  # Masked / missing V from Horizons
        delta_au=0.0,  # Edge-case zero distance
        r_au=-1.5,  # Edge-case negative distance
        phase_angle_deg=10.0,
    )
    assert ep1.apparent_mag == 20.0
    assert ep1.delta_au >= 1e-4
    assert ep1.r_au >= 1e-4

    # 3. Test EphemerisPoint with masked value
    ep2 = EphemerisPoint(
        target_id="TEST_COMET",
        timestamp_utc=now,
        bjd_tdb=2459000.5,
        ra_deg=210.0,
        dec_deg=-10.0,
        ra_rate_arcsec_min=1.0,
        dec_rate_arcsec_min=-0.5,
        total_speed_arcsec_min=1.12,
        apparent_mag=np.ma.masked,  # Masked table element
        delta_au=1.2,
        r_au=2.5,
        phase_angle_deg=15.0,
    )
    assert ep2.apparent_mag == 20.0


def test_visual_qc_png_export(tmp_path: Path) -> None:
    """Verify export_cutout_triplet_qc_png produces a readable, formatted PNG file."""
    qc_dir = tmp_path / "qc_test"
    wcs = WCSMetadata(
        crpix=(64.0, 64.0),
        crval=(180.0, 20.0),
        cd=((-0.000281, 0.0), (0.0, 0.000281)),
    )
    data = np.random.default_rng(42).normal(100.0, 10.0, (3, 128, 128)).astype(np.float32)

    triplet = CutoutTriplet(
        target_id="TEST_ASTEROID",
        tracklet_id="TRK_001",
        epoch_index=0,
        timestamp_utc=datetime.now(UTC),
        bjd_tdb=2459000.5,
        ra_deg=180.0,
        dec_deg=20.0,
        target_x=64.0,
        target_y=64.0,
        wcs_params=wcs,
        stamp_size=128,
        filter_band="r",
        survey="ZTF",
        snr=8.5,
        mag_apparent=20.1,
        is_synthetic=False,
        is_artifact=False,
        artifact_type="NONE",
        data=data,
    )

    exported_path = export_cutout_triplet_qc_png(triplet, qc_dir, prefix="test_qc")
    assert exported_path.exists()
    assert exported_path.suffix == ".png"
    assert exported_path.stat().st_size > 1000  # Non-empty PNG image


def test_cli_dry_run_execution(tmp_path: Path) -> None:
    """Verify scripts/run_mining.py executes cleanly in --dry-run mode with visual QC."""
    h5_dest = tmp_path / "cli_dryrun.h5"
    qc_dir = tmp_path / "cli_debug_qc"

    exit_code = run_cli(
        [
            "--catalog",
            "configs/objects_catalog.json",
            "--output-hdf5",
            str(h5_dest),
            "--dry-run",
            "--max-cutouts",
            "5",
            "--qc-dir",
            str(qc_dir),
            "--objects",
            "99942",
            "1",
        ]
    )

    assert exit_code == 0
    assert h5_dest.exists()
    assert qc_dir.exists()

    png_files = list(qc_dir.glob("*.png"))
    assert len(png_files) == 5
    for png in png_files:
        assert png.stat().st_size > 5000


def test_cli_taxonomy_filter(tmp_path: Path) -> None:
    """Verify --taxonomy filter isolates desired classes in run_mining.py."""
    h5_dest = tmp_path / "taxonomy_filter.h5"
    qc_dir = tmp_path / "taxonomy_qc"

    exit_code = run_cli(
        [
            "--catalog",
            "configs/objects_catalog.json",
            "--output-hdf5",
            str(h5_dest),
            "--dry-run",
            "--max-cutouts",
            "3",
            "--qc-dir",
            str(qc_dir),
            "--taxonomy",
            "COMET",
        ]
    )

    assert exit_code == 0
    png_files = list(qc_dir.glob("*.png"))
    assert len(png_files) == 3
