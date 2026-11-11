"""Solar system target taxonomy, JPL Horizons client, and barycentric ephemeris engine."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, field_validator

from LUMIA.core.schemas import LumiaBaseModel
from LUMIA.physics.astrometry import compute_bjd_tdb
from LUMIA.physics.config import ObservatoryLocation

logger = logging.getLogger(__name__)

TaxonomyClass = Literal["NEO", "MBA", "COMET", "TNO", "ISO"]
TargetLabel = Literal["BOGUS_ARTIFACT", "REAL_POINT_SOURCE", "REAL_STREAK", "REAL_COMET_COMA"]
SplitName = Literal["train", "val", "test", "ood"]

LABEL_MAP: dict[TargetLabel, int] = {
    "BOGUS_ARTIFACT": 0,
    "REAL_POINT_SOURCE": 1,
    "REAL_STREAK": 2,
    "REAL_COMET_COMA": 3,
}

# Standard MPC Observatory Locations
PALOMAR_ZTF = ObservatoryLocation(
    latitude_deg=33.3563,
    longitude_deg=-116.8648,
    elevation_m=1712.0,
    mpc_code="I41",
)

HALEAKALA_PANSTARRS = ObservatoryLocation(
    latitude_deg=20.7075,
    longitude_deg=-156.2558,
    elevation_m=3055.0,
    mpc_code="F51",
)


class TargetObjectSpec(LumiaBaseModel):
    """Specification of a solar system target body for ephemeris retrieval."""

    target_id: str = Field(
        ..., min_length=1, max_length=64, description="JPL Horizons target identifier."
    )
    target_name: str = Field(
        ..., min_length=1, max_length=128, description="Common or astronomical name."
    )
    taxonomy_class: TaxonomyClass = Field(..., description="Dynamical solar system population.")
    has_coma: bool = Field(
        default=False, description="Whether target exhibits an extended cometary coma."
    )
    is_candidate: bool = Field(
        default=False, description="Flag indicating unconfirmed or candidate hyperbolic status."
    )
    h_mag: float | None = Field(default=None, description="Absolute magnitude H if available.")
    spk_id: str | None = Field(default=None, description="JPL SPK-ID if designated.")
    designation: str | None = Field(default=None, description="Formal IAU designation.")
    magnitude_range: list[float] | tuple[float, float] | None = Field(
        default=None, description="Estimated apparent magnitude range [V_min, V_max]."
    )


class CatalogObjectEntry(LumiaBaseModel):
    """Catalog specification for a solar system body in objects_catalog.json."""

    object_id: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=128)
    taxonomy_class: TaxonomyClass = Field(...)
    magnitude_range: list[float] | tuple[float, float] = Field(...)
    spk_id: str | None = Field(default=None)
    designation: str | None = Field(default=None)
    has_coma: bool = Field(default=False)
    is_candidate: bool = Field(
        default=False, description="Flag indicating unconfirmed or candidate hyperbolic status."
    )
    h_mag: float | None = Field(default=None)

    def to_target_spec(self) -> TargetObjectSpec:
        """Convert catalog entry into pipeline TargetObjectSpec."""
        return TargetObjectSpec(
            target_id=self.object_id,
            target_name=self.name,
            taxonomy_class=self.taxonomy_class,
            has_coma=self.has_coma,
            is_candidate=self.is_candidate,
            h_mag=self.h_mag,
            spk_id=self.spk_id,
            designation=self.designation,
            magnitude_range=self.magnitude_range,
        )


class DatasetSplitsManifest(LumiaBaseModel):
    """Manifest specifying target partitioning into train, val, test, and ood splits."""

    schema_version: str = Field(default="1.0.0")
    catalog_checksum_sha256: str = Field(
        ..., description="SHA-256 hash of source objects catalog."
    )
    created_at_utc: str = Field(..., description="ISO 8601 creation timestamp.")
    splits: dict[SplitName, list[str]] = Field(
        ..., description="Target IDs assigned to each partition."
    )
    class_distribution: dict[str, dict[SplitName, int]] = Field(
        ..., description="Target count per taxonomy class and partition."
    )


def generate_dataset_splits(
    catalog_path: Path | str,
    output_manifest_path: Path | str | None = None,
) -> DatasetSplitsManifest:
    """Generate deterministic stratified dataset splits (80/10/10 + OOD for ISOs).

    Solar system bodies are grouped by taxonomy class, sorted lexicographically by object_id,
    and deterministically partitioned into 80% train, 10% val, 10% test. Small classes
    (3 <= N < 10) have a floor constraint guaranteeing at least 1 object in val and 1 in test.
    Interstellar objects (ISO) are routed to the out-of-distribution (ood) split.
    """
    cat_path = Path(catalog_path)
    if not cat_path.exists():
        raise FileNotFoundError(f"Objects catalog file not found at: {cat_path}")

    catalog_bytes = cat_path.read_bytes()
    checksum = hashlib.sha256(catalog_bytes).hexdigest()

    entries = load_objects_catalog(cat_path)

    # Group entries by taxonomy class
    grouped: dict[str, list[str]] = {}
    for entry in entries:
        grouped.setdefault(entry.taxonomy_class, []).append(entry.object_id)

    splits: dict[SplitName, list[str]] = {
        "train": [],
        "val": [],
        "test": [],
        "ood": [],
    }
    class_distribution: dict[str, dict[SplitName, int]] = {}

    # Canonical class ordering for deterministic partition output
    canonical_classes = ["NEO", "MBA", "COMET", "TNO", "ISO"]
    ordered_classes = [c for c in canonical_classes if c in grouped] + sorted(
        c for c in grouped if c not in canonical_classes
    )

    for cls in ordered_classes:
        obj_ids = sorted(grouped[cls])
        n_total = len(obj_ids)
        class_distribution[cls] = {"train": 0, "val": 0, "test": 0, "ood": 0}

        if cls == "ISO":
            splits["ood"].extend(obj_ids)
            class_distribution[cls]["ood"] = n_total
        else:
            if n_total == 0:
                n_train = n_val = n_test = 0
            elif n_total == 1:
                n_train, n_val, n_test = 1, 0, 0
            elif n_total == 2:
                n_train, n_val, n_test = 1, 1, 0
            else:
                n_val = max(1, round(n_total * 0.1))
                n_test = max(1, round(n_total * 0.1))
                n_train = n_total - n_val - n_test

            train_ids = obj_ids[:n_train]
            val_ids = obj_ids[n_train : n_train + n_val]
            test_ids = obj_ids[n_train + n_val :]

            splits["train"].extend(train_ids)
            splits["val"].extend(val_ids)
            splits["test"].extend(test_ids)

            class_distribution[cls]["train"] = len(train_ids)
            class_distribution[cls]["val"] = len(val_ids)
            class_distribution[cls]["test"] = len(test_ids)

    manifest = DatasetSplitsManifest(
        schema_version="1.0.0",
        catalog_checksum_sha256=checksum,
        created_at_utc=datetime.now(UTC).isoformat(),
        splits=splits,
        class_distribution=class_distribution,
    )

    if output_manifest_path is not None:
        out_path = Path(output_manifest_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(mode="json"), f, indent=2)
            f.write("\n")

    return manifest


def load_objects_catalog(catalog_path: Path | str) -> list[CatalogObjectEntry]:
    """Load and validate the solar system target objects catalog JSON file."""
    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(f"Objects catalog file not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_targets = data.get("targets", []) if isinstance(data, dict) else data
    return [CatalogObjectEntry.model_validate(t) for t in raw_targets]


def _safe_float(
    value: Any,
    default: float,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    """Safely convert astroquery table cell or arbitrary value to a valid finite float."""
    if value is None:
        val = default
    else:
        # Handle astropy MaskedColumn / masked elements
        if getattr(value, "mask", False) is True or value is np.ma.masked:
            val = default
        else:
            try:
                val = float(value)
                if np.isnan(val) or np.isinf(val):
                    val = default
            except TypeError, ValueError:
                val = default

    if min_val is not None and val < min_val:
        val = min_val
    if max_val is not None and val > max_val:
        val = max_val
    return float(val)


class EphemerisPoint(LumiaBaseModel):
    """Calibrated state and astrometric position of a target at a single epoch."""

    target_id: str = Field(..., min_length=1, max_length=64)
    timestamp_utc: datetime = Field(..., description="Observation midpoint in UTC.")
    bjd_tdb: float = Field(..., description="Barycentric Julian Date in TDB scale.")
    ra_deg: float = Field(..., ge=0.0, lt=360.0, description="Right Ascension in ICRS degrees.")
    dec_deg: float = Field(..., ge=-90.0, le=90.0, description="Declination in ICRS degrees.")
    ra_rate_arcsec_min: float = Field(
        ..., description="Sky-plane motion rate dRA*cos(Dec)/dt [arcsec/min]."
    )
    dec_rate_arcsec_min: float = Field(
        ..., description="Sky-plane motion rate dDec/dt [arcsec/min]."
    )
    total_speed_arcsec_min: float = Field(
        ..., ge=0.0, description="Total apparent angular speed [arcsec/min]."
    )
    apparent_mag: float = Field(default=20.0, description="Predicted visual apparent magnitude V.")
    delta_au: float = Field(..., gt=0.0, description="Distance from observer to target in AU.")
    r_au: float = Field(..., gt=0.0, description="Distance from Sun to target in AU.")
    phase_angle_deg: float = Field(
        ..., ge=0.0, le=180.0, description="Sun-Target-Observer phase angle [deg]."
    )

    @field_validator("apparent_mag", mode="before")
    @classmethod
    def _validate_apparent_mag(cls, v: Any) -> float:
        return _safe_float(v, default=20.0)

    @field_validator("delta_au", mode="before")
    @classmethod
    def _validate_delta_au(cls, v: Any) -> float:
        return _safe_float(v, default=1.0, min_val=1e-4)

    @field_validator("r_au", mode="before")
    @classmethod
    def _validate_r_au(cls, v: Any) -> float:
        return _safe_float(v, default=1.0, min_val=1e-4)


class HorizonsQueryConfig(LumiaBaseModel):
    """Configuration for querying NASA JPL Horizons."""

    observatory_location: ObservatoryLocation = Field(default=PALOMAR_ZTF)
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    step_minutes: int = Field(default=15, ge=1, le=1440)


def compute_optimal_cadence_window(
    taxonomy_class: TaxonomyClass,
    angular_speed_arcsec_min: float,
    fwhm_arcsec: float = 1.5,
) -> float:
    """Compute the optimal observational time baseline Delta_t in minutes.

    Balancing requirements:
    1. Fast NEOs: Motion must not decorrelate across frames, but must exceed seeing disk
       separation (e.g. 2 * FWHM).
    2. Slow TNOs: Motion must be large enough to distinguish from stationary stars.

    Returns:
        Baseline Delta_t in minutes, bounded between 10.0 and 1440.0 minutes (up to 24 hours).
    """
    if angular_speed_arcsec_min <= 0.0:
        return 45.0

    # Desired displacement is approximately 3 times seeing FWHM
    target_displacement_arcsec = 3.0 * fwhm_arcsec
    ideal_minutes = target_displacement_arcsec / angular_speed_arcsec_min

    # Class-specific constraints
    if taxonomy_class == "NEO":
        return float(np.clip(ideal_minutes, 10.0, 30.0))
    if taxonomy_class == "MBA":
        return float(np.clip(ideal_minutes, 25.0, 50.0))
    if taxonomy_class == "COMET":
        return float(np.clip(ideal_minutes, 20.0, 45.0))
    if taxonomy_class == "TNO":
        # TNOs move 0.01 - 0.05 arcsec/min (< 1 px / hr). Multi-hour baselines (3h to 24h)
        # ensure >= 3 px displacement between consecutive frames.
        return float(np.clip(ideal_minutes, 180.0, 1440.0))
    # ISO
    return float(np.clip(ideal_minutes, 15.0, 40.0))


class HorizonsClient:
    """Client for generating ephemeris sequences via NASA JPL Horizons."""

    def __init__(self, default_location: ObservatoryLocation = PALOMAR_ZTF) -> None:
        self.default_location = default_location

    def query_ephemeris(
        self,
        target: TargetObjectSpec,
        config: HorizonsQueryConfig,
    ) -> list[EphemerisPoint]:
        """Fetch ephemeris track for a given target across the requested time window.

        Attempts to query astroquery.jplhorizons, with fallback to an analytical Keplerian
        approximation for offline integration testing.
        """
        try:
            from astroquery.jplhorizons import Horizons  # type: ignore[import-untyped]

            epochs = {
                "start": config.start_time.strftime("%Y-%m-%d %H:%M"),
                "stop": config.end_time.strftime("%Y-%m-%d %H:%M"),
                "step": f"{config.step_minutes}m",
            }
            # Format query ID to prevent Earth-Moon barycenter conflicts and disambiguate comets
            if target.spk_id:
                if (target.spk_id.startswith("9000") and len(target.spk_id) == 8) or (
                    target.spk_id.startswith("DES=")
                ):
                    query_id = target.spk_id
                else:
                    query_id = f"DES={target.spk_id};"
            else:
                # Suffix with ';' to prevent Horizons barycenter search (e.g. '3;' vs '3')
                query_id = (
                    f"{target.target_id};" if target.target_id.isdigit() else target.target_id
                )

            try:
                obj = Horizons(
                    id=query_id,
                    location=config.observatory_location.mpc_code or "500",
                    epochs=epochs,
                )
                eph_table = obj.ephemerides()
            except Exception as eph_err:
                fallback_id = target.target_id
                if query_id != fallback_id:
                    logger.debug(
                        "Horizons query by '%s' failed (%s), falling back to '%s'",
                        query_id,
                        eph_err,
                        fallback_id,
                    )
                    obj = Horizons(
                        id=fallback_id,
                        location=config.observatory_location.mpc_code or "500",
                        epochs=epochs,
                    )
                    eph_table = obj.ephemerides()
                else:
                    raise

            points: list[EphemerisPoint] = []
            for row in eph_table:
                dt_utc = datetime.strptime(str(row["datetime_str"]), "%Y-%b-%d %H:%M").replace(
                    tzinfo=UTC
                )
                ra_val = _safe_float(row.get("RA"), default=0.0) % 360.0
                dec_val = _safe_float(row.get("DEC"), default=0.0, min_val=-90.0, max_val=90.0)
                bjd_val = compute_bjd_tdb(
                    obs_time=dt_utc,
                    ra_deg=ra_val,
                    dec_deg=dec_val,
                    observatory=config.observatory_location,
                ).bjd_tdb

                dra = _safe_float(row.get("RA_rate"), default=0.0)
                ddec = _safe_float(row.get("DEC_rate"), default=0.0)
                speed = float(np.hypot(dra, ddec))

                v_mag = _safe_float(row.get("V"), default=20.0)
                delta_val = _safe_float(row.get("delta"), default=1.0, min_val=1e-4)
                r_val = _safe_float(row.get("r"), default=1.0, min_val=1e-4)
                alpha_val = _safe_float(row.get("alpha"), default=15.0, min_val=0.0, max_val=180.0)

                points.append(
                    EphemerisPoint(
                        target_id=target.target_id,
                        timestamp_utc=dt_utc,
                        bjd_tdb=bjd_val,
                        ra_deg=ra_val,
                        dec_deg=dec_val,
                        ra_rate_arcsec_min=dra,
                        dec_rate_arcsec_min=ddec,
                        total_speed_arcsec_min=speed,
                        apparent_mag=v_mag,
                        delta_au=delta_val,
                        r_au=r_val,
                        phase_angle_deg=alpha_val,
                    )
                )
            return points

        except Exception as exc:
            logger.warning(
                "Horizons API call for '%s' failed: %s. Emitting synthetic ephemeris.",
                target.target_id,
                exc,
            )
            return self._synthesize_fallback_ephemeris(target, config)

    def _synthesize_fallback_ephemeris(
        self,
        target: TargetObjectSpec,
        config: HorizonsQueryConfig,
    ) -> list[EphemerisPoint]:
        """Synthesize deterministic kinematic ephemeris points when offline."""
        speed_defaults: dict[TaxonomyClass, float] = {
            "NEO": 8.0,
            "MBA": 0.5,
            "COMET": 1.2,
            "TNO": 0.04,
            "ISO": 4.5,
        }
        nominal_speed = speed_defaults.get(target.taxonomy_class, 0.5)

        total_seconds = max((config.end_time - config.start_time).total_seconds(), 60.0)
        n_steps = max(int(total_seconds / (config.step_minutes * 60)), 1) + 1

        points: list[EphemerisPoint] = []
        base_ra = 180.0
        base_dec = 15.0

        for i in range(n_steps):
            elapsed_min = i * config.step_minutes
            dt_utc = datetime.fromtimestamp(
                config.start_time.timestamp() + (elapsed_min * 60), tz=UTC
            )
            dra = nominal_speed * np.cos(np.radians(35.0))
            ddec = nominal_speed * np.sin(np.radians(35.0))

            ra = (base_ra + (dra * elapsed_min) / 3600.0) % 360.0
            dec = float(np.clip(base_dec + (ddec * elapsed_min) / 3600.0, -90.0, 90.0))

            bjd_val = compute_bjd_tdb(
                obs_time=dt_utc,
                ra_deg=ra,
                dec_deg=dec,
                observatory=config.observatory_location,
            ).bjd_tdb

            points.append(
                EphemerisPoint(
                    target_id=target.target_id,
                    timestamp_utc=dt_utc,
                    bjd_tdb=bjd_val,
                    ra_deg=ra,
                    dec_deg=dec,
                    ra_rate_arcsec_min=dra,
                    dec_rate_arcsec_min=ddec,
                    total_speed_arcsec_min=nominal_speed,
                    apparent_mag=20.5 if target.h_mag is None else target.h_mag + 2.0,
                    delta_au=1.5,
                    r_au=2.2,
                    phase_angle_deg=18.5,
                )
            )
        return points
