"""DuckDB repositories encapsulating candidate-detection and cutout DDL/DML (P0-1, P0-4)."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from LUMIA.core.schemas import CandidateDetection
from LUMIA.data_pipeline.database import CatalogDatabase, _validate_identifier

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

CutoutStampSize = Literal[31, 51]

_REPO_MODEL_CONFIG: ConfigDict = ConfigDict(
    strict=True,
    extra="forbid",
    frozen=True,
    allow_inf_nan=False,
)

_SELECT_COLUMNS: str = (
    "candidate_id, job_id, field_id, ra_deg, dec_deg, bjd_tdb, "
    "mag, snr, cnn_score, model_version, "
    "astrometric_uncertainty_arcsec, astrometric_residual_arcsec"
)

_CUTOUT_SELECT_COLUMNS: str = (
    "candidate_id, epoch_index, cutout_size, bjd_tdb, mag, mag_err, "
    "target_x_px, target_y_px, science_json, reference_json, difference_json, "
    "survey_id, filter_band, pixel_scale_arcsec, fov_arcsec"
)


def _validate_square_matrix(
    name: str,
    matrix: list[list[float]],
    expected_size: int,
) -> None:
    """Validate that a nested pixel grid has shape (expected_size, expected_size)."""
    if len(matrix) != expected_size or any(len(row) != expected_size for row in matrix):
        msg = f"{name} must be a {expected_size}x{expected_size} 2D pixel matrix."
        raise ValueError(msg)


class CandidateListQuery(BaseModel):
    """Validated filter, ordering, and pagination parameters for candidate queries."""

    model_config = _REPO_MODEL_CONFIG

    min_cnn_score: float | None = Field(default=0.0, ge=0.0, le=1.0)
    field_id: str | None = Field(default=None, min_length=1)
    limit: int = Field(default=100, ge=1, le=10000)
    offset: int = Field(default=0, ge=0)


class CandidateDetectionSummary(BaseModel):
    """Validated aggregate statistics across persisted candidate detections."""

    model_config = _REPO_MODEL_CONFIG

    total_candidates: int = Field(..., ge=0)
    high_confidence_count: int = Field(..., ge=0)
    distinct_fields: int = Field(..., ge=0)
    mean_snr: float | None = Field(default=None)
    brightest_mag: float | None = Field(default=None)


class CandidateCutoutStampRecord(BaseModel):
    """Persisted single-epoch odd-sized 3-channel stamp record for a candidate."""

    model_config = _REPO_MODEL_CONFIG

    candidate_id: str = Field(..., min_length=1)
    epoch_index: int = Field(..., ge=0)
    cutout_size: CutoutStampSize
    bjd_tdb: float
    mag: float
    mag_err: float = Field(..., ge=0.0)
    target_x_px: float = Field(..., ge=0.0)
    target_y_px: float = Field(..., ge=0.0)
    science_pixels: list[list[float]]
    reference_pixels: list[list[float]] | None = Field(default=None)
    difference_pixels: list[list[float]] | None = Field(default=None)
    survey_id: str = Field(default="ZTF")
    filter_band: str = Field(default="r")
    pixel_scale_arcsec: float = Field(default=1.012, gt=0.0)
    fov_arcsec: float | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_pixel_grids(self) -> CandidateCutoutStampRecord:
        _validate_square_matrix("science_pixels", self.science_pixels, self.cutout_size)
        if self.reference_pixels is not None:
            _validate_square_matrix(
                "reference_pixels",
                self.reference_pixels,
                self.cutout_size,
            )
        if self.difference_pixels is not None:
            _validate_square_matrix(
                "difference_pixels",
                self.difference_pixels,
                self.cutout_size,
            )
        return self

    @classmethod
    def from_triplet(
        cls,
        *,
        candidate_id: str,
        epoch_index: int,
        bjd_tdb: float,
        mag: float,
        mag_err: float,
        target_x_px: float,
        target_y_px: float,
        triplet: ArrayLike,
        survey_id: str = "ZTF",
        filter_band: str = "r",
        pixel_scale_arcsec: float = 1.012,
        fov_arcsec: float | None = None,
        decimals: int = 4,
    ) -> CandidateCutoutStampRecord:
        """Build a validated stamp record from a (3, S, S) tensor or array triplet."""
        raw_obj: Any = triplet
        if hasattr(raw_obj, "detach") and hasattr(raw_obj, "cpu"):
            arr = np.asarray(raw_obj.detach().cpu().numpy(), dtype=np.float64)
        else:
            arr = np.asarray(raw_obj, dtype=np.float64)

        if arr.ndim != 3 or arr.shape[0] != 3 or arr.shape[1] != arr.shape[2]:
            msg = f"Expected (3, S, S) cutout triplet, got shape {arr.shape}."
            raise ValueError(msg)
        size = int(arr.shape[1])
        if size not in (31, 51):
            msg = f"Triplet stamp size must be 31 or 51, got {size}."
            raise ValueError(msg)
        if not bool(np.all(np.isfinite(arr))):
            msg = "Triplet contains non-finite (NaN or Inf) pixel values."
            raise ValueError(msg)

        rounded = np.round(arr, decimals=decimals)
        stamp_size_lit: CutoutStampSize = 31 if size == 31 else 51
        return cls(
            candidate_id=candidate_id,
            epoch_index=epoch_index,
            cutout_size=stamp_size_lit,
            bjd_tdb=bjd_tdb,
            mag=mag,
            mag_err=mag_err,
            target_x_px=target_x_px,
            target_y_px=target_y_px,
            science_pixels=rounded[0].tolist(),
            reference_pixels=rounded[1].tolist(),
            difference_pixels=rounded[2].tolist(),
            survey_id=survey_id,
            filter_band=filter_band,
            pixel_scale_arcsec=pixel_scale_arcsec,
            fov_arcsec=fov_arcsec,
        )


def _row_to_candidate(row: tuple[Any, ...]) -> CandidateDetection:
    """Deserialize a DuckDB result tuple into the canonical CandidateDetection schema."""
    return CandidateDetection(
        candidate_id=str(row[0]),
        job_id=str(row[1]) if row[1] is not None else None,
        field_id=str(row[2]),
        ra_deg=float(row[3]),
        dec_deg=float(row[4]),
        bjd_tdb=float(row[5]),
        mag=float(row[6]) if row[6] is not None else None,
        snr=float(row[7]) if row[7] is not None else None,
        cnn_score=float(row[8]) if row[8] is not None else None,
        model_version=str(row[9]) if row[9] is not None else None,
        astrometric_uncertainty_arcsec=float(row[10]),
        astrometric_residual_arcsec=float(row[11]) if row[11] is not None else None,
    )


class CandidateDetectionRepository:
    """Repository encapsulating all DuckDB persistence for CandidateDetection entities."""

    def __init__(
        self,
        db: CatalogDatabase,
        *,
        table_name: str = "candidate_detections",
    ) -> None:
        self._db: CatalogDatabase = db
        self._table: str = _validate_identifier(table_name)

    @property
    def table_name(self) -> str:
        """Return the validated SQL table name."""
        return self._table

    def ensure_table(self) -> None:
        """Create the candidate detections table if it does not already exist.

        Optional fields on ``CandidateDetection`` (``mag``, ``snr``, ``cnn_score``,
        ``model_version``, and ``astrometric_residual_arcsec``) are nullable in SQL so
        unclassified or unlinked single-epoch detections can be persisted cleanly.
        """
        conn = self._db._require_connection()  # noqa: SLF001
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                candidate_id VARCHAR PRIMARY KEY,
                job_id VARCHAR,
                field_id VARCHAR NOT NULL,
                ra_deg DOUBLE NOT NULL,
                dec_deg DOUBLE NOT NULL,
                bjd_tdb DOUBLE NOT NULL,
                mag DOUBLE,
                snr DOUBLE,
                cnn_score DOUBLE,
                model_version VARCHAR,
                astrometric_uncertainty_arcsec DOUBLE NOT NULL,
                astrometric_residual_arcsec DOUBLE
            )
            """
        )
        conn.execute(f"ALTER TABLE {self._table} ADD COLUMN IF NOT EXISTS job_id VARCHAR")

    def insert_many(self, candidates: Sequence[CandidateDetection]) -> int:
        """Batch-insert validated CandidateDetection records via bound parameters."""
        if not candidates:
            return 0
        rows: list[tuple[Any, ...]] = [
            (
                cand.candidate_id,
                cand.job_id,
                cand.field_id,
                cand.ra_deg,
                cand.dec_deg,
                cand.bjd_tdb,
                cand.mag,
                cand.snr,
                cand.cnn_score,
                cand.model_version,
                cand.astrometric_uncertainty_arcsec,
                cand.astrometric_residual_arcsec,
            )
            for cand in candidates
        ]
        conn = self._db._require_connection()  # noqa: SLF001
        conn.executemany(
            f"INSERT INTO {self._table} ({_SELECT_COLUMNS}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        return len(rows)

    def get_by_id(self, candidate_id: str) -> CandidateDetection | None:
        """Fetch a single CandidateDetection by primary key, or None if absent."""
        conn = self._db._require_connection()  # noqa: SLF001
        row = conn.execute(
            f"SELECT {_SELECT_COLUMNS} FROM {self._table} WHERE candidate_id = ?",
            [candidate_id],
        ).fetchone()
        if row is None:
            return None
        return _row_to_candidate(row)

    def list_candidates(self, query: CandidateListQuery) -> list[CandidateDetection]:
        """Return filtered, ordered, and paginated CandidateDetection records."""
        clauses: list[str] = []
        bound_params: list[Any] = []

        if query.min_cnn_score is not None:
            clauses.append("cnn_score >= ?")
            bound_params.append(query.min_cnn_score)

        if query.field_id is not None:
            clauses.append("field_id = ?")
            bound_params.append(query.field_id)

        where_clause = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        sql = (
            f"SELECT {_SELECT_COLUMNS} "
            f"FROM {self._table} "
            f"{where_clause}"
            "ORDER BY cnn_score DESC NULLS LAST, snr DESC NULLS LAST, candidate_id ASC "
            "LIMIT ? OFFSET ?"
        )
        bound_params.extend([query.limit, query.offset])

        conn = self._db._require_connection()  # noqa: SLF001
        rows = conn.execute(sql, bound_params).fetchall()
        return [_row_to_candidate(row) for row in rows]

    def summarize(self, *, min_cnn_score: float = 0.8) -> CandidateDetectionSummary:
        """Compute aggregate candidate detection metrics in DuckDB."""
        if not math.isfinite(min_cnn_score) or min_cnn_score < 0.0 or min_cnn_score > 1.0:
            msg = f"min_cnn_score must be in [0.0, 1.0], got {min_cnn_score}."
            raise ValueError(msg)

        sql = f"""
            SELECT
                COUNT(*) AS total_candidates,
                COALESCE(SUM(CASE WHEN cnn_score >= ? THEN 1 ELSE 0 END), 0) AS high_conf,
                COUNT(DISTINCT field_id) AS distinct_fields,
                AVG(snr) AS mean_snr,
                MIN(mag) AS brightest_mag
            FROM {self._table}
        """
        conn = self._db._require_connection()  # noqa: SLF001
        row = conn.execute(sql, [min_cnn_score]).fetchone()
        if row is None:
            return CandidateDetectionSummary(
                total_candidates=0,
                high_confidence_count=0,
                distinct_fields=0,
                mean_snr=None,
                brightest_mag=None,
            )

        total = int(row[0])
        high_conf = int(row[1])
        distinct_fields = int(row[2])
        raw_mean_snr = float(row[3]) if row[3] is not None else None
        raw_brightest_mag = float(row[4]) if row[4] is not None else None

        return CandidateDetectionSummary(
            total_candidates=total,
            high_confidence_count=high_conf,
            distinct_fields=distinct_fields,
            mean_snr=(
                raw_mean_snr
                if raw_mean_snr is not None and math.isfinite(raw_mean_snr)
                else None
            ),
            brightest_mag=(
                raw_brightest_mag
                if raw_brightest_mag is not None and math.isfinite(raw_brightest_mag)
                else None
            ),
        )


class CandidateCutoutRepository:
    """Repository encapsulating DuckDB persistence for extracted multi-epoch stamps."""

    def __init__(
        self,
        db: CatalogDatabase,
        *,
        table_name: str = "candidate_cutout_stamps",
    ) -> None:
        self._db: CatalogDatabase = db
        self._table: str = _validate_identifier(table_name)

    @property
    def table_name(self) -> str:
        """Return the validated SQL table name."""
        return self._table

    def ensure_table(self) -> None:
        """Create the multi-epoch candidate cutout stamps table if absent."""
        conn = self._db._require_connection()  # noqa: SLF001
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                candidate_id VARCHAR NOT NULL,
                epoch_index INTEGER NOT NULL,
                cutout_size INTEGER NOT NULL,
                bjd_tdb DOUBLE NOT NULL,
                mag DOUBLE NOT NULL,
                mag_err DOUBLE NOT NULL,
                target_x_px DOUBLE NOT NULL,
                target_y_px DOUBLE NOT NULL,
                science_json VARCHAR NOT NULL,
                reference_json VARCHAR,
                difference_json VARCHAR,
                survey_id VARCHAR DEFAULT 'ZTF',
                filter_band VARCHAR DEFAULT 'r',
                pixel_scale_arcsec DOUBLE DEFAULT 1.012,
                fov_arcsec DOUBLE,
                PRIMARY KEY (candidate_id, epoch_index)
            )
            """
        )
        conn.execute(f"ALTER TABLE {self._table} ADD COLUMN IF NOT EXISTS survey_id VARCHAR DEFAULT 'ZTF'")
        conn.execute(f"ALTER TABLE {self._table} ADD COLUMN IF NOT EXISTS filter_band VARCHAR DEFAULT 'r'")
        conn.execute(f"ALTER TABLE {self._table} ADD COLUMN IF NOT EXISTS pixel_scale_arcsec DOUBLE DEFAULT 1.012")
        conn.execute(f"ALTER TABLE {self._table} ADD COLUMN IF NOT EXISTS fov_arcsec DOUBLE")

    def upsert_many(self, records: Sequence[CandidateCutoutStampRecord]) -> int:
        """Batch-upsert validated CandidateCutoutStampRecord entries via bound parameters."""
        if not records:
            return 0
        rows: list[tuple[Any, ...]] = [
            (
                rec.candidate_id,
                rec.epoch_index,
                rec.cutout_size,
                rec.bjd_tdb,
                rec.mag,
                rec.mag_err,
                rec.target_x_px,
                rec.target_y_px,
                json.dumps(rec.science_pixels, allow_nan=False),
                (
                    json.dumps(rec.reference_pixels, allow_nan=False)
                    if rec.reference_pixels is not None
                    else None
                ),
                (
                    json.dumps(rec.difference_pixels, allow_nan=False)
                    if rec.difference_pixels is not None
                    else None
                ),
                rec.survey_id,
                rec.filter_band,
                rec.pixel_scale_arcsec,
                rec.fov_arcsec,
            )
            for rec in records
        ]
        conn = self._db._require_connection()  # noqa: SLF001
        conn.executemany(
            f"INSERT OR REPLACE INTO {self._table} ({_CUTOUT_SELECT_COLUMNS}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        return len(rows)

    def get_sequence(self, candidate_id: str) -> list[CandidateCutoutStampRecord]:
        """Return ordered multi-epoch cutout stamps for candidate_id, or [] if absent."""
        conn = self._db._require_connection()  # noqa: SLF001
        table_exists = conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ? LIMIT 1",
            [self._table],
        ).fetchone()
        if table_exists is None:
            return []

        rows = conn.execute(
            f"SELECT {_CUTOUT_SELECT_COLUMNS} FROM {self._table} "
            "WHERE candidate_id = ? ORDER BY epoch_index ASC",
            [candidate_id],
        ).fetchall()

        results: list[CandidateCutoutStampRecord] = []
        for row in rows:
            size_val = int(row[2])
            if size_val not in (31, 51):
                msg = f"Invalid persisted cutout_size={size_val} for '{candidate_id}'."
                raise ValueError(msg)
            stamp_size_lit: CutoutStampSize = 31 if size_val == 31 else 51
            results.append(
                CandidateCutoutStampRecord(
                    candidate_id=str(row[0]),
                    epoch_index=int(row[1]),
                    cutout_size=stamp_size_lit,
                    bjd_tdb=float(row[3]),
                    mag=float(row[4]),
                    mag_err=float(row[5]),
                    target_x_px=float(row[6]),
                    target_y_px=float(row[7]),
                    science_pixels=json.loads(str(row[8])),
                    reference_pixels=(
                        json.loads(str(row[9])) if row[9] is not None else None
                    ),
                    difference_pixels=(
                        json.loads(str(row[10])) if row[10] is not None else None
                    ),
                    survey_id=str(row[11]) if len(row) > 11 and row[11] is not None else "ZTF",
                    filter_band=str(row[12]) if len(row) > 12 and row[12] is not None else "r",
                    pixel_scale_arcsec=float(row[13]) if len(row) > 13 and row[13] is not None else 1.012,
                    fov_arcsec=float(row[14]) if len(row) > 14 and row[14] is not None else None,
                )
            )
        return results

    @classmethod
    def generate_synthetic_sequence(
        cls,
        candidate: CandidateDetection,
        stamp_size: CutoutStampSize = 31,
        epochs_count: int = 4,
    ) -> list[CandidateCutoutStampRecord]:
        """Generate physically plausible 3-channel multi-epoch cutouts for preview/testing."""
        records: list[CandidateCutoutStampRecord] = []
        base_bjd = candidate.bjd_tdb
        vx = 1.8
        vy = -1.2
        snr = candidate.snr if candidate.snr is not None else 10.0
        peak_flux = max(5.0, snr * 2.5)

        for epoch_idx in range(epochs_count):
            t_offset = epoch_idx - 1.5
            tx = 15.5 + t_offset * vx
            ty = 15.5 + t_offset * vy
            epoch_bjd = base_bjd + (epoch_idx * 0.0104)

            sci: list[list[float]] = []
            ref: list[list[float]] = []
            diff: list[list[float]] = []

            for y in range(stamp_size):
                row_sci: list[float] = []
                row_ref: list[float] = []
                row_diff: list[float] = []
                for x in range(stamp_size):
                    noise = 1.0 + 0.15 * math.sin(x * 12.9898 + y * 78.233 + epoch_idx * 37.719)
                    d_star = math.sqrt((x - 7.0) ** 2 + (y - 7.0) ** 2)
                    star_flux = 30.0 * math.exp(-0.5 * (d_star / 1.5) ** 2)
                    d_cand = math.sqrt((x - tx) ** 2 + (y - ty) ** 2)
                    cand_flux = peak_flux * math.exp(-0.5 * (d_cand / 1.4) ** 2)

                    ref_val = round(noise + star_flux, 4)
                    sci_val = round(noise + star_flux + cand_flux, 4)
                    diff_val = round(cand_flux + 0.05 * math.cos(x * 5.1 + y * 3.3), 4)

                    row_sci.append(sci_val)
                    row_ref.append(ref_val)
                    row_diff.append(diff_val)

                sci.append(row_sci)
                ref.append(row_ref)
                diff.append(row_diff)

            survey = getattr(candidate, "survey_id", "ZTF") or "ZTF"
            filter_band = getattr(candidate, "filter_band", "r") or "r"
            scales = {"ZTF": 1.012, "Pan-STARRS": 0.25, "Rubin": 0.20, "JWST": 0.031}
            pixel_scale = scales.get(survey, 1.012)
            fov = float(stamp_size * pixel_scale)

            records.append(
                CandidateCutoutStampRecord(
                    candidate_id=candidate.candidate_id,
                    epoch_index=epoch_idx,
                    cutout_size=stamp_size,
                    bjd_tdb=epoch_bjd,
                    mag=candidate.mag if candidate.mag is not None else 20.5,
                    mag_err=0.08,
                    target_x_px=round(tx, 2),
                    target_y_px=round(ty, 2),
                    science_pixels=sci,
                    reference_pixels=ref,
                    difference_pixels=diff,
                    survey_id=survey,
                    filter_band=filter_band,
                    pixel_scale_arcsec=pixel_scale,
                    fov_arcsec=fov,
                )
            )
        return records

    def seed_synthetic_cutouts_if_empty(
        self, candidates: Sequence[CandidateDetection]
    ) -> int:
        """Seed synthetic cutout sequences for candidates if no cutouts currently exist."""
        if not candidates:
            return 0
        first_cand = candidates[0]
        existing = self.get_sequence(first_cand.candidate_id)
        if existing:
            return 0
        all_stamps: list[CandidateCutoutStampRecord] = []
        for cand in candidates:
            all_stamps.extend(self.generate_synthetic_sequence(cand))
        return self.upsert_many(all_stamps)
