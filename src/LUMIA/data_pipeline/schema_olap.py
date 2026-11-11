"""DuckDB analytical table definitions and Hive-partitioned Parquet OLAP schema (Module 2).

Implements Section 3 of ``docs/database_schema_blueprint.md`` in strict compliance with
ADR-0002 and ADR-0003:
- In-memory (``:memory:``) DuckDB execution during active pipeline runs to eliminate SSD
  B-tree/WAL churn.
- Immutable Zstandard (``ZSTD``)-compressed, row-group-sorted, Hive-partitioned Apache Parquet
  persistence for wide-field stellar catalogs, single-epoch candidate detections, odd-sized
  3-channel cutout stamps, velocity-space states, and linked tracklets.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from LUMIA.data_pipeline.database import CatalogDatabase

if TYPE_CHECKING:
    from datetime import date

    import duckdb

__all__ = [
    "OLAP_TABLE_DEFINITIONS",
    "OLAPColumnSpec",
    "OLAPDatasetName",
    "OLAPJobDatasetName",
    "OLAPSchemaConfig",
    "OLAPTableDefinition",
    "TRACKLET_TRIAGE_SUMMARY_VIEW_DDL",
    "export_job_table_to_parquet",
    "export_stellar_catalog_zone_to_parquet",
    "initialize_olap_schema",
    "initialize_parquet_directory_layout",
    "job_dataset_partition_path",
    "stellar_catalog_partition_path",
]

OLAPDatasetName = Literal[
    "stellar_catalog",
    "candidate_detections",
    "candidate_cutout_stamps",
    "space_velocity_states",
    "tracklets",
    "tracklet_members",
]

OLAPJobDatasetName = Literal[
    "candidate_detections",
    "candidate_cutout_stamps",
    "space_velocity_states",
    "tracklets",
    "tracklet_members",
]

ParquetCompressionCodec = Literal["ZSTD", "SNAPPY", "UNCOMPRESSED"]

_OLAP_MODEL_CONFIG: ConfigDict = ConfigDict(
    strict=True,
    extra="forbid",
    frozen=True,
    allow_inf_nan=False,
)

_SAFE_ASCII_ID_PATTERN: re.Pattern[str] = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def _validate_safe_linkage_id(label: str, value: str) -> str:
    """Validate immutable cross-store identifiers against path traversal and SQL injection."""
    if not _SAFE_ASCII_ID_PATTERN.match(value) or value in {".", ".."} or ".." in value:
        msg = (
            f"Invalid {label} identifier {value!r}: must match "
            r"^[A-Za-z0-9_.-]{1,128}$ without '..' traversal segments."
        )
        raise ValueError(msg)
    return value


def _escape_duckdb_path_literal(path: Path) -> str:
    """Format a filesystem path as a safe single-quoted POSIX literal for DuckDB COPY."""
    posix_str = path.resolve().as_posix()
    escaped = posix_str.replace("'", "''")
    return f"'{escaped}'"


def _unwrap_duckdb_connection(
    conn_or_db: duckdb.DuckDBPyConnection | CatalogDatabase,
) -> duckdb.DuckDBPyConnection:
    """Resolve either a raw DuckDB connection or a CatalogDatabase wrapper."""
    if isinstance(conn_or_db, CatalogDatabase):
        return conn_or_db._require_connection()  # noqa: SLF001
    return conn_or_db


class OLAPColumnSpec(BaseModel):
    """Immutable specification of a single DuckDB / Apache Parquet column."""

    model_config = _OLAP_MODEL_CONFIG

    name: str = Field(..., min_length=1, pattern=r"^[a-z_][a-z0-9_]*$")
    duckdb_type: str = Field(..., min_length=1)
    nullable: bool = Field(default=False)
    check_sql: str | None = Field(default=None)


class OLAPTableDefinition(BaseModel):
    """Immutable DDL, Hive partitioning, and row-group sorting contract for an OLAP table."""

    model_config = _OLAP_MODEL_CONFIG

    table_name: OLAPDatasetName
    columns: tuple[OLAPColumnSpec, ...]
    primary_key: tuple[str, ...] = Field(default=())
    hive_partition_keys: tuple[str, ...] = Field(default=())
    row_group_order_by_sql: str = Field(..., min_length=1)
    ddl_sql: str = Field(..., min_length=1)


class OLAPSchemaConfig(BaseModel):
    """Validated filesystem and Parquet compression settings for the OLAP storage plane."""

    model_config = _OLAP_MODEL_CONFIG

    parquet_root_dir: Path = Field(
        default=Path("data/olap"),
        description="Root directory containing Hive-partitioned Apache Parquet datasets.",
    )
    compression: ParquetCompressionCodec = Field(
        default="ZSTD",
        description="Sequential Parquet block compression codec (ZSTD per blueprint).",
    )
    row_group_size: int = Field(
        default=100_000,
        ge=1024,
        le=10_000_000,
        description="Target Parquet row-group cardinality for zone-map min/max pruning.",
    )


# ---------------------------------------------------------------------------
# Canonical DuckDB DDL & Metadata Definitions (Blueprint Section 3.1)
# ---------------------------------------------------------------------------

_STELLAR_CATALOG_DDL: str = """
CREATE TABLE IF NOT EXISTS stellar_catalog (
    dec_zone        SMALLINT NOT NULL CHECK (dec_zone BETWEEN -90 AND 90),
    healpix_ipix    BIGINT   NOT NULL CHECK (healpix_ipix >= 0),
    source_id       BIGINT   PRIMARY KEY CHECK (source_id >= 0),
    ra_deg          DOUBLE   NOT NULL CHECK (ra_deg >= 0.0 AND ra_deg < 360.0),
    dec_deg         DOUBLE   NOT NULL CHECK (dec_deg BETWEEN -90.0 AND 90.0),
    mag_g           DOUBLE   NOT NULL,
    mag_g_err       FLOAT    NULL CHECK (mag_g_err IS NULL OR mag_g_err >= 0.0),
    mag_bp          FLOAT    NULL,
    mag_rp          FLOAT    NULL,
    pm_ra_mas_yr    FLOAT    NULL,
    pm_dec_mas_yr   FLOAT    NULL,
    ref_epoch_jy    FLOAT    NULL
);
""".strip()

_CANDIDATE_DETECTIONS_DDL: str = """
CREATE TABLE IF NOT EXISTS candidate_detections (
    candidate_id                    VARCHAR PRIMARY KEY,
    job_id                          VARCHAR NOT NULL,
    observation_night               DATE    NOT NULL,
    field_id                        VARCHAR NOT NULL,
    ra_deg                          DOUBLE  NOT NULL CHECK (ra_deg >= 0.0 AND ra_deg < 360.0),
    dec_deg                         DOUBLE  NOT NULL CHECK (dec_deg BETWEEN -90.0 AND 90.0),
    bjd_tdb                         DOUBLE  NOT NULL,
    x_centroid_px                   DOUBLE  NULL,
    y_centroid_px                   DOUBLE  NULL,
    net_flux_adu                    DOUBLE  NULL,
    flux_err_adu                    DOUBLE  NULL
                                    CHECK (flux_err_adu IS NULL OR flux_err_adu >= 0.0),
    mag                             DOUBLE  NULL,
    mag_err                         DOUBLE  NULL CHECK (mag_err IS NULL OR mag_err >= 0.0),
    snr                             DOUBLE  NULL CHECK (snr IS NULL OR snr >= 0.0),
    cnn_score                       DOUBLE  NULL
                                    CHECK (
                                        cnn_score IS NULL
                                        OR (cnn_score >= 0.0 AND cnn_score <= 1.0)
                                    ),
    model_version                   VARCHAR NULL,
    astrometric_uncertainty_arcsec  DOUBLE  NOT NULL CHECK (astrometric_uncertainty_arcsec >= 0.0),
    astrometric_residual_arcsec     DOUBLE  NULL
                                    CHECK (
                                        astrometric_residual_arcsec IS NULL
                                        OR astrometric_residual_arcsec >= 0.0
                                    ),
    tracklet_id                     VARCHAR NULL
);
""".strip()

_CANDIDATE_CUTOUT_STAMPS_DDL: str = """
CREATE TABLE IF NOT EXISTS candidate_cutout_stamps (
    candidate_id      VARCHAR   NOT NULL,
    epoch_index       INTEGER   NOT NULL CHECK (epoch_index >= 0),
    observation_night DATE      NOT NULL,
    field_id          VARCHAR   NOT NULL,
    cutout_size       SMALLINT  NOT NULL CHECK (cutout_size IN (31, 51)),
    bjd_tdb           DOUBLE    NOT NULL,
    mag               DOUBLE    NOT NULL,
    mag_err           DOUBLE    NOT NULL CHECK (mag_err >= 0.0),
    target_x_px       DOUBLE    NOT NULL CHECK (target_x_px >= 0.0),
    target_y_px       DOUBLE    NOT NULL CHECK (target_y_px >= 0.0),
    science_pixels    FLOAT[][] NOT NULL,
    reference_pixels  FLOAT[][] NULL,
    difference_pixels FLOAT[][] NULL,
    PRIMARY KEY (candidate_id, epoch_index)
);
""".strip()

_SPACE_VELOCITY_STATES_DDL: str = """
CREATE TABLE IF NOT EXISTS space_velocity_states (
    job_id                     VARCHAR NOT NULL,
    observation_night          DATE    NOT NULL,
    field_id                   VARCHAR NOT NULL,
    source_candidate_id        VARCHAR NOT NULL,
    target_candidate_id        VARCHAR NOT NULL,
    reference_bjd_tdb          DOUBLE  NOT NULL,
    baseline_hours             DOUBLE  NOT NULL CHECK (baseline_hours > 0.0),
    ref_xi_arcsec              DOUBLE  NOT NULL,
    ref_eta_arcsec             DOUBLE  NOT NULL,
    v_ra_cosdec_arcsec_per_hr  DOUBLE  NOT NULL,
    v_dec_arcsec_per_hr        DOUBLE  NOT NULL
);
""".strip()

_TRACKLETS_DDL: str = """
CREATE TABLE IF NOT EXISTS tracklets (
    tracklet_id                    VARCHAR PRIMARY KEY,
    job_id                         VARCHAR NOT NULL,
    observation_night              DATE    NOT NULL,
    field_id                       VARCHAR NOT NULL,
    n_epochs                       INTEGER NOT NULL CHECK (n_epochs >= 3),
    reference_bjd_tdb              DOUBLE  NOT NULL,
    v_ra_cosdec_arcsec_per_hr      DOUBLE  NOT NULL,
    v_dec_arcsec_per_hr            DOUBLE  NOT NULL,
    angular_speed_arcsec_per_hr    DOUBLE  NOT NULL CHECK (angular_speed_arcsec_per_hr >= 0.0),
    angular_accel_arcsec_per_day2  DOUBLE  NOT NULL CHECK (angular_accel_arcsec_per_day2 >= 0.0),
    rms_residual_arcsec            DOUBLE  NOT NULL CHECK (rms_residual_arcsec >= 0.0),
    implied_transverse_speed_km_s  DOUBLE  NOT NULL CHECK (implied_transverse_speed_km_s >= 0.0)
);
""".strip()

_TRACKLET_MEMBERS_DDL: str = """
CREATE TABLE IF NOT EXISTS tracklet_members (
    tracklet_id         VARCHAR  NOT NULL,
    epoch_order         SMALLINT NOT NULL CHECK (epoch_order >= 0),
    candidate_id        VARCHAR  NOT NULL,
    residual_arcsec     DOUBLE   NOT NULL CHECK (residual_arcsec >= 0.0),
    PRIMARY KEY (tracklet_id, epoch_order)
);
""".strip()

TRACKLET_TRIAGE_SUMMARY_VIEW_DDL: str = """
CREATE OR REPLACE VIEW v_tracklet_triage_summary AS
SELECT
    t.tracklet_id,
    t.job_id,
    t.observation_night,
    t.field_id,
    t.n_epochs,
    t.reference_bjd_tdb,
    t.angular_speed_arcsec_per_hr,
    t.angular_accel_arcsec_per_day2,
    t.rms_residual_arcsec,
    t.implied_transverse_speed_km_s,
    AVG(c.cnn_score) AS mean_cnn_score,
    MIN(c.cnn_score) AS min_cnn_score,
    MIN(c.mag)       AS brightest_mag,
    AVG(c.snr)       AS mean_snr
FROM tracklets AS t
JOIN tracklet_members AS tm ON tm.tracklet_id = t.tracklet_id
JOIN candidate_detections AS c ON c.candidate_id = tm.candidate_id
GROUP BY
    t.tracklet_id,
    t.job_id,
    t.observation_night,
    t.field_id,
    t.n_epochs,
    t.reference_bjd_tdb,
    t.angular_speed_arcsec_per_hr,
    t.angular_accel_arcsec_per_day2,
    t.rms_residual_arcsec,
    t.implied_transverse_speed_km_s;
""".strip()

OLAP_TABLE_DEFINITIONS: dict[OLAPDatasetName, OLAPTableDefinition] = {
    "stellar_catalog": OLAPTableDefinition(
        table_name="stellar_catalog",
        columns=(
            OLAPColumnSpec(
                name="dec_zone",
                duckdb_type="SMALLINT",
                nullable=False,
                check_sql="dec_zone BETWEEN -90 AND 90",
            ),
            OLAPColumnSpec(
                name="healpix_ipix",
                duckdb_type="BIGINT",
                nullable=False,
                check_sql="healpix_ipix >= 0",
            ),
            OLAPColumnSpec(
                name="source_id",
                duckdb_type="BIGINT",
                nullable=False,
                check_sql="source_id >= 0",
            ),
            OLAPColumnSpec(
                name="ra_deg",
                duckdb_type="DOUBLE",
                nullable=False,
                check_sql="ra_deg >= 0.0 AND ra_deg < 360.0",
            ),
            OLAPColumnSpec(
                name="dec_deg",
                duckdb_type="DOUBLE",
                nullable=False,
                check_sql="dec_deg BETWEEN -90.0 AND 90.0",
            ),
            OLAPColumnSpec(name="mag_g", duckdb_type="DOUBLE", nullable=False),
            OLAPColumnSpec(
                name="mag_g_err",
                duckdb_type="FLOAT",
                nullable=True,
                check_sql="mag_g_err >= 0.0",
            ),
            OLAPColumnSpec(name="mag_bp", duckdb_type="FLOAT", nullable=True),
            OLAPColumnSpec(name="mag_rp", duckdb_type="FLOAT", nullable=True),
            OLAPColumnSpec(name="pm_ra_mas_yr", duckdb_type="FLOAT", nullable=True),
            OLAPColumnSpec(name="pm_dec_mas_yr", duckdb_type="FLOAT", nullable=True),
            OLAPColumnSpec(name="ref_epoch_jy", duckdb_type="FLOAT", nullable=True),
        ),
        primary_key=("source_id",),
        hive_partition_keys=("dec_zone",),
        row_group_order_by_sql="healpix_ipix ASC, ra_deg ASC, dec_deg ASC, mag_g ASC",
        ddl_sql=_STELLAR_CATALOG_DDL,
    ),
    "candidate_detections": OLAPTableDefinition(
        table_name="candidate_detections",
        columns=(
            OLAPColumnSpec(name="candidate_id", duckdb_type="VARCHAR", nullable=False),
            OLAPColumnSpec(name="job_id", duckdb_type="VARCHAR", nullable=False),
            OLAPColumnSpec(name="observation_night", duckdb_type="DATE", nullable=False),
            OLAPColumnSpec(name="field_id", duckdb_type="VARCHAR", nullable=False),
            OLAPColumnSpec(name="ra_deg", duckdb_type="DOUBLE", nullable=False),
            OLAPColumnSpec(name="dec_deg", duckdb_type="DOUBLE", nullable=False),
            OLAPColumnSpec(name="bjd_tdb", duckdb_type="DOUBLE", nullable=False),
            OLAPColumnSpec(name="x_centroid_px", duckdb_type="DOUBLE", nullable=True),
            OLAPColumnSpec(name="y_centroid_px", duckdb_type="DOUBLE", nullable=True),
            OLAPColumnSpec(name="net_flux_adu", duckdb_type="DOUBLE", nullable=True),
            OLAPColumnSpec(name="flux_err_adu", duckdb_type="DOUBLE", nullable=True),
            OLAPColumnSpec(name="mag", duckdb_type="DOUBLE", nullable=True),
            OLAPColumnSpec(name="mag_err", duckdb_type="DOUBLE", nullable=True),
            OLAPColumnSpec(name="snr", duckdb_type="DOUBLE", nullable=True),
            OLAPColumnSpec(name="cnn_score", duckdb_type="DOUBLE", nullable=True),
            OLAPColumnSpec(name="model_version", duckdb_type="VARCHAR", nullable=True),
            OLAPColumnSpec(
                name="astrometric_uncertainty_arcsec",
                duckdb_type="DOUBLE",
                nullable=False,
            ),
            OLAPColumnSpec(
                name="astrometric_residual_arcsec",
                duckdb_type="DOUBLE",
                nullable=True,
            ),
            OLAPColumnSpec(name="tracklet_id", duckdb_type="VARCHAR", nullable=True),
        ),
        primary_key=("candidate_id",),
        hive_partition_keys=("observation_night", "field_id"),
        row_group_order_by_sql="cnn_score DESC NULLS LAST, snr DESC NULLS LAST, candidate_id ASC",
        ddl_sql=_CANDIDATE_DETECTIONS_DDL,
    ),
    "candidate_cutout_stamps": OLAPTableDefinition(
        table_name="candidate_cutout_stamps",
        columns=(
            OLAPColumnSpec(name="candidate_id", duckdb_type="VARCHAR", nullable=False),
            OLAPColumnSpec(name="epoch_index", duckdb_type="INTEGER", nullable=False),
            OLAPColumnSpec(name="observation_night", duckdb_type="DATE", nullable=False),
            OLAPColumnSpec(name="field_id", duckdb_type="VARCHAR", nullable=False),
            OLAPColumnSpec(name="cutout_size", duckdb_type="SMALLINT", nullable=False),
            OLAPColumnSpec(name="bjd_tdb", duckdb_type="DOUBLE", nullable=False),
            OLAPColumnSpec(name="mag", duckdb_type="DOUBLE", nullable=False),
            OLAPColumnSpec(name="mag_err", duckdb_type="DOUBLE", nullable=False),
            OLAPColumnSpec(name="target_x_px", duckdb_type="DOUBLE", nullable=False),
            OLAPColumnSpec(name="target_y_px", duckdb_type="DOUBLE", nullable=False),
            OLAPColumnSpec(
                name="science_pixels",
                duckdb_type="FLOAT[][]",
                nullable=False,
            ),
            OLAPColumnSpec(
                name="reference_pixels",
                duckdb_type="FLOAT[][]",
                nullable=True,
            ),
            OLAPColumnSpec(
                name="difference_pixels",
                duckdb_type="FLOAT[][]",
                nullable=True,
            ),
        ),
        primary_key=("candidate_id", "epoch_index"),
        hive_partition_keys=("observation_night", "field_id"),
        row_group_order_by_sql="candidate_id ASC, epoch_index ASC",
        ddl_sql=_CANDIDATE_CUTOUT_STAMPS_DDL,
    ),
    "space_velocity_states": OLAPTableDefinition(
        table_name="space_velocity_states",
        columns=(
            OLAPColumnSpec(name="job_id", duckdb_type="VARCHAR", nullable=False),
            OLAPColumnSpec(name="observation_night", duckdb_type="DATE", nullable=False),
            OLAPColumnSpec(name="field_id", duckdb_type="VARCHAR", nullable=False),
            OLAPColumnSpec(
                name="source_candidate_id",
                duckdb_type="VARCHAR",
                nullable=False,
            ),
            OLAPColumnSpec(
                name="target_candidate_id",
                duckdb_type="VARCHAR",
                nullable=False,
            ),
            OLAPColumnSpec(name="reference_bjd_tdb", duckdb_type="DOUBLE", nullable=False),
            OLAPColumnSpec(name="baseline_hours", duckdb_type="DOUBLE", nullable=False),
            OLAPColumnSpec(name="ref_xi_arcsec", duckdb_type="DOUBLE", nullable=False),
            OLAPColumnSpec(name="ref_eta_arcsec", duckdb_type="DOUBLE", nullable=False),
            OLAPColumnSpec(
                name="v_ra_cosdec_arcsec_per_hr",
                duckdb_type="DOUBLE",
                nullable=False,
            ),
            OLAPColumnSpec(
                name="v_dec_arcsec_per_hr",
                duckdb_type="DOUBLE",
                nullable=False,
            ),
        ),
        primary_key=(),
        hive_partition_keys=("observation_night", "field_id"),
        row_group_order_by_sql=(
            "reference_bjd_tdb ASC, source_candidate_id ASC, target_candidate_id ASC"
        ),
        ddl_sql=_SPACE_VELOCITY_STATES_DDL,
    ),
    "tracklets": OLAPTableDefinition(
        table_name="tracklets",
        columns=(
            OLAPColumnSpec(name="tracklet_id", duckdb_type="VARCHAR", nullable=False),
            OLAPColumnSpec(name="job_id", duckdb_type="VARCHAR", nullable=False),
            OLAPColumnSpec(name="observation_night", duckdb_type="DATE", nullable=False),
            OLAPColumnSpec(name="field_id", duckdb_type="VARCHAR", nullable=False),
            OLAPColumnSpec(name="n_epochs", duckdb_type="INTEGER", nullable=False),
            OLAPColumnSpec(name="reference_bjd_tdb", duckdb_type="DOUBLE", nullable=False),
            OLAPColumnSpec(
                name="v_ra_cosdec_arcsec_per_hr",
                duckdb_type="DOUBLE",
                nullable=False,
            ),
            OLAPColumnSpec(
                name="v_dec_arcsec_per_hr",
                duckdb_type="DOUBLE",
                nullable=False,
            ),
            OLAPColumnSpec(
                name="angular_speed_arcsec_per_hr",
                duckdb_type="DOUBLE",
                nullable=False,
            ),
            OLAPColumnSpec(
                name="angular_accel_arcsec_per_day2",
                duckdb_type="DOUBLE",
                nullable=False,
            ),
            OLAPColumnSpec(
                name="rms_residual_arcsec",
                duckdb_type="DOUBLE",
                nullable=False,
            ),
            OLAPColumnSpec(
                name="implied_transverse_speed_km_s",
                duckdb_type="DOUBLE",
                nullable=False,
            ),
        ),
        primary_key=("tracklet_id",),
        hive_partition_keys=("observation_night", "field_id"),
        row_group_order_by_sql="rms_residual_arcsec ASC, tracklet_id ASC",
        ddl_sql=_TRACKLETS_DDL,
    ),
    "tracklet_members": OLAPTableDefinition(
        table_name="tracklet_members",
        columns=(
            OLAPColumnSpec(name="tracklet_id", duckdb_type="VARCHAR", nullable=False),
            OLAPColumnSpec(name="epoch_order", duckdb_type="SMALLINT", nullable=False),
            OLAPColumnSpec(name="candidate_id", duckdb_type="VARCHAR", nullable=False),
            OLAPColumnSpec(name="residual_arcsec", duckdb_type="DOUBLE", nullable=False),
        ),
        primary_key=("tracklet_id", "epoch_order"),
        hive_partition_keys=("observation_night", "field_id"),
        row_group_order_by_sql="tracklet_id ASC, epoch_order ASC",
        ddl_sql=_TRACKLET_MEMBERS_DDL,
    ),
}


# ---------------------------------------------------------------------------
# DuckDB Schema Initialization & Hive-Partitioned Parquet Export Helpers
# ---------------------------------------------------------------------------


def initialize_olap_schema(
    conn_or_db: duckdb.DuckDBPyConnection | CatalogDatabase,
) -> None:
    """Create all 6 OLAP tables and ``v_tracklet_triage_summary`` in DuckDB."""
    conn = _unwrap_duckdb_connection(conn_or_db)
    for table_def in OLAP_TABLE_DEFINITIONS.values():
        conn.execute(table_def.ddl_sql)
    conn.execute(TRACKLET_TRIAGE_SUMMARY_VIEW_DDL)


def initialize_parquet_directory_layout(
    config: OLAPSchemaConfig,
) -> dict[OLAPDatasetName, Path]:
    """Create top-level Hive dataset directories under ``config.parquet_root_dir``."""
    root = config.parquet_root_dir
    root.mkdir(parents=True, exist_ok=True)
    dataset_dirs: dict[OLAPDatasetName, Path] = {}
    for dataset_name in OLAP_TABLE_DEFINITIONS:
        target = root / dataset_name
        target.mkdir(parents=True, exist_ok=True)
        dataset_dirs[dataset_name] = target
    return dataset_dirs


def stellar_catalog_partition_path(
    parquet_root_dir: Path,
    *,
    dec_zone: int,
    part_index: int = 0,
) -> Path:
    """Return Hive partition file path ``stellar_catalog/dec_zone=<z>/part-XXXX.parquet``."""
    if isinstance(dec_zone, bool) or dec_zone < -90 or dec_zone > 90:
        msg = f"dec_zone must be an integer in [-90, 90], got {dec_zone!r}."
        raise ValueError(msg)
    if isinstance(part_index, bool) or part_index < 0:
        msg = f"part_index must be a non-negative integer, got {part_index!r}."
        raise ValueError(msg)
    return (
        parquet_root_dir
        / "stellar_catalog"
        / f"dec_zone={dec_zone}"
        / f"part-{part_index:04d}.parquet"
    )


def job_dataset_partition_path(
    parquet_root_dir: Path,
    dataset_name: OLAPDatasetName,
    *,
    observation_night: date,
    field_id: str,
    job_id: str,
) -> Path:
    """Return Hive-partitioned Parquet file path for a completed field/job run."""
    if dataset_name not in OLAP_TABLE_DEFINITIONS or dataset_name == "stellar_catalog":
        msg = f"Unsupported job OLAP dataset {dataset_name!r}."
        raise ValueError(msg)
    safe_field = _validate_safe_linkage_id("field_id", field_id)
    safe_job = _validate_safe_linkage_id("job_id", job_id)
    night_iso = observation_night.isoformat()
    return (
        parquet_root_dir
        / dataset_name
        / f"observation_night={night_iso}"
        / f"field_id={safe_field}"
        / f"job_id={safe_job}.parquet"
    )


def export_stellar_catalog_zone_to_parquet(
    conn_or_db: duckdb.DuckDBPyConnection | CatalogDatabase,
    config: OLAPSchemaConfig,
    *,
    dec_zone: int,
    part_index: int = 0,
) -> Path:
    """Write a ``dec_zone`` slice of ``stellar_catalog`` as a sorted ZSTD Parquet file."""
    conn = _unwrap_duckdb_connection(conn_or_db)
    target_path = stellar_catalog_partition_path(
        config.parquet_root_dir,
        dec_zone=dec_zone,
        part_index=part_index,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    order_clause = OLAP_TABLE_DEFINITIONS["stellar_catalog"].row_group_order_by_sql
    escaped_target = _escape_duckdb_path_literal(target_path)

    copy_sql = (
        "COPY ("
        "SELECT * FROM stellar_catalog WHERE dec_zone = ? "
        f"ORDER BY {order_clause}"
        f") TO {escaped_target} "
        f"(FORMAT PARQUET, COMPRESSION {config.compression}, "
        f"ROW_GROUP_SIZE {config.row_group_size})"
    )
    conn.execute(copy_sql, [dec_zone])
    return target_path


def export_job_table_to_parquet(
    conn_or_db: duckdb.DuckDBPyConnection | CatalogDatabase,
    config: OLAPSchemaConfig,
    *,
    dataset_name: OLAPJobDatasetName,
    observation_night: date,
    field_id: str,
    job_id: str,
) -> Path:
    """Flush an in-memory DuckDB job dataset sequentially to Hive-partitioned Parquet."""
    conn = _unwrap_duckdb_connection(conn_or_db)
    target_path = job_dataset_partition_path(
        config.parquet_root_dir,
        dataset_name,
        observation_night=observation_night,
        field_id=field_id,
        job_id=job_id,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)

    table_def = OLAP_TABLE_DEFINITIONS[dataset_name]
    escaped_target = _escape_duckdb_path_literal(target_path)

    if dataset_name in {"candidate_detections", "space_velocity_states", "tracklets"}:
        select_sql = (
            f"SELECT * FROM {dataset_name} "
            "WHERE observation_night = ? AND field_id = ? AND job_id = ? "
            f"ORDER BY {table_def.row_group_order_by_sql}"
        )
        params: list[object] = [observation_night, field_id, job_id]
    elif dataset_name == "candidate_cutout_stamps":
        select_sql = (
            "SELECT s.* FROM candidate_cutout_stamps AS s "
            "LEFT JOIN candidate_detections AS c ON c.candidate_id = s.candidate_id "
            "WHERE s.observation_night = ? AND s.field_id = ? "
            "AND (c.job_id = ? OR c.job_id IS NULL) "
            "ORDER BY s.candidate_id ASC, s.epoch_index ASC"
        )
        params = [observation_night, field_id, job_id]
    else:
        # tracklet_members joins tracklets to filter by (observation_night, field_id, job_id)
        select_sql = (
            "SELECT tm.* FROM tracklet_members AS tm "
            "JOIN tracklets AS t ON t.tracklet_id = tm.tracklet_id "
            "WHERE t.observation_night = ? AND t.field_id = ? AND t.job_id = ? "
            "ORDER BY tm.tracklet_id ASC, tm.epoch_order ASC"
        )
        params = [observation_night, field_id, job_id]

    copy_sql = (
        f"COPY ({select_sql}) TO {escaped_target} "
        f"(FORMAT PARQUET, COMPRESSION {config.compression}, "
        f"ROW_GROUP_SIZE {config.row_group_size})"
    )
    conn.execute(copy_sql, params)
    return target_path
