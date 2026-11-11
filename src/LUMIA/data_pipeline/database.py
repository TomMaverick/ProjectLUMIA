"""DuckDB analytical storage and parametrized NumPy catalog queries for Module 2."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    import duckdb

_VALID_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(identifier: str) -> str:
    """Ensure a SQL table identifier contains only safe alphanumeric and underscore characters."""
    if not _VALID_SQL_IDENTIFIER.match(identifier):
        msg = f"Invalid SQL identifier: {identifier!r}"
        raise ValueError(msg)
    return identifier


class DuckDBConfig(BaseModel):
    """Validated connection parameters for analytical DuckDB storage."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    database: str | Path = Field(
        default=":memory:",
        description="Target ':memory:' instance or filesystem path to a DuckDB file.",
    )
    read_only: bool = Field(
        default=False,
        description="Whether to open a file-backed DuckDB database in read-only mode.",
    )

    @model_validator(mode="after")
    def _check_memory_read_only(self) -> Self:
        if self.read_only and str(self.database) == ":memory:":
            msg = "An in-memory DuckDB database (':memory:') cannot be opened in read_only mode."
            raise ValueError(msg)
        return self


class StellarCatalogRecord(BaseModel):
    """Validated astrometric and photometric stellar catalog entry."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    source_id: int = Field(
        ...,
        ge=0,
        description="Unique non-negative catalog source identifier.",
    )
    ra_deg: float = Field(
        ...,
        ge=0.0,
        lt=360.0,
        description="ICRS Right Ascension in decimal degrees [0, 360).",
    )
    dec_deg: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="ICRS Declination in decimal degrees [-90, 90].",
    )
    mag_g: float = Field(
        ...,
        description="Apparent G-band magnitude.",
    )


class CatalogQueryParams(BaseModel):
    """Validated sky-box and limiting-magnitude parameters for catalog queries."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    ra_min_deg: float = Field(..., ge=0.0, lt=360.0)
    ra_max_deg: float = Field(..., ge=0.0, le=360.0)
    dec_min_deg: float = Field(..., ge=-90.0, le=90.0)
    dec_max_deg: float = Field(..., ge=-90.0, le=90.0)
    max_mag: float = Field(..., description="Maximum (faintest) magnitude threshold.")

    @model_validator(mode="after")
    def _validate_coordinate_ranges(self) -> Self:
        if self.dec_min_deg > self.dec_max_deg:
            msg = "dec_min_deg must be less than or equal to dec_max_deg."
            raise ValueError(msg)
        return self


class CatalogDatabase:
    """Context-managed DuckDB connection for OLAP stellar catalog operations."""

    def __init__(self, config: DuckDBConfig | None = None) -> None:
        import duckdb as _duckdb

        self._config: DuckDBConfig = config if config is not None else DuckDBConfig()
        db_target: str = (
            str(self._config.database)
            if isinstance(self._config.database, Path)
            else self._config.database
        )
        self._conn: duckdb.DuckDBPyConnection | None = _duckdb.connect(
            database=db_target,
            read_only=self._config.read_only,
        )

    @property
    def config(self) -> DuckDBConfig:
        """Return the validated DuckDB configuration."""
        return self._config

    def _require_connection(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            msg = "DuckDB connection is already closed."
            raise RuntimeError(msg)
        return self._conn

    def close(self) -> None:
        """Close the underlying DuckDB connection cleanly to release native handles."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def create_stellar_catalog_table(self, table_name: str = "stellar_catalog") -> None:
        """Create the analytical stellar catalog table if it does not already exist."""
        safe_table: str = _validate_identifier(table_name)
        conn = self._require_connection()
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {safe_table} (
                source_id BIGINT PRIMARY KEY,
                ra_deg DOUBLE NOT NULL,
                dec_deg DOUBLE NOT NULL,
                mag_g DOUBLE NOT NULL
            )
            """
        )

    def insert_stellar_records(
        self,
        records: Sequence[StellarCatalogRecord],
        table_name: str = "stellar_catalog",
    ) -> int:
        """Insert validated StellarCatalogRecord instances using bound parameters."""
        if not records:
            return 0
        safe_table: str = _validate_identifier(table_name)
        rows: list[tuple[int, float, float, float]] = [
            (rec.source_id, rec.ra_deg, rec.dec_deg, rec.mag_g) for rec in records
        ]
        conn = self._require_connection()
        conn.executemany(
            f"INSERT INTO {safe_table} (source_id, ra_deg, dec_deg, mag_g) VALUES (?, ?, ?, ?)",
            rows,
        )
        return len(rows)

    def query_numpy(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
    ) -> np.ndarray:
        """Execute a parametrized SQL query and return a structured NumPy array."""
        conn = self._require_connection()
        cursor = conn.execute(sql, list(params)) if params is not None else conn.execute(sql)
        columns: dict[str, Any] = cursor.fetchnumpy()
        if not columns:
            return np.empty(0)

        first_col = next(iter(columns.values()))
        n_rows: int = len(first_col)
        dtype_spec: list[tuple[str, np.dtype[Any]]] = [
            (col_name, np.asarray(col_values).dtype)
            for col_name, col_values in columns.items()
        ]
        structured: np.ndarray = np.empty(n_rows, dtype=dtype_spec)
        for col_name, col_values in columns.items():
            if np.ma.isMaskedArray(col_values):
                arr = np.asarray(col_values.data)
                if np.issubdtype(arr.dtype, np.floating):
                    arr = np.ma.filled(col_values, np.nan)
                structured[col_name] = arr
            else:
                structured[col_name] = col_values
        return structured

    def ping(self) -> bool:
        """Execute a lightweight probe query to verify active DuckDB connectivity."""
        if self._conn is None:
            return False
        row = self._conn.execute("SELECT 1").fetchone()
        return bool(row is not None and int(row[0]) == 1)

    def query_stellar_catalog(
        self,
        params: CatalogQueryParams,
        table_name: str = "stellar_catalog",
    ) -> np.ndarray:
        """Query stellar catalog entries within a sky box and limiting magnitude."""
        safe_table: str = _validate_identifier(table_name)
        ra_clause = (
            "ra_deg BETWEEN ? AND ?"
            if params.ra_min_deg <= params.ra_max_deg
            else "(ra_deg >= ? OR ra_deg <= ?)"
        )
        sql: str = (
            f"SELECT source_id, ra_deg, dec_deg, mag_g "
            f"FROM {safe_table} "
            f"WHERE {ra_clause} "
            f"AND dec_deg BETWEEN ? AND ? "
            f"AND mag_g <= ? "
            f"ORDER BY source_id ASC"
        )
        bound_params: list[float] = [
            params.ra_min_deg,
            params.ra_max_deg,
            params.dec_min_deg,
            params.dec_max_deg,
            params.max_mag,
        ]
        return self.query_numpy(sql, params=bound_params)

    def query_stellar_records(
        self,
        params: CatalogQueryParams,
        table_name: str = "stellar_catalog",
    ) -> list[StellarCatalogRecord]:
        """Query stellar catalog entries and return validated Pydantic records."""
        safe_table: str = _validate_identifier(table_name)
        ra_clause = (
            "ra_deg BETWEEN ? AND ?"
            if params.ra_min_deg <= params.ra_max_deg
            else "(ra_deg >= ? OR ra_deg <= ?)"
        )
        sql: str = (
            f"SELECT source_id, ra_deg, dec_deg, mag_g "
            f"FROM {safe_table} "
            f"WHERE {ra_clause} "
            f"AND dec_deg BETWEEN ? AND ? "
            f"AND mag_g <= ? "
            f"ORDER BY source_id ASC"
        )
        conn = self._require_connection()
        rows = conn.execute(
            sql,
            [
                params.ra_min_deg,
                params.ra_max_deg,
                params.dec_min_deg,
                params.dec_max_deg,
                params.max_mag,
            ],
        ).fetchall()
        return [
            StellarCatalogRecord(
                source_id=int(r[0]),
                ra_deg=float(r[1]),
                dec_deg=float(r[2]),
                mag_g=float(r[3]),
            )
            for r in rows
        ]
