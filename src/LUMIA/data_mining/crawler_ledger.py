"""Append-only transaction ledger for tracking idempotent crawler tracklet ingestion."""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from LUMIA.core.schemas import LumiaBaseModel

DEFAULT_LEDGER_DB_PATH = Path("data/ground_truth/crawler_checkpoint.db")


class CrawlerLedgerEntry(LumiaBaseModel):
    """Ledger transaction record for an ingested tracklet."""

    object_id: str = Field(..., min_length=1, max_length=64)
    tracklet_id: str = Field(..., min_length=1, max_length=128)
    n_frames: int = Field(..., ge=1)
    split: str = Field(..., min_length=1, max_length=32)
    status: str = Field(default="COMPLETED", min_length=1, max_length=32)
    created_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class SQLiteCrawlerLedger:
    """Lightweight thread-safe SQLite WAL checkpoint ledger for tracklet ingestion."""

    def __init__(self, db_path: Path | str = DEFAULT_LEDGER_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection configured with WAL journal mode."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        """Initialize the ledger table with primary key constraints."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tracklet_ingestion_log (
                    object_id TEXT NOT NULL,
                    tracklet_id TEXT NOT NULL,
                    n_frames INTEGER NOT NULL,
                    split TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    PRIMARY KEY (object_id, tracklet_id)
                )
                """
            )

    def is_tracklet_completed(self, object_id: str, tracklet_id: str) -> bool:
        """Check whether a tracklet has already been successfully committed."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 1 FROM tracklet_ingestion_log
                WHERE object_id = ? AND tracklet_id = ? AND status = 'COMPLETED'
                """,
                (object_id, tracklet_id),
            )
            return cursor.fetchone() is not None

    def commit_tracklet(self, entry: CrawlerLedgerEntry) -> None:
        """Atomically record a completed tracklet ingestion transaction."""
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO tracklet_ingestion_log (
                    object_id, tracklet_id, n_frames, split, status, created_at_utc
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(object_id, tracklet_id) DO UPDATE SET
                    n_frames = excluded.n_frames,
                    split = excluded.split,
                    status = excluded.status,
                    created_at_utc = excluded.created_at_utc
                """,
                (
                    entry.object_id,
                    entry.tracklet_id,
                    entry.n_frames,
                    entry.split,
                    entry.status,
                    entry.created_at_utc,
                ),
            )

    def get_completed_ids(self) -> set[tuple[str, str]]:
        """Retrieve the set of all completed (object_id, tracklet_id) tuples."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT object_id, tracklet_id FROM tracklet_ingestion_log WHERE status = 'COMPLETED'"
            )
            return {(row[0], row[1]) for row in cursor.fetchall()}

    def clear(self) -> None:
        """Clear all records from the crawler ledger."""
        with self._lock, self._get_connection() as conn:
            conn.execute("DELETE FROM tracklet_ingestion_log")

    def delete_for_objects(self, object_ids: list[str]) -> None:
        """Delete checkpoint records for a specific list of object IDs."""
        if not object_ids:
            return
        placeholders = ",".join("?" for _ in object_ids)
        with self._lock, self._get_connection() as conn:
            conn.execute(
                f"DELETE FROM tracklet_ingestion_log WHERE object_id IN ({placeholders})",
                object_ids,
            )

    def delete_synthetic(self) -> None:
        """Delete all synthetic tracklet records (starting with SYNTH_)."""
        with self._lock, self._get_connection() as conn:
            conn.execute("DELETE FROM tracklet_ingestion_log WHERE tracklet_id LIKE 'SYNTH_%'")
