"""Strict TDD tests for the candidate detection DuckDB repository (P0-1)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from LUMIA.core.schemas import CandidateDetection
from LUMIA.data_pipeline.database import CatalogDatabase, DuckDBConfig
from LUMIA.data_pipeline.repositories import (
    CandidateDetectionRepository,
    CandidateDetectionSummary,
    CandidateListQuery,
)


def _sample_candidates() -> list[CandidateDetection]:
    return [
        CandidateDetection(
            candidate_id="cand-001",
            field_id="ZTF_0412",
            ra_deg=180.125,
            dec_deg=15.340,
            bjd_tdb=2460755.75658,
            mag=20.4,
            snr=12.5,
            cnn_score=0.96,
            model_version="rb-0.1",
            astrometric_uncertainty_arcsec=0.08,
            astrometric_residual_arcsec=0.11,
        ),
        CandidateDetection(
            candidate_id="cand-002",
            field_id="ZTF_0412",
            ra_deg=180.440,
            dec_deg=15.610,
            bjd_tdb=2460755.75662,
            mag=21.2,
            snr=7.5,
            cnn_score=0.84,
            model_version="rb-0.1",
            astrometric_uncertainty_arcsec=0.12,
            astrometric_residual_arcsec=0.18,
        ),
        CandidateDetection(
            candidate_id="cand-003",
            field_id="ZTF_0412",
            ra_deg=180.890,
            dec_deg=15.120,
            bjd_tdb=2460755.75650,
            mag=21.9,
            snr=4.8,
            cnn_score=0.42,
            astrometric_residual_arcsec=None,  # unlinked detection
        ),
        CandidateDetection(
            candidate_id="cand-004",
            field_id="ZTF_0999",
            ra_deg=210.220,
            dec_deg=-12.450,
            bjd_tdb=2460755.81200,
            mag=None,
            snr=None,
            cnn_score=None,
            astrometric_residual_arcsec=None,
        ),
    ]


class TestCandidateDetectionRepository:
    """Isolated tests for CandidateDetectionRepository DDL and DML."""

    def test_persists_full_and_nullable_candidates_faithfully(self) -> None:
        """NULL astrometric_residual_arcsec and optional photometry/CNN fields round-trip."""
        samples = _sample_candidates()
        with CatalogDatabase(DuckDBConfig()) as db:
            repo = CandidateDetectionRepository(db)
            repo.ensure_table()

            assert repo.insert_many([]) == 0
            assert repo.insert_many(samples) == 4

            for expected in samples:
                fetched = repo.get_by_id(expected.candidate_id)
                assert fetched == expected

            assert repo.get_by_id("does-not-exist") is None

    def test_summary_handles_empty_table_and_nullable_columns(self) -> None:
        with CatalogDatabase(DuckDBConfig()) as db:
            repo = CandidateDetectionRepository(db)
            repo.ensure_table()

            empty_summary: CandidateDetectionSummary = repo.summarize(min_cnn_score=0.8)
            assert empty_summary.total_candidates == 0
            assert empty_summary.high_confidence_count == 0
            assert empty_summary.distinct_fields == 0
            assert empty_summary.mean_snr is None
            assert empty_summary.brightest_mag is None

            repo.insert_many(_sample_candidates())
            summary = repo.summarize(min_cnn_score=0.8)
            assert summary.total_candidates == 4
            assert summary.high_confidence_count == 2
            assert summary.distinct_fields == 2
            assert summary.mean_snr == pytest.approx((12.5 + 7.5 + 4.8) / 3.0)
            assert summary.brightest_mag == pytest.approx(20.4)

    def test_list_candidates_filters_orders_and_paginates(self) -> None:
        with CatalogDatabase(DuckDBConfig()) as db:
            repo = CandidateDetectionRepository(db)
            repo.ensure_table()
            repo.insert_many(_sample_candidates())

            # min_cnn_score filters out NULL cnn_score rows and low-confidence rows
            filtered = repo.list_candidates(
                CandidateListQuery(min_cnn_score=0.8, field_id="ZTF_0412")
            )
            assert [c.candidate_id for c in filtered] == ["cand-001", "cand-002"]

            # Without min_cnn_score, all rows are returned with scored rows ordered first
            all_rows = repo.list_candidates(CandidateListQuery(min_cnn_score=None))
            assert [c.candidate_id for c in all_rows] == [
                "cand-001",
                "cand-002",
                "cand-003",
                "cand-004",
            ]

            # Pagination
            page = repo.list_candidates(
                CandidateListQuery(min_cnn_score=0.0, limit=1, offset=1)
            )
            assert [c.candidate_id for c in page] == ["cand-002"]

    def test_ping_checks_database_connectivity(self) -> None:
        with CatalogDatabase(DuckDBConfig()) as db:
            assert db.ping() is True

    def test_query_and_summary_models_enforce_strict_finite_policy(self) -> None:
        with pytest.raises(ValidationError):
            CandidateListQuery(min_cnn_score=float("nan"))
        with pytest.raises(ValidationError):
            CandidateListQuery(limit=0)
        with pytest.raises(ValidationError):
            CandidateListQuery(extra_key=1)  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            CandidateDetectionSummary(
                total_candidates=1,
                high_confidence_count=0,
                distinct_fields=1,
                mean_snr=float("inf"),
            )
