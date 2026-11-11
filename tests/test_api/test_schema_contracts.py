"""Tests for NaN/Inf and extra-field rejection on API response schemas (P0-2, P1-2 subset)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from LUMIA.api.schemas import (
    CandidateCutoutEpoch,
    CandidateCutoutSequenceResponse,
    CandidateListResponse,
    CatalogStarItem,
    DetectionSummaryResponse,
    PipelineStatusResponse,
)
from LUMIA.core.schemas import CandidateDetection


def _epoch_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "epoch_index": 0,
        "bjd_tdb": 2460755.75658,
        "mag": 20.4,
        "mag_err": 0.08,
        "target_x_px": 15.0,
        "target_y_px": 15.0,
        "pixels": [[1.0] * 31 for _ in range(31)],
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_cutout_epoch_rejects_non_finite_scalars_and_pixels(bad: float) -> None:
    for field in ("bjd_tdb", "mag", "mag_err", "target_x_px", "target_y_px"):
        with pytest.raises(ValidationError):
            CandidateCutoutEpoch(**_epoch_kwargs(**{field: bad}))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        CandidateCutoutEpoch(**_epoch_kwargs(pixels=[[1.0, bad], [3.0, 4.0]]))  # type: ignore[arg-type]


def test_cutout_sequence_rejects_extra_fields_and_even_stamp_sizes() -> None:
    epoch = CandidateCutoutEpoch(**_epoch_kwargs())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        CandidateCutoutSequenceResponse(
            candidate_id="c1",
            field_id="f1",
            cutout_size=31,
            epochs=[epoch],
            surprise="x",  # type: ignore[call-arg]
        )

    # Even stamp size (32) must be rejected per P0-4 / Module 7 odd-dimension contract
    with pytest.raises(ValidationError):
        CandidateCutoutSequenceResponse(
            candidate_id="c1",
            field_id="f1",
            cutout_size=32,  # type: ignore[arg-type]
            epochs=[epoch],
        )

    # Mismatched pixel matrix dimensions must be rejected
    bad_epoch = CandidateCutoutEpoch(**_epoch_kwargs(pixels=[[1.0] * 10 for _ in range(10)]))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        CandidateCutoutSequenceResponse(
            candidate_id="c1",
            field_id="f1",
            cutout_size=31,
            epochs=[bad_epoch],
        )


def test_summary_and_catalog_items_reject_non_finite_values() -> None:
    with pytest.raises(ValidationError):
        DetectionSummaryResponse(
            total_candidates=1,
            high_confidence_count=1,
            distinct_fields=1,
            mean_snr=float("nan"),
            brightest_mag=20.0,
        )
    with pytest.raises(ValidationError):
        DetectionSummaryResponse(
            total_candidates=1,
            high_confidence_count=1,
            distinct_fields=1,
            mean_snr=5.0,
            brightest_mag=float("inf"),
        )
    with pytest.raises(ValidationError):
        CatalogStarItem(source_id=1, ra_deg=10.0, dec_deg=5.0, mag_g=float("nan"))


def test_status_response_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        PipelineStatusResponse(
            status="operational",
            duckdb_connected=True,
            total_jobs=0,
            debug=True,  # type: ignore[call-arg]
        )


def test_candidate_list_response_uses_canonical_candidate_and_serializes_new_fields() -> None:
    det = CandidateDetection(
        candidate_id="cand-001",
        field_id="ZTF_0412",
        ra_deg=180.0,
        dec_deg=15.0,
        bjd_tdb=2460755.5,
        mag=20.4,
        snr=12.5,
        cnn_score=0.9,
        model_version="rb-0.1",
        astrometric_uncertainty_arcsec=0.08,
        astrometric_residual_arcsec=0.11,
    )
    resp = CandidateListResponse(total_returned=1, candidates=[det])
    payload = resp.model_dump()["candidates"][0]

    assert payload["astrometric_residual_arcsec"] == pytest.approx(0.11)
    assert payload["astrometric_uncertainty_arcsec"] == pytest.approx(0.08)
    assert payload["model_version"] == "rb-0.1"
