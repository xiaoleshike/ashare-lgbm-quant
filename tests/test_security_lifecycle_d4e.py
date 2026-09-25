"""Contract tests for the final D.4E evidence batch."""

from __future__ import annotations

import pandas as pd
import pytest

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_lifecycle_d4e import (
    FINAL_DETAIL_COLUMNS,
    RESOLUTION_COLUMNS,
    _current_residuals,
    _normalize_final_details,
    _normalize_resolution,
    _validate_references,
)


def _requests() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"request_id": "interval-1"},
            {"request_id": "boundary-1"},
        ]
    )


def _resolutions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "request_id": "interval-1",
                "resolution_status": "VERIFIED_RESOLVED",
                "resolution_reason": "official lifecycle interval",
                "lifecycle_package_ids": "lifecycle-1",
                "listing_metadata_package_ids": "",
                "transition_package_ids": "",
                "reviewed_fact": "formal suspension and resumption",
                "notes": "",
            },
            {
                "request_id": "boundary-1",
                "resolution_status": "VERIFIED_RESOLVED",
                "resolution_reason": "official listing date",
                "lifecycle_package_ids": "",
                "listing_metadata_package_ids": "metadata-1",
                "transition_package_ids": "",
                "reviewed_fact": "listed on the authoritative date",
                "notes": "",
            },
        ],
        columns=RESOLUTION_COLUMNS,
    )


def test_d4e_resolution_keeps_interval_and_boundary_requests_exact() -> None:
    normalized = _normalize_resolution(_resolutions(), _requests())

    assert normalized["request_id"].tolist() == ["boundary-1", "interval-1"]


def test_d4e_resolution_rejects_missing_request() -> None:
    with pytest.raises(
        DataValidationError,
        match="SECURITY_LIFECYCLE_D4E_BATCH_REQUEST_RECONCILIATION_FAILED",
    ):
        _normalize_resolution(_resolutions().iloc[:1], _requests())


def test_d4e_verified_resolution_requires_valid_frozen_package() -> None:
    resolution = _normalize_resolution(_resolutions(), _requests())
    lifecycle = [{"package_id": "lifecycle-1"}]
    metadata = [{"package_id": "metadata-1"}]

    _validate_references(resolution, lifecycle, metadata)
    with pytest.raises(
        DataValidationError,
        match="SECURITY_LIFECYCLE_D4E_BATCH_PACKAGE_REFERENCE_INVALID",
    ):
        _validate_references(resolution, [], metadata)


def test_d4e_final_queue_reconciles_every_current_interval() -> None:
    unresolved = pd.DataFrame(
        [
            {
                "canonical_ts_code": "000001.SZ",
                "gap_start": "20200102",
                "gap_end": "20200103",
                "session_count": 2,
                "reason": "authoritative lifecycle state missing",
            }
        ]
    )
    current = _current_residuals(unresolved, "security_lifecycle_scan")
    details = pd.DataFrame(
        [
            {
                "current_interval_id": current.iloc[0]["current_interval_id"],
                "official_sources_searched": "official exchange index",
                "documents_reviewed": "listing metadata only",
                "why_evidence_insufficient": "no suspension fact",
                "next_operator_action": "retrieve issuer archive",
                "final_classification": "MANUAL_ARCHIVE_RETRIEVAL_REQUIRED",
            }
        ],
        columns=FINAL_DETAIL_COLUMNS,
    )

    normalized = _normalize_final_details(details, unresolved, "security_lifecycle_scan")

    assert normalized["current_interval_id"].tolist() == current["current_interval_id"].tolist()


def test_d4e_final_queue_rejects_missing_current_interval() -> None:
    unresolved = pd.DataFrame(
        [
            {
                "canonical_ts_code": "000001.SZ",
                "gap_start": "20200102",
                "gap_end": "20200103",
                "session_count": 2,
                "reason": "authoritative lifecycle state missing",
            }
        ]
    )

    with pytest.raises(
        DataValidationError, match="SECURITY_LIFECYCLE_D4E_FINAL_QUEUE_DETAILS_INVALID"
    ):
        _normalize_final_details(
            pd.DataFrame(columns=FINAL_DETAIL_COLUMNS), unresolved, "security_lifecycle_scan"
        )
