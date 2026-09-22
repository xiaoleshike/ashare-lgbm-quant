from __future__ import annotations

import pandas as pd
import pytest

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_lifecycle_evidence_closure import (
    closure_counts,
    h5_reachability_diagnostic,
    reconcile_evidence_queue,
)


def _queue() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "resolution_segment_id": "input-1",
                "parent_interval_id": "parent-1",
                "canonical_ts_code": "000001.SZ",
                "exchange": "SZSE",
                "segment_start": "20240102",
                "segment_end": "20240104",
                "session_count": 3,
                "source_resolution": "PROVIDER_HAS_NO_SUSPEND_EVIDENCE",
                "lifecycle_resolution": "STILL_UNRESOLVED",
            }
        ]
    )


def _current() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "resolution_segment_id": "current-1",
                "parent_interval_id": "parent-1",
                "canonical_ts_code": "000001.SZ",
                "segment_start": "20240102",
                "segment_end": "20240103",
                "source_resolution": "PROVIDER_HAS_NO_SUSPEND_EVIDENCE",
                "lifecycle_resolution": "VERIFIED_LISTING_SUSPENSION",
                "official_evidence_package": "official-1",
                "evidence_source_category": "SZSE_BULK_LIFECYCLE_INDEX",
                "repair_required": False,
                "final_blocking": False,
            },
            {
                "resolution_segment_id": "current-2",
                "parent_interval_id": "parent-1",
                "canonical_ts_code": "000001.SZ",
                "segment_start": "20240104",
                "segment_end": "20240104",
                "source_resolution": "PROVIDER_HAS_NO_SUSPEND_EVIDENCE",
                "lifecycle_resolution": "STILL_UNRESOLVED",
                "official_evidence_package": "",
                "evidence_source_category": "",
                "repair_required": False,
                "final_blocking": True,
            },
        ]
    )


def test_closure_reconciles_every_input_session_without_overlap() -> None:
    result = reconcile_evidence_queue(_queue(), _current(), ("20240102", "20240103", "20240104"))

    assert int(result["session_count"].sum()) == 3
    assert set(result["lifecycle_resolution"]) == {
        "VERIFIED_LISTING_SUSPENSION",
        "STILL_UNRESOLVED",
    }
    assert set(result["input_segment_id"]) == {"input-1"}


def test_closure_fails_when_current_segments_do_not_cover_input_session() -> None:
    current = _current().iloc[:1].copy()

    with pytest.raises(DataValidationError, match="QUEUE_OVERLAP"):
        reconcile_evidence_queue(_queue(), current, ("20240102", "20240103", "20240104"))


def test_h5_reachability_uses_listed_envelope_not_labels_or_universe() -> None:
    triage = pd.DataFrame(
        [
            {
                "parent_interval_id": "parent-1",
                "list_date": "20200101",
                "delist_date": None,
                "previous_valid_quote": "20231229",
            }
        ]
    )

    result = h5_reachability_diagnostic(_queue(), triage)

    assert result.iloc[0]["reachability"] == "POTENTIALLY_H5_REACHABLE"
    assert "label" not in " ".join(result.columns).lower()


def test_h5_reachability_accepts_reconciled_blocking_child_segments() -> None:
    triage = pd.DataFrame(
        [
            {
                "parent_interval_id": "parent-1",
                "list_date": "20200101",
                "delist_date": None,
                "previous_valid_quote": "20231229",
            }
        ]
    )
    blocking = reconcile_evidence_queue(_queue(), _current(), ("20240102", "20240103", "20240104"))
    blocking = blocking[blocking["final_blocking"]].copy()

    result = h5_reachability_diagnostic(blocking, triage)

    assert len(result) == 1
    assert result.iloc[0]["session_count"] == 1
    assert result.iloc[0]["reachability"] == "POTENTIALLY_H5_REACHABLE"


def test_closure_counts_keep_unresolved_blocking() -> None:
    coverage = reconcile_evidence_queue(_queue(), _current(), ("20240102", "20240103", "20240104"))
    reachability = pd.DataFrame(
        [
            {
                "canonical_ts_code": "000001.SZ",
                "session_count": 3,
                "reachability": "POTENTIALLY_H5_REACHABLE",
            }
        ]
    )
    unresolved = pd.DataFrame([{"input_segment_id": "input-1"}])

    counts = closure_counts(_queue(), coverage, unresolved, reachability)

    assert counts["still_unresolved"] == 1
    assert counts["unverified_required_evidence"] == 1
