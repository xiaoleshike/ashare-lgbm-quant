from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.data.security_identity import SecurityIdentityResolver
from ashare_quant.data.security_lifecycle import SecurityLifecycleResolver
from ashare_quant.data.security_lifecycle_audit import file_sha256
from ashare_quant.data.security_lifecycle_official_index import (
    OfficialLifecycleIndexService,
    publish_bulk_official_source_package,
    validate_bulk_official_source_package,
    validate_official_lifecycle_index,
)
from ashare_quant.data.security_lifecycle_resolution_hardened import (
    _relevant_calendar_hash,
    build_resolution_segments,
)


def _source(document: Path, **overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "exchange": "SSE",
        "official_source_type": "SSE_FACTBOOK",
        "official_url": "https://www.sse.com.cn/about/factbook.pdf",
        "official_document_id": "SSE-FACTBOOK-TEST",
        "source_year": 2012,
        "retrieved_at": "20260921T000000Z",
        "document_hash": file_sha256(document),
    }
    result.update(overrides)
    return result


def _record(code: str, **overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "canonical_ts_code": code,
        "security_name": f"Test {code}",
        "exchange": "SSE",
        "event_type": "FORMAL_LISTING_SUSPENSION_START",
        "announcement_date": "20110101",
        "effective_start": "20110104",
        "effective_end": "20110104",
        "source_locator": "page 10 table 2 row 1",
        "source_year": 2012,
    }
    result.update(overrides)
    return result


def test_one_bulk_package_indexes_multiple_securities(tmp_path: Path) -> None:
    document = tmp_path / "factbook.pdf"
    document.write_bytes(b"synthetic official exchange factbook")
    package = publish_bulk_official_source_package(
        source=_source(document),
        records=[_record("600001.SH"), _record("600002.SH")],
        document=document,
        reports_root=tmp_path / "reports",
    )

    result = OfficialLifecycleIndexService(
        reports_root=tmp_path / "reports",
        identity_resolver=SecurityIdentityResolver.empty(),
        lifecycle_evidence=SecurityLifecycleResolver.empty(),
        bulk_source_packages=(package,),
    ).build()

    manifest = validate_official_lifecycle_index(result.output_dir)
    events = pd.read_parquet(result.output_dir / "official_events.parquet")
    assert manifest["counts"]["events"] == 2
    assert set(events["canonical_ts_code"]) == {"600001.SH", "600002.SH"}
    assert not events["carry_in"].astype(bool).any()


def test_pre_research_official_start_is_recorded_as_carry_in(tmp_path: Path) -> None:
    document = tmp_path / "factbook.pdf"
    document.write_bytes(b"synthetic official exchange factbook")
    package = publish_bulk_official_source_package(
        source=_source(document, source_year=2009, official_document_id="SSE-2009"),
        records=[
            _record(
                "600001.SH",
                source_year=2009,
                announcement_date="20090520",
                effective_start="20090527",
                effective_end="20090527",
            )
        ],
        document=document,
        reports_root=tmp_path / "reports",
    )

    result = OfficialLifecycleIndexService(
        reports_root=tmp_path / "reports",
        identity_resolver=SecurityIdentityResolver.empty(),
        lifecycle_evidence=SecurityLifecycleResolver.empty(),
        bulk_source_packages=(package,),
    ).build()

    events = pd.read_parquet(result.output_dir / "official_events.parquet")
    assert bool(events.iloc[0]["carry_in"])


@pytest.mark.parametrize(
    "record",
    [
        _record("000001.SZ"),
        _record("600001.SH", effective_start="20110105", effective_end="20110104"),
    ],
)
def test_bulk_source_rejects_wrong_exchange_or_invalid_dates(
    tmp_path: Path, record: dict[str, object]
) -> None:
    document = tmp_path / "factbook.pdf"
    document.write_bytes(b"synthetic official exchange factbook")

    with pytest.raises(Exception, match="SECURITY_LIFECYCLE_BULK_SOURCE"):
        publish_bulk_official_source_package(
            source=_source(document),
            records=[record],
            document=document,
            reports_root=tmp_path / "reports",
        )


def test_duplicate_rows_deduplicate_and_conflicts_fail_closed(tmp_path: Path) -> None:
    document = tmp_path / "factbook.pdf"
    document.write_bytes(b"synthetic official exchange factbook")
    duplicate = _record("600001.SH")
    package = publish_bulk_official_source_package(
        source=_source(document),
        records=[duplicate, duplicate],
        document=document,
        reports_root=tmp_path / "reports",
    )
    _, rows = validate_bulk_official_source_package(package)
    assert len(rows) == 1

    with pytest.raises(Exception, match="CONFLICT"):
        publish_bulk_official_source_package(
            source=_source(document, official_document_id="SSE-CONFLICT"),
            records=[
                duplicate,
                _record("600001.SH", event_type="TERMINAL_DELISTING"),
            ],
            document=document,
            reports_root=tmp_path / "reports",
        )


def test_bulk_artifact_detects_tampered_document_and_rows(tmp_path: Path) -> None:
    document = tmp_path / "factbook.pdf"
    document.write_bytes(b"synthetic official exchange factbook")
    package = publish_bulk_official_source_package(
        source=_source(document),
        records=[_record("600001.SH")],
        document=document,
        reports_root=tmp_path / "reports",
    )
    frozen_document = next((package / "documents").iterdir())
    frozen_document.write_bytes(b"tampered")
    with pytest.raises(Exception, match="HASH_MISMATCH"):
        validate_bulk_official_source_package(package)

    frozen_document.write_bytes(b"synthetic official exchange factbook")
    rows = pd.read_parquet(package / "normalized_rows.parquet")
    rows.loc[0, "security_name"] = "tampered"
    rows.to_parquet(package / "normalized_rows.parquet", index=False)
    with pytest.raises(Exception, match="HASH_MISMATCH"):
        validate_bulk_official_source_package(package)


def _triage(session_count: int, end: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "parent_interval_id": "parent-1",
                "canonical_ts_code": "688766.SH",
                "gap_start": "20251126",
                "gap_end": end,
                "session_count": session_count,
                "triage_category": "LIKELY_PROVIDER_SUSPEND_D_GAP",
                "exchange": "SSE",
            }
        ]
    )


def test_partial_source_repair_splits_parent_and_reconciles_sessions() -> None:
    sessions = (
        "20251126",
        "20251127",
        "20251128",
        "20251201",
        "20251202",
        "20251203",
        "20251204",
        "20251205",
    )
    missing = [{"source_ts_code": "688766.SH", "trade_date": date} for date in sessions[1:]]
    provider = pd.DataFrame(
        [
            {
                "parent_interval_id": "parent-1",
                "source_completeness_status": "LOCAL_SUSPEND_D_INCOMPLETE",
                "missing_suspend_rows": json.dumps(missing),
                "missing_daily_rows": "[]",
            }
        ]
    )

    segments, reconciliation = build_resolution_segments(
        _triage(8, sessions[-1]), provider, pd.DataFrame(), sessions
    )

    assert len(segments) == 2
    assert list(segments["session_count"]) == [1, 7]
    assert list(segments["lifecycle_resolution"]) == [
        "STILL_UNRESOLVED",
        "RESOLVED_LOCAL_SUSPEND_D_GAP",
    ]
    assert int(segments["session_count"].sum()) == 8
    assert bool(reconciliation.iloc[0]["reconciled"])


def test_provider_no_suspend_evidence_is_not_lifecycle_resolution() -> None:
    provider = pd.DataFrame(
        [
            {
                "parent_interval_id": "parent-1",
                "source_completeness_status": "PROVIDER_HAS_NO_SUSPEND_EVIDENCE",
                "missing_suspend_rows": "[]",
                "missing_daily_rows": "[]",
            }
        ]
    )
    sessions = ("20251126",)

    segments, _ = build_resolution_segments(
        _triage(1, sessions[-1]), provider, pd.DataFrame(), sessions
    )

    assert segments.iloc[0]["source_resolution"] == "PROVIDER_HAS_NO_SUSPEND_EVIDENCE"
    assert segments.iloc[0]["lifecycle_resolution"] == "STILL_UNRESOLVED"
    assert bool(segments.iloc[0]["final_blocking"])


def test_out_of_range_official_record_does_not_resolve_gap() -> None:
    events = pd.DataFrame(
        [
            {
                "canonical_ts_code": "688766.SH",
                "event_type": "ORDINARY_FULL_DAY_SUSPENSION",
                "effective_start": "20251125",
                "effective_end": "20251125",
                "row_identity": "row-1",
                "package_id": "package-1",
                "evidence_source_category": "SSE_COMPANY_ANNOUNCEMENT",
                "official_source_type": "SSE_ANNOUNCEMENT",
                "status": "VERIFIED",
            }
        ]
    )
    sessions = ("20251126",)

    segments, _ = build_resolution_segments(
        _triage(1, sessions[-1]), pd.DataFrame(), events, sessions
    )

    assert segments.iloc[0]["lifecycle_resolution"] == "STILL_UNRESOLVED"


def test_formal_suspension_can_close_at_verified_terminal_transition() -> None:
    events = pd.DataFrame(
        [
            {
                "canonical_ts_code": "688766.SH",
                "event_type": "FORMAL_LISTING_SUSPENSION_START",
                "effective_start": "20251125",
                "effective_end": "20251125",
                "row_identity": "pause",
                "package_id": "package-1",
                "evidence_source_category": "SSE_BULK_LIFECYCLE_INDEX",
                "official_source_type": "SSE_FACTBOOK",
                "status": "VERIFIED",
            },
            {
                "canonical_ts_code": "688766.SH",
                "event_type": "TERMINAL_DELISTING",
                "effective_start": "20251128",
                "effective_end": "20251128",
                "row_identity": "terminal",
                "package_id": "package-1",
                "evidence_source_category": "SSE_BULK_LIFECYCLE_INDEX",
                "official_source_type": "SSE_FACTBOOK",
                "status": "VERIFIED",
            },
        ]
    )
    sessions = ("20251125", "20251126", "20251127")

    triage = _triage(3, sessions[-1]).assign(gap_start=sessions[0])
    segments, _ = build_resolution_segments(triage, pd.DataFrame(), events, sessions)

    assert segments["lifecycle_resolution"].tolist() == ["VERIFIED_LISTING_SUSPENSION"]
    assert int(segments.iloc[0]["session_count"]) == 3


def test_resolution_identity_calendar_hash_ignores_only_out_of_scope_sessions() -> None:
    triage = _triage(2, "20251128")
    baseline = _relevant_calendar_hash(triage, ("20251126", "20251128"))

    assert baseline == _relevant_calendar_hash(triage, ("20251126", "20251128", "20260102"))
    assert baseline != _relevant_calendar_hash(triage, ("20251126", "20251127", "20251128"))
