from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.data.security_identity import SecurityIdentityResolver
from ashare_quant.data.security_lifecycle import SecurityLifecycleResolver
from ashare_quant.data.security_lifecycle_audit import file_sha256
from ashare_quant.data.security_lifecycle_resolution import (
    _interval_id,
    _resolve_intervals,
    _validate_resolution_reconciliation,
    publish_official_evidence_package,
    validate_official_evidence_package,
)
from ashare_quant.data.security_lifecycle_source_probe import SecurityLifecycleSourceProbe
from ashare_quant.data.security_lifecycle_triage import REQUIRED_ARTIFACTS


class FakeProvider:
    def __init__(
        self,
        *,
        suspend_d: pd.DataFrame | None = None,
        daily: pd.DataFrame | None = None,
        error: Exception | None = None,
    ) -> None:
        self.frames = {
            "suspend_d": suspend_d if suspend_d is not None else _empty_suspend(),
            "daily": daily if daily is not None else _empty_daily(),
        }
        self.error = error

    def query(self, endpoint: str, **params: object) -> pd.DataFrame:
        del params
        if self.error is not None:
            raise self.error
        return self.frames[endpoint].copy()


def _empty_suspend() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts_code", "trade_date", "suspend_timing", "suspend_type"])


def _empty_daily() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]
    )


def _suspend() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20240103"],
            "suspend_timing": [None],
            "suspend_type": ["S"],
        }
    )


def _daily() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20240103"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "vol": [100.0],
            "amount": [1020.0],
        }
    )


def _write_fixture(
    tmp_path: Path,
    *,
    local_suspend: pd.DataFrame | None = None,
    local_daily: pd.DataFrame | None = None,
) -> tuple[Path, Path, Path, SecurityIdentityResolver]:
    resolver = SecurityIdentityResolver.empty()
    raw = tmp_path / "raw"
    for dataset, frame in {
        "trade_cal": pd.DataFrame(
            {
                "cal_date": ["20240102", "20240103", "20240104"],
                "is_open": [1, 1, 1],
            }
        ),
        "suspend_d": local_suspend if local_suspend is not None else _empty_suspend(),
        "daily": local_daily if local_daily is not None else _empty_daily(),
    }.items():
        path = raw / dataset
        path.mkdir(parents=True)
        frame.to_parquet(path / "part.parquet", index=False)

    triage = tmp_path / "security_lifecycle_triage_fixture"
    triage.mkdir()
    unresolved = pd.DataFrame(
        {
            "canonical_ts_code": ["000001.SZ"],
            "gap_start": ["20240103"],
            "gap_end": ["20240103"],
            "session_count": [1],
            "triage_category": ["LIKELY_PROVIDER_SUSPEND_D_GAP"],
        }
    )
    for name in REQUIRED_ARTIFACTS:
        path = triage / name
        if name == "unresolved_triage.parquet":
            unresolved.to_parquet(path, index=False)
        elif name.endswith(".parquet"):
            pd.DataFrame({"fixture": pd.Series(dtype="string")}).to_parquet(path, index=False)
        else:
            path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_name": "security_lifecycle_triage",
        "triage_id": triage.name,
        "logical_identity": {
            "security_identity_mapping_hash": resolver.mapping_hash,
        },
        "counts": {"unresolved_intervals": 1},
        "artifact_hashes": {
            name: file_sha256(triage / name) for name in sorted(REQUIRED_ARTIFACTS)
        },
    }
    (triage / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return triage / "manifest.json", raw, tmp_path / "reports", resolver


def _run(
    tmp_path: Path,
    provider: FakeProvider,
    *,
    local_suspend: pd.DataFrame | None = None,
    local_daily: pd.DataFrame | None = None,
) -> tuple[str, Path]:
    triage, raw, reports, resolver = _write_fixture(
        tmp_path,
        local_suspend=local_suspend,
        local_daily=local_daily,
    )
    result = SecurityLifecycleSourceProbe(
        triage_manifest=triage,
        raw_root=raw,
        reports_root=reports,
        identity_resolver=resolver,
        provider_client=provider,
        buffer_sessions=0,
    ).run()
    comparison = pd.read_parquet(result.output_dir / "comparison.parquet")
    return str(comparison.iloc[0]["source_completeness_status"]), result.output_dir


def test_provider_suspend_rows_missing_locally_require_suspend_repair(tmp_path: Path) -> None:
    status, output = _run(tmp_path, FakeProvider(suspend_d=_suspend()))

    assert status == "LOCAL_SUSPEND_D_INCOMPLETE"
    assert bool(pd.read_parquet(output / "comparison.parquet").iloc[0]["repair_required"])


def test_provider_daily_row_missing_locally_requires_daily_repair(tmp_path: Path) -> None:
    status, _ = _run(tmp_path, FakeProvider(daily=_daily()))

    assert status == "LOCAL_DAILY_INCOMPLETE"


def test_matching_provider_and_local_rows_agree(tmp_path: Path) -> None:
    status, _ = _run(
        tmp_path,
        FakeProvider(suspend_d=_suspend()),
        local_suspend=_suspend(),
    )

    assert status == "PROVIDER_AND_LOCAL_AGREE"


def test_no_provider_suspend_evidence_does_not_resolve_suspension(tmp_path: Path) -> None:
    status, _ = _run(tmp_path, FakeProvider())

    assert status == "PROVIDER_HAS_NO_SUSPEND_EVIDENCE"


def test_provider_failure_is_inconclusive_and_redacts_error_message(tmp_path: Path) -> None:
    secret_marker = "secret-token-must-not-leak"  # noqa: S105 - redaction fixture.
    status, output = _run(tmp_path, FakeProvider(error=RuntimeError(secret_marker)))

    assert status == "PROBE_INCONCLUSIVE"
    for path in output.iterdir():
        assert secret_marker.encode() not in path.read_bytes()


@pytest.mark.parametrize("buffer_sessions", [-1])
def test_invalid_probe_buffer_fails_closed(tmp_path: Path, buffer_sessions: int) -> None:
    triage, raw, reports, resolver = _write_fixture(tmp_path)
    service = SecurityLifecycleSourceProbe(
        triage_manifest=triage,
        raw_root=raw,
        reports_root=reports,
        identity_resolver=resolver,
        provider_client=FakeProvider(),
        buffer_sessions=buffer_sessions,
    )

    with pytest.raises(Exception, match="BUFFER_INVALID"):
        service.run()


def _official_payload(document: Path, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "canonical_ts_code": "600001.SH",
        "event_type": "LISTING_SUSPENDED",
        "effective_start": "20240103",
        "effective_end": "20240105",
        "exchange": "SSE",
        "official_source_type": "SSE_ANNOUNCEMENT",
        "official_url": "https://www.sse.com.cn/official.pdf",
        "official_document_id": "SSE-TEST-1",
        "publication_date": "20240102",
        "effective_date": "20240103",
        "retrieved_at": "20260921T000000Z",
        "document_hash": file_sha256(document),
        "reviewed_fact": "Formal listing suspension begins on the effective date.",
        "evidence_status": "VERIFIED",
        "document_security_codes": ["600001.SH"],
        "document_effective_dates": ["20240103"],
    }
    payload.update(overrides)
    return payload


def test_verified_official_evidence_is_frozen_and_hash_validated(tmp_path: Path) -> None:
    document = tmp_path / "official.pdf"
    document.write_bytes(b"synthetic official document")

    package = publish_official_evidence_package(
        payload=_official_payload(document),
        document=document,
        reports_root=tmp_path / "reports",
    )

    evidence = validate_official_evidence_package(package)
    assert evidence["evidence_status"] == "VERIFIED"
    frozen_document = next((package / "documents").iterdir())
    frozen_document.write_bytes(b"tampered")
    with pytest.raises(Exception, match="HASH_MISMATCH"):
        validate_official_evidence_package(package)


@pytest.mark.parametrize(
    "overrides",
    [
        {"official_url": "https://example.com/not-official.pdf"},
        {"document_security_codes": ["600002.SH"]},
        {"document_effective_dates": ["20240104"]},
    ],
)
def test_invalid_official_evidence_fails_closed(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    document = tmp_path / "official.pdf"
    document.write_bytes(b"synthetic official document")

    with pytest.raises(Exception, match="OFFICIAL"):
        publish_official_evidence_package(
            payload=_official_payload(document, **overrides),
            document=document,
            reports_root=tmp_path / "reports",
        )


def test_resolution_preserves_all_parent_interval_identities_and_sessions() -> None:
    triage = pd.DataFrame(
        [
            {
                "canonical_ts_code": f"{index:06d}.SZ",
                "gap_start": "20240103",
                "gap_end": "20240103",
                "session_count": 1,
                "triage_category": "UNKNOWN_REQUIRES_OFFICIAL_EVIDENCE",
            }
            for index in range(121)
        ]
    )
    triage["parent_interval_id"] = triage.apply(_interval_id, axis=1)

    resolved = _resolve_intervals(
        triage,
        pd.DataFrame(),
        pd.DataFrame(),
        SecurityLifecycleResolver.empty(),
    )

    _validate_resolution_reconciliation(triage, resolved)
    assert len(resolved) == 121
    assert int(resolved["session_count"].sum()) == 121
    assert set(resolved["resolution"]) == {"STILL_UNRESOLVED"}


def test_partial_provider_suspension_coverage_does_not_resolve_whole_interval() -> None:
    triage = pd.DataFrame(
        [
            {
                "canonical_ts_code": "600001.SH",
                "gap_start": "20240102",
                "gap_end": "20240103",
                "session_count": 2,
                "triage_category": "LIKELY_PROVIDER_SUSPEND_D_GAP",
            }
        ]
    )
    triage["parent_interval_id"] = triage.apply(_interval_id, axis=1)
    provider = pd.DataFrame(
        [
            {
                "parent_interval_id": triage.iloc[0]["parent_interval_id"],
                "source_completeness_status": "LOCAL_SUSPEND_D_INCOMPLETE",
                "probe_failed": False,
                "provider_suspend_hash": "suspend-hash",
                "provider_daily_hash": "daily-hash",
                "provider_full_day_s_rows": 1,
                "provider_valid_daily_rows": 0,
                "local_full_day_s_rows": 0,
                "local_valid_daily_rows": 0,
                "missing_suspend_rows": json.dumps(
                    [{"source_ts_code": "600001.SH", "trade_date": "20240102"}]
                ),
                "missing_daily_rows": "[]",
            }
        ]
    )

    resolved = _resolve_intervals(
        triage, provider, pd.DataFrame(), SecurityLifecycleResolver.empty()
    )

    assert resolved.iloc[0]["resolution"] == "STILL_UNRESOLVED"
    assert resolved.iloc[0]["repair_action"] == "REINGEST_SUSPEND_D"
    assert int(resolved.iloc[0]["unexplained_sessions_after_probe"]) == 1
