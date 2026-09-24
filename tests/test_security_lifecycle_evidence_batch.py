from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_lifecycle_audit import file_sha256
from ashare_quant.data.security_lifecycle_evidence_batch import (
    REPAIR_COLUMNS,
    RESOLUTION_COLUMNS,
    publish_d4d_evidence_batch,
    validate_d4d_evidence_batch,
)
from ashare_quant.data.security_lifecycle_official_index import (
    publish_bulk_official_source_package,
)
from ashare_quant.data.security_lifecycle_resolution import publish_official_evidence_package
from ashare_quant.utils.manifest import atomic_write_json


def test_d4d_batch_reconciles_interval_and_boundary_requests(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    document = tmp_path / "official.pdf"
    document.write_bytes(b"official")
    package = publish_official_evidence_package(
        payload=_evidence(document), document=document, reports_root=tmp_path
    )
    resolution = pd.DataFrame(
        [
            {
                "request_id": "request_gap",
                "resolution_status": "RESOLVED",
                "resolution_reason": "VERIFIED_NEW_EVIDENCE",
                "lifecycle_package_ids": package.name,
                "transition_package_ids": "",
                "provider_repair_ids": "",
                "authoritative_source_kind": "SSE_COMPANY_ANNOUNCEMENT",
                "notes": "verified",
            },
            {
                "request_id": "request_boundary",
                "resolution_status": "UNRESOLVED",
                "resolution_reason": "IDENTITY_TRANSITION_EVIDENCE_REQUIRED",
                "lifecycle_package_ids": "",
                "transition_package_ids": "",
                "provider_repair_ids": "",
                "authoritative_source_kind": "SZSE",
                "notes": "official transition document not found",
            },
        ],
        columns=RESOLUTION_COLUMNS,
    )
    result = publish_d4d_evidence_batch(
        d4c_audit=audit,
        request_resolution=resolution,
        lifecycle_evidence_packages=(package,),
        transition_evidence_packages=(),
        provider_source_repairs=pd.DataFrame(columns=REPAIR_COLUMNS),
        reports_root=tmp_path,
    )
    manifest = validate_d4d_evidence_batch(result.output_dir, d4c_audit=audit)
    rows = pd.read_parquet(result.output_dir / "request_resolution.parquet")
    assert manifest["counts"] == {
        "input_requests": 2,
        "input_intervals": 1,
        "resolved": 1,
        "partial": 0,
        "unresolved": 1,
        "retrieval_failed": 0,
        "verified_lifecycle_packages": 1,
        "verified_transition_packages": 0,
        "provider_source_repairs": 0,
        "interval_requests": 1,
        "boundary_requests": 1,
    }
    assert set(rows["finding_kind"]) == {"MISSING_PRICE_INTERVAL", "BOUNDARY_FINDING"}
    assert (
        rows.loc[rows["request_id"].eq("request_boundary"), "finding_id"]
        .item()
        .startswith("boundary_finding_")
    )


def test_d4d_batch_rejects_missing_request(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    resolution = pd.DataFrame(
        [
            {
                "request_id": "request_gap",
                "resolution_status": "UNRESOLVED",
                "resolution_reason": "NO_EVIDENCE",
                "lifecycle_package_ids": "",
                "transition_package_ids": "",
                "provider_repair_ids": "",
                "authoritative_source_kind": "SSE",
                "notes": "",
            }
        ],
        columns=RESOLUTION_COLUMNS,
    )
    with pytest.raises(
        DataValidationError, match="SECURITY_LIFECYCLE_D4D_BATCH_REQUEST_RECONCILIATION_FAILED"
    ):
        publish_d4d_evidence_batch(
            d4c_audit=audit,
            request_resolution=resolution,
            lifecycle_evidence_packages=(),
            transition_evidence_packages=(),
            provider_source_repairs=pd.DataFrame(columns=REPAIR_COLUMNS),
            reports_root=tmp_path,
        )


def test_d4d_batch_rejects_unproven_resolved_request(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    resolution = _resolutions().assign(
        resolution_status="RESOLVED", resolution_reason="VERIFIED_NEW_EVIDENCE"
    )
    with pytest.raises(
        DataValidationError, match="SECURITY_LIFECYCLE_D4D_BATCH_RESOLUTION_UNPROVEN"
    ):
        publish_d4d_evidence_batch(
            d4c_audit=audit,
            request_resolution=resolution,
            lifecycle_evidence_packages=(),
            transition_evidence_packages=(),
            provider_source_repairs=pd.DataFrame(columns=REPAIR_COLUMNS),
            reports_root=tmp_path,
        )


def test_d4d_batch_accepts_and_recursively_validates_bulk_event_package(
    tmp_path: Path,
) -> None:
    audit = _audit(tmp_path)
    document = tmp_path / "official.pdf"
    document.write_bytes(b"official bulk")
    package = publish_bulk_official_source_package(
        source={
            "exchange": "SSE",
            "official_source_type": "SSE_OFFICIAL_LIFECYCLE_NOTICE",
            "official_url": "https://www.sse.com.cn/official.pdf",
            "official_document_id": "fixture-bulk",
            "source_year": 2024,
            "retrieved_at": "2026-09-22T00:00:00+00:00",
            "document_hash": file_sha256(document),
        },
        records=[
            {
                "canonical_ts_code": "600000.SH",
                "security_name": "fixture",
                "event_type": "FORMAL_LISTING_SUSPENSION_START",
                "announcement_date": "20240102",
                "effective_start": "20240102",
                "effective_end": "20240102",
                "source_locator": "page 1",
            }
        ],
        document=document,
        reports_root=tmp_path,
    )
    resolution = _resolutions()
    resolution.loc[resolution["request_id"].eq("request_gap"), :] = [
        "request_gap",
        "RESOLVED",
        "VERIFIED_NEW_EVIDENCE",
        package.name,
        "",
        "",
        "SSE_BULK_LIFECYCLE_INDEX",
        "verified",
    ]
    result = publish_d4d_evidence_batch(
        d4c_audit=audit,
        request_resolution=resolution,
        lifecycle_evidence_packages=(package,),
        transition_evidence_packages=(),
        provider_source_repairs=pd.DataFrame(columns=REPAIR_COLUMNS),
        reports_root=tmp_path,
    )
    validate_d4d_evidence_batch(result.output_dir, d4c_audit=audit)
    frozen = package / "documents" / "document.pdf"
    frozen.write_bytes(b"tampered")
    with pytest.raises(DataValidationError, match="HASH_MISMATCH"):
        validate_d4d_evidence_batch(result.output_dir, d4c_audit=audit)


def _audit(root: Path) -> Path:
    audit_id = "security_lifecycle_d4c_audit_fixture"
    path = root / "audit" / audit_id
    path.mkdir(parents=True)
    intervals = pd.DataFrame(
        [
            {
                "current_interval_id": "interval_1",
                "canonical_ts_code": "600000.SH",
                "session_count": 1,
            }
        ]
    )
    boundaries = pd.DataFrame(
        [
            {
                "canonical_ts_code": "300114.SZ",
                "trade_date": "20100827",
                "boundary_type": "LISTING_METADATA_MISSING",
            }
        ]
    )
    requests = pd.DataFrame(
        [
            {
                "request_id": "request_gap",
                "current_interval_id": "interval_1",
                "canonical_ts_code": "600000.SH",
                "start_date": "20240102",
                "end_date": "20240102",
                "session_count": 1,
                "request_category": "SAME_DAY_SR_OFFICIAL_EVIDENCE_REQUIRED",
                "known": "gap",
                "missing_fact": "state",
                "preferred_official_source": "SSE",
                "blocking": True,
            },
            {
                "request_id": "request_boundary",
                "current_interval_id": "",
                "canonical_ts_code": "300114.SZ",
                "start_date": "20100827",
                "end_date": "20100827",
                "session_count": 0,
                "request_category": "IDENTITY_TRANSITION_EVIDENCE_REQUIRED",
                "known": "metadata",
                "missing_fact": "transition",
                "preferred_official_source": "SZSE",
                "blocking": True,
            },
        ]
    )
    summary = {
        "audit_id": audit_id,
        "counts": {
            "official_evidence_requests": 2,
            "unresolved_intervals": 1,
            "unresolved_sessions": 1,
            "blocking_boundaries": 1,
        },
    }
    atomic_write_json(path / "summary.json", summary)
    boundaries.to_parquet(path / "boundary_root_causes.parquet", index=False)
    intervals.to_parquet(path / "current_unresolved.parquet", index=False)
    pd.DataFrame().to_parquet(path / "existing_evidence_matches.parquet", index=False)
    requests.to_parquet(path / "official_evidence_requests.parquet", index=False)
    (path / "report.md").write_text("fixture\n", encoding="utf-8")
    hashes = {
        name: file_sha256(path / name)
        for name in (
            "summary.json",
            "boundary_root_causes.parquet",
            "current_unresolved.parquet",
            "existing_evidence_matches.parquet",
            "official_evidence_requests.parquet",
            "report.md",
        )
    }
    atomic_write_json(
        path / "manifest.json",
        {
            "schema_version": 1,
            "artifact_name": "security_lifecycle_d4c_audit",
            "audit_id": audit_id,
            "counts": summary["counts"],
            "logical_identity": {
                "source_scan_id": "scan_fixture",
                "source_scan_manifest_hash": "a" * 64,
            },
            "artifact_hashes": hashes,
        },
    )
    return path


def _resolutions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "request_id": request_id,
                "resolution_status": "UNRESOLVED",
                "resolution_reason": "NO_EVIDENCE",
                "lifecycle_package_ids": "",
                "transition_package_ids": "",
                "provider_repair_ids": "",
                "authoritative_source_kind": "SSE" if request_id == "request_gap" else "SZSE",
                "notes": "",
            }
            for request_id in ("request_gap", "request_boundary")
        ],
        columns=RESOLUTION_COLUMNS,
    )


def _evidence(document: Path) -> dict[str, object]:
    return {
        "canonical_ts_code": "600000.SH",
        "event_type": "ORDINARY_SUSPENSION",
        "effective_start": "20240102",
        "effective_end": "20240102",
        "exchange": "SSE",
        "official_source_type": "OFFICIAL_ISSUER_DISCLOSURE",
        "official_url": "https://www.sse.com.cn/official.pdf",
        "official_document_id": "fixture",
        "publication_date": "20240103",
        "effective_date": "20240102",
        "retrieved_at": "2026-09-22T00:00:00+00:00",
        "document_hash": file_sha256(document),
        "reviewed_fact": "The security was suspended for the full session.",
        "evidence_status": "VERIFIED",
        "document_security_codes": ["600000.SH"],
        "document_effective_dates": ["20240102"],
    }
