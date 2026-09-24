"""Immutable evidence batches for current-contract lifecycle requests.

The batch records evidence research outcomes without changing lifecycle state.  Runtime
activation remains a separate Official Index and typed-catalog publication step.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_identity_transition import validate_transition_evidence_package
from ashare_quant.data.security_lifecycle_audit import canonical_payload_hash, file_sha256
from ashare_quant.data.security_lifecycle_official_index import (
    BULK_ARTIFACT_NAME,
    validate_bulk_official_source_package,
)
from ashare_quant.data.security_lifecycle_resolution import validate_official_evidence_package
from ashare_quant.utils.manifest import atomic_write_json

type DataFrame = pd.DataFrame
type JsonObject = dict[str, Any]

D4D_EVIDENCE_BATCH_SCHEMA_VERSION = 1
D4D_EVIDENCE_BATCH_CONTRACT_VERSION = "security_lifecycle_d4d_evidence_batch_v1"
D4D_EVIDENCE_BATCH_ARTIFACT_NAME = "security_lifecycle_d4d_evidence_batch"
D4C_AUDIT_ARTIFACT_NAME = "security_lifecycle_d4c_audit"
D4C_AUDIT_FILES = frozenset(
    {
        "summary.json",
        "boundary_root_causes.parquet",
        "current_unresolved.parquet",
        "existing_evidence_matches.parquet",
        "official_evidence_requests.parquet",
        "report.md",
    }
)
D4D_BATCH_FILES = frozenset(
    {
        "request_resolution.parquet",
        "verified_lifecycle_packages.json",
        "verified_transition_packages.json",
        "provider_source_repairs.parquet",
        "remaining_requests.parquet",
        "summary.json",
    }
)
RESOLUTION_STATUSES = frozenset({"RESOLVED", "PARTIAL", "UNRESOLVED", "EVIDENCE_RETRIEVAL_FAILED"})
RESOLUTION_COLUMNS = (
    "request_id",
    "resolution_status",
    "resolution_reason",
    "lifecycle_package_ids",
    "transition_package_ids",
    "provider_repair_ids",
    "authoritative_source_kind",
    "notes",
)
REPAIR_COLUMNS = (
    "repair_id",
    "request_id",
    "dataset",
    "canonical_ts_code",
    "start_date",
    "end_date",
    "row_count",
    "source_artifact_id",
    "source_artifact_hash",
    "repair_action",
)


@dataclass(frozen=True, slots=True)
class D4DEvidenceBatchResult:
    """Published D.4D evidence-request reconciliation."""

    batch_id: str
    output_dir: Path
    counts: JsonObject
    idempotent: bool


def publish_d4d_evidence_batch(
    *,
    d4c_audit: Path,
    request_resolution: DataFrame,
    lifecycle_evidence_packages: tuple[Path, ...],
    transition_evidence_packages: tuple[Path, ...],
    provider_source_repairs: DataFrame,
    reports_root: Path,
) -> D4DEvidenceBatchResult:
    """Validate every current request and publish one immutable evidence batch."""

    audit, requests, intervals, boundaries = validate_d4c_request_artifact(d4c_audit)
    lifecycle = [_validate_lifecycle_package(path) for path in lifecycle_evidence_packages]
    transitions = [
        {**evidence, "package_id": path.name, "package_hash": manifest_hash}
        for path in transition_evidence_packages
        for evidence, manifest_hash in [validate_transition_evidence_package(path)]
    ]
    resolution = _normalize_request_resolution(request_resolution, requests)
    repairs = _normalize_repairs(provider_source_repairs, requests)
    _validate_resolution_references(resolution, lifecycle, transitions, repairs)
    reconciled = _attach_request_identity(resolution, requests, intervals, boundaries)
    remaining = reconciled[reconciled["resolution_status"].astype(str).ne("RESOLVED")].copy()
    lifecycle_index = _package_index(lifecycle)
    transition_index = _package_index(transitions)
    counts = {
        "input_requests": int(len(requests)),
        "input_intervals": int(len(intervals)),
        "resolved": int(reconciled["resolution_status"].eq("RESOLVED").sum()),
        "partial": int(reconciled["resolution_status"].eq("PARTIAL").sum()),
        "unresolved": int(reconciled["resolution_status"].eq("UNRESOLVED").sum()),
        "retrieval_failed": int(
            reconciled["resolution_status"].eq("EVIDENCE_RETRIEVAL_FAILED").sum()
        ),
        "verified_lifecycle_packages": len(lifecycle_index),
        "verified_transition_packages": len(transition_index),
        "provider_source_repairs": int(len(repairs)),
        "interval_requests": int(reconciled["finding_kind"].eq("MISSING_PRICE_INTERVAL").sum()),
        "boundary_requests": int(reconciled["finding_kind"].eq("BOUNDARY_FINDING").sum()),
    }
    logical: JsonObject = {
        "schema_version": D4D_EVIDENCE_BATCH_SCHEMA_VERSION,
        "contract_version": D4D_EVIDENCE_BATCH_CONTRACT_VERSION,
        "d4c_audit_id": audit["audit_id"],
        "d4c_audit_manifest_hash": file_sha256(d4c_audit / "manifest.json"),
        "source_scan_id": cast(JsonObject, audit["logical_identity"])["source_scan_id"],
        "source_scan_manifest_hash": cast(JsonObject, audit["logical_identity"])[
            "source_scan_manifest_hash"
        ],
        "request_resolution_hash": _frame_hash(reconciled),
        "lifecycle_package_hashes": sorted(str(item["package_hash"]) for item in lifecycle_index),
        "transition_package_hashes": sorted(str(item["package_hash"]) for item in transition_index),
        "provider_source_repairs_hash": _frame_hash(repairs),
    }
    batch_id = f"security_lifecycle_d4d_evidence_batch_{canonical_payload_hash(logical)[:24]}"
    output = reports_root / "security_lifecycle_d4d_evidence_batch" / batch_id
    if output.exists():
        manifest = validate_d4d_evidence_batch(output, d4c_audit=d4c_audit)
        return D4DEvidenceBatchResult(batch_id, output, cast(JsonObject, manifest["counts"]), True)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{batch_id}.staging-"))
    try:
        reconciled.to_parquet(staging / "request_resolution.parquet", index=False)
        atomic_write_json(
            staging / "verified_lifecycle_packages.json", {"packages": lifecycle_index}
        )
        atomic_write_json(
            staging / "verified_transition_packages.json", {"packages": transition_index}
        )
        repairs.to_parquet(staging / "provider_source_repairs.parquet", index=False)
        remaining.to_parquet(staging / "remaining_requests.parquet", index=False)
        atomic_write_json(
            staging / "summary.json",
            {
                "batch_id": batch_id,
                "counts": counts,
                "request_reconciliation_complete": sum(
                    int(counts[key])
                    for key in ("resolved", "partial", "unresolved", "retrieval_failed")
                )
                == counts["input_requests"],
            },
        )
        hashes = {name: file_sha256(staging / name) for name in sorted(D4D_BATCH_FILES)}
        atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": D4D_EVIDENCE_BATCH_SCHEMA_VERSION,
                "artifact_name": D4D_EVIDENCE_BATCH_ARTIFACT_NAME,
                "batch_id": batch_id,
                "logical_identity": logical,
                "counts": counts,
                "artifact_hashes": hashes,
                "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    validate_d4d_evidence_batch(output, d4c_audit=d4c_audit)
    return D4DEvidenceBatchResult(batch_id, output, counts, False)


def validate_d4d_evidence_batch(path: Path, *, d4c_audit: Path) -> JsonObject:
    """Recompute request reconciliation and all immutable child identities."""

    manifest = _read_json(path / "manifest.json", "SECURITY_LIFECYCLE_D4D_BATCH_INVALID")
    if (
        manifest.get("schema_version") != D4D_EVIDENCE_BATCH_SCHEMA_VERSION
        or manifest.get("artifact_name") != D4D_EVIDENCE_BATCH_ARTIFACT_NAME
        or manifest.get("batch_id") != path.name
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_D4D_BATCH_INVALID")
    _validate_hash_set(path, manifest, D4D_BATCH_FILES, "SECURITY_LIFECYCLE_D4D_BATCH")
    audit, requests, intervals, boundaries = validate_d4c_request_artifact(d4c_audit)
    resolution = pd.read_parquet(path / "request_resolution.parquet")
    normalized = _attach_request_identity(
        _normalize_request_resolution(resolution.loc[:, list(RESOLUTION_COLUMNS)], requests),
        requests,
        intervals,
        boundaries,
    )
    if _frame_hash(normalized) != _frame_hash(resolution):
        raise DataValidationError("SECURITY_LIFECYCLE_D4D_BATCH_RESOLUTION_MISMATCH")
    repairs = _normalize_repairs(
        pd.read_parquet(path / "provider_source_repairs.parquet"), requests
    )
    lifecycle_payload = _read_json(
        path / "verified_lifecycle_packages.json", "SECURITY_LIFECYCLE_D4D_BATCH_INVALID"
    )
    transition_payload = _read_json(
        path / "verified_transition_packages.json", "SECURITY_LIFECYCLE_D4D_BATCH_INVALID"
    )
    lifecycle = _package_list(lifecycle_payload)
    transitions = _package_list(transition_payload)
    reports_root = path.parent.parent
    for item in lifecycle:
        if item["package_kind"] == "BULK_OFFICIAL_SOURCE":
            validated, _ = validate_bulk_official_source_package(
                reports_root / "security_lifecycle_bulk_source" / str(item["package_id"])
            )
        else:
            validated = validate_official_evidence_package(
                reports_root / "security_lifecycle_evidence" / str(item["package_id"])
            )
        if str(validated["package_hash"]) != str(item["package_hash"]):
            raise DataValidationError("SECURITY_LIFECYCLE_D4D_BATCH_PACKAGE_HASH_MISMATCH")
    for item in transitions:
        _, package_hash = validate_transition_evidence_package(
            reports_root / "security_identity_transition_evidence" / str(item["package_id"])
        )
        if package_hash != str(item["package_hash"]):
            raise DataValidationError("SECURITY_LIFECYCLE_D4D_BATCH_PACKAGE_HASH_MISMATCH")
    _validate_resolution_references(normalized, lifecycle, transitions, repairs)
    remaining = pd.read_parquet(path / "remaining_requests.parquet")
    expected_remaining = normalized[normalized["resolution_status"].ne("RESOLVED")]
    if _frame_hash(remaining) != _frame_hash(expected_remaining):
        raise DataValidationError("SECURITY_LIFECYCLE_D4D_BATCH_REMAINING_MISMATCH")
    logical = cast(JsonObject, manifest.get("logical_identity", {}))
    expected_logical: JsonObject = {
        "schema_version": D4D_EVIDENCE_BATCH_SCHEMA_VERSION,
        "contract_version": D4D_EVIDENCE_BATCH_CONTRACT_VERSION,
        "d4c_audit_id": audit["audit_id"],
        "d4c_audit_manifest_hash": file_sha256(d4c_audit / "manifest.json"),
        "source_scan_id": cast(JsonObject, audit["logical_identity"])["source_scan_id"],
        "source_scan_manifest_hash": cast(JsonObject, audit["logical_identity"])[
            "source_scan_manifest_hash"
        ],
        "request_resolution_hash": _frame_hash(normalized),
        "lifecycle_package_hashes": sorted(str(item["package_hash"]) for item in lifecycle),
        "transition_package_hashes": sorted(str(item["package_hash"]) for item in transitions),
        "provider_source_repairs_hash": _frame_hash(repairs),
    }
    if logical != expected_logical or path.name != (
        f"security_lifecycle_d4d_evidence_batch_{canonical_payload_hash(logical)[:24]}"
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_D4D_BATCH_IDENTITY_MISMATCH")
    summary = _read_json(path / "summary.json", "SECURITY_LIFECYCLE_D4D_BATCH_INVALID")
    statuses = normalized["resolution_status"].astype(str)
    expected_counts = {
        "input_requests": int(len(requests)),
        "input_intervals": int(len(intervals)),
        "resolved": int(statuses.eq("RESOLVED").sum()),
        "partial": int(statuses.eq("PARTIAL").sum()),
        "unresolved": int(statuses.eq("UNRESOLVED").sum()),
        "retrieval_failed": int(statuses.eq("EVIDENCE_RETRIEVAL_FAILED").sum()),
        "verified_lifecycle_packages": len(lifecycle),
        "verified_transition_packages": len(transitions),
        "provider_source_repairs": int(len(repairs)),
        "interval_requests": int(normalized["finding_kind"].eq("MISSING_PRICE_INTERVAL").sum()),
        "boundary_requests": int(normalized["finding_kind"].eq("BOUNDARY_FINDING").sum()),
    }
    if manifest.get("counts") != expected_counts or summary.get("counts") != expected_counts:
        raise DataValidationError("SECURITY_LIFECYCLE_D4D_BATCH_COUNT_MISMATCH")
    return manifest


def validate_d4c_request_artifact(
    path: Path,
) -> tuple[JsonObject, DataFrame, DataFrame, DataFrame]:
    """Validate the immutable schema-7 request queue used by D.4D."""

    manifest = _read_json(path / "manifest.json", "SECURITY_LIFECYCLE_D4C_AUDIT_INVALID")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_name") != D4C_AUDIT_ARTIFACT_NAME
        or manifest.get("audit_id") != path.name
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_D4C_AUDIT_INVALID")
    _validate_hash_set(path, manifest, D4C_AUDIT_FILES, "SECURITY_LIFECYCLE_D4C_AUDIT")
    summary = _read_json(path / "summary.json", "SECURITY_LIFECYCLE_D4C_AUDIT_INVALID")
    requests = pd.read_parquet(path / "official_evidence_requests.parquet")
    intervals = pd.read_parquet(path / "current_unresolved.parquet")
    boundaries = pd.read_parquet(path / "boundary_root_causes.parquet")
    counts = cast(JsonObject, manifest.get("counts", {}))
    if (
        summary.get("audit_id") != path.name
        or summary.get("counts") != counts
        or int(counts.get("official_evidence_requests", -1)) != len(requests)
        or int(counts.get("unresolved_intervals", -1)) != len(intervals)
        or int(counts.get("unresolved_sessions", -1)) != int(intervals["session_count"].sum())
        or int(counts.get("blocking_boundaries", -1)) != len(boundaries)
        or requests["request_id"].duplicated().any()
        or intervals["current_interval_id"].duplicated().any()
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_D4C_AUDIT_RECONCILIATION_FAILED")
    interval_ids = set(intervals["current_interval_id"].astype(str))
    for row in requests.itertuples(index=False):
        interval_id = str(row.current_interval_id).strip()
        if interval_id and interval_id not in interval_ids:
            raise DataValidationError("SECURITY_LIFECYCLE_D4C_AUDIT_REQUEST_ORPHAN")
        if not interval_id:
            matches = boundaries[
                boundaries["canonical_ts_code"].astype(str).eq(str(row.canonical_ts_code))
                & boundaries["trade_date"].astype(str).eq(str(row.start_date))
            ]
            if len(matches) != 1:
                raise DataValidationError("SECURITY_LIFECYCLE_D4C_AUDIT_BOUNDARY_ORPHAN")
    return manifest, requests, intervals, boundaries


def _normalize_request_resolution(frame: DataFrame, requests: DataFrame) -> DataFrame:
    if set(frame.columns) != set(RESOLUTION_COLUMNS):
        raise DataValidationError("SECURITY_LIFECYCLE_D4D_BATCH_RESOLUTION_SCHEMA_INVALID")
    result = frame.loc[:, list(RESOLUTION_COLUMNS)].copy()
    for column in RESOLUTION_COLUMNS:
        result[column] = result[column].fillna("").astype(str).str.strip()
    if (
        result["request_id"].duplicated().any()
        or set(result["request_id"]) != set(requests["request_id"].astype(str))
        or not set(result["resolution_status"]).issubset(RESOLUTION_STATUSES)
        or result["resolution_reason"].eq("").any()
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_D4D_BATCH_REQUEST_RECONCILIATION_FAILED")
    return result.sort_values(by="request_id").reset_index(drop=True)


def _normalize_repairs(frame: DataFrame, requests: DataFrame) -> DataFrame:
    if frame.empty and len(frame.columns) == 0:
        return pd.DataFrame(columns=REPAIR_COLUMNS)
    if set(frame.columns) != set(REPAIR_COLUMNS):
        raise DataValidationError("SECURITY_LIFECYCLE_D4D_BATCH_REPAIR_SCHEMA_INVALID")
    result = frame.loc[:, list(REPAIR_COLUMNS)].copy()
    for column in REPAIR_COLUMNS:
        if column == "row_count":
            continue
        result[column] = result[column].fillna("").astype(str).str.strip()
    result["row_count"] = pd.to_numeric(result["row_count"], errors="raise").astype(int)
    if (
        result["repair_id"].eq("").any()
        or result["repair_id"].duplicated().any()
        or result["request_id"].duplicated().any()
        or not set(result["request_id"]).issubset(set(requests["request_id"].astype(str)))
        or result["row_count"].lt(0).any()
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_D4D_BATCH_REPAIR_INVALID")
    return result.sort_values(by="repair_id").reset_index(drop=True)


def _attach_request_identity(
    resolution: DataFrame, requests: DataFrame, intervals: DataFrame, boundaries: DataFrame
) -> DataFrame:
    request_columns = [
        "request_id",
        "current_interval_id",
        "canonical_ts_code",
        "start_date",
        "end_date",
        "session_count",
        "request_category",
        "known",
        "missing_fact",
        "preferred_official_source",
        "blocking",
    ]
    merged = requests.loc[:, request_columns].merge(
        resolution, on="request_id", how="left", validate="one_to_one"
    )
    interval_ids = set(intervals["current_interval_id"].astype(str))
    kinds: list[str] = []
    finding_ids: list[str] = []
    for row in merged.itertuples(index=False):
        interval_id = str(row.current_interval_id).strip()
        if interval_id:
            if interval_id not in interval_ids:
                raise DataValidationError("SECURITY_LIFECYCLE_D4D_BATCH_INTERVAL_MISMATCH")
            kinds.append("MISSING_PRICE_INTERVAL")
            finding_ids.append(interval_id)
            continue
        match = boundaries[
            boundaries["canonical_ts_code"].astype(str).eq(str(row.canonical_ts_code))
            & boundaries["trade_date"].astype(str).eq(str(row.start_date))
        ]
        if len(match) != 1:
            raise DataValidationError("SECURITY_LIFECYCLE_D4D_BATCH_BOUNDARY_MISMATCH")
        boundary = match.iloc[0]
        kinds.append("BOUNDARY_FINDING")
        finding_ids.append(
            "boundary_finding_"
            + canonical_payload_hash(
                {
                    "canonical_ts_code": str(boundary["canonical_ts_code"]),
                    "trade_date": str(boundary["trade_date"]),
                    "boundary_type": str(boundary["boundary_type"]),
                }
            )[:24]
        )
    merged.insert(1, "finding_kind", kinds)
    merged.insert(2, "finding_id", finding_ids)
    return merged.sort_values("request_id").reset_index(drop=True)


def _validate_resolution_references(
    resolution: DataFrame,
    lifecycle: list[JsonObject],
    transitions: list[JsonObject],
    repairs: DataFrame,
) -> None:
    lifecycle_ids = {str(item["package_id"]) for item in lifecycle}
    transition_ids = {str(item["package_id"]) for item in transitions}
    repair_ids = set(repairs["repair_id"].astype(str)) if not repairs.empty else set()
    for row in resolution.itertuples(index=False):
        referenced_lifecycle = _references(str(row.lifecycle_package_ids))
        referenced_transitions = _references(str(row.transition_package_ids))
        referenced_repairs = _references(str(row.provider_repair_ids))
        if (
            not referenced_lifecycle.issubset(lifecycle_ids)
            or not referenced_transitions.issubset(transition_ids)
            or not referenced_repairs.issubset(repair_ids)
        ):
            raise DataValidationError("SECURITY_LIFECYCLE_D4D_BATCH_PACKAGE_REFERENCE_INVALID")
        if row.resolution_status == "RESOLVED" and not (
            referenced_lifecycle
            or referenced_transitions
            or referenced_repairs
            or row.resolution_reason == "RESOLVED_BY_EXISTING_EVIDENCE"
        ):
            raise DataValidationError("SECURITY_LIFECYCLE_D4D_BATCH_RESOLUTION_UNPROVEN")


def _package_index(items: list[JsonObject]) -> list[JsonObject]:
    return sorted(
        [
            {
                "package_id": str(item["package_id"]),
                "package_hash": str(item["package_hash"]),
                "package_kind": str(item.get("package_kind", "INDIVIDUAL_OFFICIAL_EVIDENCE")),
                "canonical_ts_code": str(
                    item.get("canonical_ts_code", item.get("predecessor_ts_code", ""))
                ),
                "event_type": str(item.get("event_type", item.get("transition_type", ""))),
                "status": str(item.get("evidence_status", item.get("status", "VERIFIED"))),
            }
            for item in items
        ],
        key=lambda item: (str(item["package_id"]), str(item["package_hash"])),
    )


def _validate_lifecycle_package(path: Path) -> JsonObject:
    manifest = _read_json(path / "manifest.json", "SECURITY_LIFECYCLE_D4D_BATCH_PACKAGE_INVALID")
    if manifest.get("artifact_name") == BULK_ARTIFACT_NAME:
        source, rows = validate_bulk_official_source_package(path)
        codes = sorted(set(rows["canonical_ts_code"].astype(str)))
        event_types = sorted(set(rows["event_type"].astype(str)))
        return {
            **source,
            "package_kind": "BULK_OFFICIAL_SOURCE",
            "canonical_ts_code": codes[0] if len(codes) == 1 else "MULTI",
            "event_type": event_types[0] if len(event_types) == 1 else "MULTI",
            "status": "VERIFIED",
        }
    return {
        **validate_official_evidence_package(path),
        "package_kind": "INDIVIDUAL_OFFICIAL_EVIDENCE",
    }


def _package_list(payload: JsonObject) -> list[JsonObject]:
    packages = payload.get("packages")
    if not isinstance(packages, list) or any(not isinstance(item, dict) for item in packages):
        raise DataValidationError("SECURITY_LIFECYCLE_D4D_BATCH_PACKAGE_INDEX_INVALID")
    return cast(list[JsonObject], packages)


def _references(value: str) -> set[str]:
    return {item.strip() for item in value.split(";") if item.strip()}


def _validate_hash_set(
    path: Path, manifest: JsonObject, expected_files: frozenset[str], error_prefix: str
) -> None:
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict) or set(hashes) != expected_files:
        raise DataValidationError(f"{error_prefix}_ARTIFACT_SET_INVALID")
    for name, expected_hash in hashes.items():
        if not isinstance(expected_hash, str) or file_sha256(path / name) != expected_hash:
            raise DataValidationError(f"{error_prefix}_HASH_MISMATCH: {name}")


def _read_json(path: Path, error_code: str) -> JsonObject:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(error_code) from error
    if not isinstance(value, dict):
        raise DataValidationError(error_code)
    return cast(JsonObject, value)


def _frame_hash(frame: DataFrame) -> str:
    columns = sorted(frame.columns)
    working = frame.loc[:, columns].astype(object).where(pd.notna(frame.loc[:, columns]), None)
    rows = working.sort_values(columns, key=lambda values: values.astype(str)).to_dict("records")
    return canonical_payload_hash({"columns": columns, "rows": rows})
