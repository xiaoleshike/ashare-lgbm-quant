"""Final current-contract lifecycle evidence batch for D.4E."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_lifecycle_audit import (
    canonical_payload_hash,
    file_sha256,
    validate_security_lifecycle_artifact,
)
from ashare_quant.data.security_lifecycle_official_index import (
    validate_bulk_official_source_package,
)
from ashare_quant.data.security_lifecycle_resolution import validate_official_evidence_package
from ashare_quant.data.security_listing_metadata import validate_listing_metadata_evidence
from ashare_quant.utils.manifest import atomic_write_json

type DataFrame = pd.DataFrame
type JsonObject = dict[str, Any]

SCHEMA_VERSION = 1
CONTRACT_VERSION = "security_lifecycle_d4e_evidence_batch_v1"
ARTIFACT_NAME = "security_lifecycle_d4e_evidence_batch"
QUEUE_ARTIFACT_NAME = "security_lifecycle_d4d_residual_queue"
FINAL_QUEUE_ARTIFACT_NAME = "security_lifecycle_d4e_final_residual_queue"
STATUSES = frozenset(
    {"VERIFIED_RESOLVED", "VERIFIED_PARTIAL", "STILL_UNRESOLVED", "EVIDENCE_RETRIEVAL_FAILED"}
)
FILES = frozenset(
    {
        "request_resolution.parquet",
        "verified_lifecycle_packages.json",
        "verified_listing_metadata_packages.json",
        "verified_transition_packages.json",
        "remaining_requests.parquet",
        "summary.json",
        "report.md",
    }
)
RESOLUTION_COLUMNS = (
    "request_id",
    "resolution_status",
    "resolution_reason",
    "lifecycle_package_ids",
    "listing_metadata_package_ids",
    "transition_package_ids",
    "reviewed_fact",
    "notes",
)
FINAL_DETAIL_COLUMNS = (
    "current_interval_id",
    "official_sources_searched",
    "documents_reviewed",
    "why_evidence_insufficient",
    "next_operator_action",
    "final_classification",
)
FINAL_CLASSIFICATIONS = frozenset(
    {
        "ONLINE_OFFICIAL_EVIDENCE_EXHAUSTED",
        "MANUAL_ARCHIVE_RETRIEVAL_REQUIRED",
        "CONTRACT_REVIEW_REQUIRED",
        "SOURCE_REPAIR_REQUIRED",
    }
)


@dataclass(frozen=True, slots=True)
class D4EEvidenceBatchResult:
    """Published final evidence batch."""

    batch_id: str
    output_dir: Path
    counts: JsonObject
    idempotent: bool


@dataclass(frozen=True, slots=True)
class D4EFinalResidualQueueResult:
    """Published unresolved intervals after the final D.4E scan."""

    queue_id: str
    output_dir: Path
    counts: JsonObject
    idempotent: bool


def validate_d4d_residual_queue(path: Path, *, source_scan: Path) -> tuple[JsonObject, DataFrame]:
    """Validate the exact schema-8 residual queue used as D.4E authority."""

    scan = validate_security_lifecycle_artifact(source_scan)
    manifest = _read_json(path / "manifest.json", "SECURITY_LIFECYCLE_D4D_QUEUE_INVALID")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_name") != QUEUE_ARTIFACT_NAME
        or manifest.get("artifact_id") != path.name
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_D4D_QUEUE_INVALID")
    expected_files = {"current_requests.parquet", "summary.json", "report.md"}
    hashes = cast(JsonObject, manifest.get("artifact_hashes", {}))
    if set(hashes) != expected_files:
        raise DataValidationError("SECURITY_LIFECYCLE_D4D_QUEUE_SET_INVALID")
    for name, expected in hashes.items():
        if file_sha256(path / name) != expected:
            raise DataValidationError("SECURITY_LIFECYCLE_D4D_QUEUE_HASH_MISMATCH")
    logical = cast(JsonObject, manifest.get("logical_identity", {}))
    if logical.get("source_scan_id") != scan.get("scan_id") or logical.get(
        "source_scan_manifest_hash"
    ) != file_sha256(source_scan / "manifest.json"):
        raise DataValidationError("SECURITY_LIFECYCLE_D4D_QUEUE_SCAN_MISMATCH")
    requests = pd.read_parquet(path / "current_requests.parquet")
    required = {
        "request_id",
        "finding_kind",
        "finding_id",
        "canonical_ts_code",
        "start_date",
        "end_date",
        "session_count",
        "request_category",
        "blocking",
    }
    counts = cast(JsonObject, manifest.get("counts", {}))
    if (
        not required.issubset(requests.columns)
        or requests["request_id"].duplicated().any()
        or int(counts.get("requests", -1)) != len(requests)
        or int(counts.get("interval_requests", -1))
        != int(requests["finding_kind"].eq("MISSING_PRICE_INTERVAL").sum())
        or int(counts.get("boundary_requests", -1))
        != int(requests["finding_kind"].eq("BOUNDARY_FINDING").sum())
        or int(counts.get("sessions", -1)) != int(requests["session_count"].sum())
        or re.fullmatch(r"[0-9a-f]{64}", str(logical.get("request_content_hash", ""))) is None
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_D4D_QUEUE_RECONCILIATION_FAILED")
    return manifest, requests


def publish_d4e_evidence_batch(
    *,
    source_scan: Path,
    residual_queue: Path,
    request_resolution: DataFrame,
    lifecycle_evidence_packages: tuple[Path, ...],
    listing_metadata_packages: tuple[Path, ...],
    reports_root: Path,
    repository_commit_sha: str,
) -> D4EEvidenceBatchResult:
    """Publish one reconciled batch after every current request has an outcome."""

    queue, requests = validate_d4d_residual_queue(residual_queue, source_scan=source_scan)
    lifecycle = [_lifecycle_package(path) for path in lifecycle_evidence_packages]
    metadata = [validate_listing_metadata_evidence(path) for path in listing_metadata_packages]
    resolution = _normalize_resolution(request_resolution, requests)
    _validate_references(resolution, lifecycle, metadata)
    reconciled = requests.merge(resolution, on="request_id", how="left", validate="one_to_one")
    remaining = reconciled[reconciled["resolution_status"].ne("VERIFIED_RESOLVED")].copy()
    counts: JsonObject = {
        "input_requests": len(requests),
        "input_intervals": int(requests["finding_kind"].eq("MISSING_PRICE_INTERVAL").sum()),
        "input_boundaries": int(requests["finding_kind"].eq("BOUNDARY_FINDING").sum()),
        "resolved": int(resolution["resolution_status"].eq("VERIFIED_RESOLVED").sum()),
        "partial": int(resolution["resolution_status"].eq("VERIFIED_PARTIAL").sum()),
        "unresolved": int(resolution["resolution_status"].eq("STILL_UNRESOLVED").sum()),
        "retrieval_failed": int(
            resolution["resolution_status"].eq("EVIDENCE_RETRIEVAL_FAILED").sum()
        ),
        "fully_resolved_intervals": int(
            (
                reconciled["finding_kind"].eq("MISSING_PRICE_INTERVAL")
                & reconciled["resolution_status"].eq("VERIFIED_RESOLVED")
            ).sum()
        ),
        "verified_lifecycle_packages": len(lifecycle),
        "verified_listing_metadata_packages": len(metadata),
        "verified_transition_packages": 0,
    }
    lifecycle_index = _package_index(lifecycle)
    metadata_index = _package_index(metadata)
    logical = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "source_scan_id": cast(JsonObject, queue["logical_identity"])["source_scan_id"],
        "source_scan_manifest_hash": cast(JsonObject, queue["logical_identity"])[
            "source_scan_manifest_hash"
        ],
        "residual_queue_id": residual_queue.name,
        "residual_queue_manifest_hash": file_sha256(residual_queue / "manifest.json"),
        "repository_commit_sha": repository_commit_sha,
        "request_resolution_hash": _frame_hash(reconciled),
        "lifecycle_package_hashes": sorted(str(item["package_hash"]) for item in lifecycle_index),
        "listing_metadata_package_hashes": sorted(
            str(item["package_hash"]) for item in metadata_index
        ),
        "transition_package_hashes": [],
    }
    batch_id = f"security_lifecycle_d4e_evidence_batch_{canonical_payload_hash(logical)[:24]}"
    output = reports_root / "security_lifecycle_d4e_evidence_batch" / batch_id
    if output.exists():
        manifest = validate_d4e_evidence_batch(
            output, source_scan=source_scan, residual_queue=residual_queue
        )
        return D4EEvidenceBatchResult(batch_id, output, cast(JsonObject, manifest["counts"]), True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{batch_id}.staging-"))
    try:
        reconciled.to_parquet(staging / "request_resolution.parquet", index=False)
        atomic_write_json(
            staging / "verified_lifecycle_packages.json", {"packages": lifecycle_index}
        )
        atomic_write_json(
            staging / "verified_listing_metadata_packages.json", {"packages": metadata_index}
        )
        atomic_write_json(staging / "verified_transition_packages.json", {"packages": []})
        remaining.to_parquet(staging / "remaining_requests.parquet", index=False)
        atomic_write_json(staging / "summary.json", {"batch_id": batch_id, "counts": counts})
        (staging / "report.md").write_text(_report(batch_id, counts), encoding="utf-8")
        atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_name": ARTIFACT_NAME,
                "batch_id": batch_id,
                "logical_identity": logical,
                "counts": counts,
                "artifact_hashes": {name: file_sha256(staging / name) for name in sorted(FILES)},
            },
        )
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    validate_d4e_evidence_batch(output, source_scan=source_scan, residual_queue=residual_queue)
    return D4EEvidenceBatchResult(batch_id, output, counts, False)


def validate_d4e_evidence_batch(
    path: Path, *, source_scan: Path, residual_queue: Path
) -> JsonObject:
    """Validate D.4E request reconciliation and all recursive evidence."""

    _, requests = validate_d4d_residual_queue(residual_queue, source_scan=source_scan)
    manifest = _read_json(path / "manifest.json", "SECURITY_LIFECYCLE_D4E_BATCH_INVALID")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_name") != ARTIFACT_NAME
        or manifest.get("batch_id") != path.name
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_D4E_BATCH_INVALID")
    hashes = cast(JsonObject, manifest.get("artifact_hashes", {}))
    if set(hashes) != FILES:
        raise DataValidationError("SECURITY_LIFECYCLE_D4E_BATCH_SET_INVALID")
    for name, expected in hashes.items():
        if file_sha256(path / name) != expected:
            raise DataValidationError("SECURITY_LIFECYCLE_D4E_BATCH_HASH_MISMATCH")
    resolution = pd.read_parquet(path / "request_resolution.parquet")
    normalized = _normalize_resolution(resolution.loc[:, list(RESOLUTION_COLUMNS)], requests)
    expected = requests.merge(normalized, on="request_id", how="left", validate="one_to_one")
    if _frame_hash(expected) != _frame_hash(resolution):
        raise DataValidationError("SECURITY_LIFECYCLE_D4E_BATCH_RESOLUTION_MISMATCH")
    lifecycle = _package_list(
        _read_json(
            path / "verified_lifecycle_packages.json", "SECURITY_LIFECYCLE_D4E_BATCH_INVALID"
        )
    )
    metadata = _package_list(
        _read_json(
            path / "verified_listing_metadata_packages.json",
            "SECURITY_LIFECYCLE_D4E_BATCH_INVALID",
        )
    )
    reports_root = path.parent.parent
    for item in lifecycle:
        validated = _lifecycle_package(
            reports_root / str(item["package_dir"]) / str(item["package_id"])
        )
        if validated["package_hash"] != item["package_hash"]:
            raise DataValidationError("SECURITY_LIFECYCLE_D4E_BATCH_PACKAGE_MISMATCH")
    for item in metadata:
        validated = validate_listing_metadata_evidence(
            reports_root / "security_listing_metadata_evidence" / str(item["package_id"])
        )
        if validated["package_hash"] != item["package_hash"]:
            raise DataValidationError("SECURITY_LIFECYCLE_D4E_BATCH_PACKAGE_MISMATCH")
    _validate_references(normalized, lifecycle, metadata)
    logical = cast(JsonObject, manifest.get("logical_identity", {}))
    expected_logical = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "source_scan_id": validate_security_lifecycle_artifact(source_scan)["scan_id"],
        "source_scan_manifest_hash": file_sha256(source_scan / "manifest.json"),
        "residual_queue_id": residual_queue.name,
        "residual_queue_manifest_hash": file_sha256(residual_queue / "manifest.json"),
        "repository_commit_sha": logical.get("repository_commit_sha"),
        "request_resolution_hash": _frame_hash(expected),
        "lifecycle_package_hashes": sorted(str(item["package_hash"]) for item in lifecycle),
        "listing_metadata_package_hashes": sorted(str(item["package_hash"]) for item in metadata),
        "transition_package_hashes": [],
    }
    if (
        logical != expected_logical
        or path.name
        != f"security_lifecycle_d4e_evidence_batch_{canonical_payload_hash(logical)[:24]}"
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_D4E_BATCH_IDENTITY_MISMATCH")
    remaining = pd.read_parquet(path / "remaining_requests.parquet")
    expected_remaining = expected[expected["resolution_status"].ne("VERIFIED_RESOLVED")]
    if _frame_hash(remaining) != _frame_hash(expected_remaining):
        raise DataValidationError("SECURITY_LIFECYCLE_D4E_BATCH_REMAINING_MISMATCH")
    return manifest


def publish_d4e_final_residual_queue(
    *,
    source_scan: Path,
    evidence_batch: Path,
    residual_details: DataFrame,
    reports_root: Path,
) -> D4EFinalResidualQueueResult:
    """Publish exact operator actions for every blocker in the final scan."""

    scan = validate_security_lifecycle_artifact(source_scan)
    unresolved = pd.read_parquet(source_scan / "unresolved.parquet")
    details = _normalize_final_details(residual_details, unresolved, str(scan["scan_id"]))
    current = _current_residuals(unresolved, str(scan["scan_id"]))
    reconciled = current.merge(details, on="current_interval_id", validate="one_to_one")
    logical = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": "security_lifecycle_d4e_final_residual_queue_v1",
        "source_scan_id": scan["scan_id"],
        "source_scan_manifest_hash": file_sha256(source_scan / "manifest.json"),
        "evidence_batch_id": evidence_batch.name,
        "evidence_batch_manifest_hash": file_sha256(evidence_batch / "manifest.json"),
        "residual_content_hash": _frame_hash(reconciled),
    }
    queue_id = f"{FINAL_QUEUE_ARTIFACT_NAME}_{canonical_payload_hash(logical)[:24]}"
    output = reports_root / FINAL_QUEUE_ARTIFACT_NAME / queue_id
    counts: JsonObject = {
        "intervals": len(reconciled),
        "sessions": int(reconciled["session_count"].sum()),
        "securities": int(reconciled["canonical_ts_code"].nunique()),
    }
    if output.exists():
        manifest = validate_d4e_final_residual_queue(
            output, source_scan=source_scan, evidence_batch=evidence_batch
        )
        return D4EFinalResidualQueueResult(
            queue_id, output, cast(JsonObject, manifest["counts"]), True
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{queue_id}.staging-"))
    try:
        reconciled.to_parquet(staging / "current_requests.parquet", index=False)
        atomic_write_json(staging / "summary.json", {"queue_id": queue_id, "counts": counts})
        (staging / "report.md").write_text(
            _final_queue_report(queue_id, counts, reconciled), encoding="utf-8"
        )
        files = {"current_requests.parquet", "summary.json", "report.md"}
        atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_name": FINAL_QUEUE_ARTIFACT_NAME,
                "queue_id": queue_id,
                "logical_identity": logical,
                "counts": counts,
                "artifact_hashes": {name: file_sha256(staging / name) for name in sorted(files)},
            },
        )
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    validate_d4e_final_residual_queue(
        output, source_scan=source_scan, evidence_batch=evidence_batch
    )
    return D4EFinalResidualQueueResult(queue_id, output, counts, False)


def validate_d4e_final_residual_queue(
    path: Path, *, source_scan: Path, evidence_batch: Path
) -> JsonObject:
    """Validate final blocker reconciliation and immutable source identities."""

    scan = validate_security_lifecycle_artifact(source_scan)
    manifest = _read_json(path / "manifest.json", "SECURITY_LIFECYCLE_D4E_FINAL_QUEUE_INVALID")
    files = {"current_requests.parquet", "summary.json", "report.md"}
    hashes = cast(JsonObject, manifest.get("artifact_hashes", {}))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_name") != FINAL_QUEUE_ARTIFACT_NAME
        or manifest.get("queue_id") != path.name
        or set(hashes) != files
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_D4E_FINAL_QUEUE_INVALID")
    if any(file_sha256(path / name) != expected for name, expected in hashes.items()):
        raise DataValidationError("SECURITY_LIFECYCLE_D4E_FINAL_QUEUE_HASH_MISMATCH")
    unresolved = pd.read_parquet(source_scan / "unresolved.parquet")
    expected = _current_residuals(unresolved, str(scan["scan_id"]))
    actual = pd.read_parquet(path / "current_requests.parquet")
    details = _normalize_final_details(
        actual.loc[:, list(FINAL_DETAIL_COLUMNS)], unresolved, str(scan["scan_id"])
    )
    recomputed = expected.merge(details, on="current_interval_id", validate="one_to_one")
    if _frame_hash(actual) != _frame_hash(recomputed):
        raise DataValidationError("SECURITY_LIFECYCLE_D4E_FINAL_QUEUE_RECONCILIATION_FAILED")
    logical = cast(JsonObject, manifest.get("logical_identity", {}))
    expected_logical = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": "security_lifecycle_d4e_final_residual_queue_v1",
        "source_scan_id": scan["scan_id"],
        "source_scan_manifest_hash": file_sha256(source_scan / "manifest.json"),
        "evidence_batch_id": evidence_batch.name,
        "evidence_batch_manifest_hash": file_sha256(evidence_batch / "manifest.json"),
        "residual_content_hash": _frame_hash(recomputed),
    }
    if logical != expected_logical or path.name != (
        f"{FINAL_QUEUE_ARTIFACT_NAME}_{canonical_payload_hash(logical)[:24]}"
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_D4E_FINAL_QUEUE_IDENTITY_MISMATCH")
    counts = {
        "intervals": len(recomputed),
        "sessions": int(recomputed["session_count"].sum()),
        "securities": int(recomputed["canonical_ts_code"].nunique()),
    }
    summary = _read_json(path / "summary.json", "SECURITY_LIFECYCLE_D4E_FINAL_QUEUE_INVALID")
    if manifest.get("counts") != counts or summary != {"queue_id": path.name, "counts": counts}:
        raise DataValidationError("SECURITY_LIFECYCLE_D4E_FINAL_QUEUE_COUNT_MISMATCH")
    return manifest


def _normalize_resolution(frame: DataFrame, requests: DataFrame) -> DataFrame:
    if set(frame.columns) != set(RESOLUTION_COLUMNS):
        raise DataValidationError("SECURITY_LIFECYCLE_D4E_BATCH_RESOLUTION_SCHEMA_INVALID")
    result = frame.loc[:, list(RESOLUTION_COLUMNS)].copy()
    for column in RESOLUTION_COLUMNS:
        result[column] = result[column].fillna("").astype(str)
    if (
        result["request_id"].duplicated().any()
        or set(result["request_id"]) != set(requests["request_id"].astype(str))
        or not set(result["resolution_status"]).issubset(STATUSES)
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_D4E_BATCH_REQUEST_RECONCILIATION_FAILED")
    return result.sort_values("request_id").reset_index(drop=True)


def _current_residuals(unresolved: DataFrame, scan_id: str) -> DataFrame:
    required = {"canonical_ts_code", "gap_start", "gap_end", "session_count", "reason"}
    if not required.issubset(unresolved.columns):
        raise DataValidationError("SECURITY_LIFECYCLE_D4E_FINAL_QUEUE_SOURCE_INVALID")
    result = unresolved.loc[:, sorted(required)].copy()
    result.insert(
        0,
        "current_interval_id",
        [
            "current_interval_"
            + canonical_payload_hash(
                {
                    "scan_id": scan_id,
                    "canonical_ts_code": str(row.canonical_ts_code),
                    "gap_start": str(row.gap_start),
                    "gap_end": str(row.gap_end),
                }
            )[:24]
            for row in result.itertuples(index=False)
        ],
    )
    if result["current_interval_id"].duplicated().any():
        raise DataValidationError("SECURITY_LIFECYCLE_D4E_FINAL_QUEUE_SOURCE_INVALID")
    return result.sort_values("current_interval_id").reset_index(drop=True)


def _normalize_final_details(frame: DataFrame, unresolved: DataFrame, scan_id: str) -> DataFrame:
    if set(frame.columns) != set(FINAL_DETAIL_COLUMNS):
        raise DataValidationError("SECURITY_LIFECYCLE_D4E_FINAL_QUEUE_DETAILS_INVALID")
    result = frame.loc[:, list(FINAL_DETAIL_COLUMNS)].copy()
    for column in FINAL_DETAIL_COLUMNS:
        result[column] = result[column].fillna("").astype(str).str.strip()
    expected_ids = set(_current_residuals(unresolved, scan_id)["current_interval_id"])
    if (
        result["current_interval_id"].duplicated().any()
        or set(result["current_interval_id"]) != expected_ids
        or any(
            result[column].eq("").any()
            for column in FINAL_DETAIL_COLUMNS
            if column != "current_interval_id"
        )
        or not set(result["final_classification"]).issubset(FINAL_CLASSIFICATIONS)
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_D4E_FINAL_QUEUE_DETAILS_INVALID")
    return result.sort_values("current_interval_id").reset_index(drop=True)


def _validate_references(
    resolution: DataFrame, lifecycle: list[JsonObject], metadata: list[JsonObject]
) -> None:
    lifecycle_ids = {str(item["package_id"]) for item in lifecycle}
    metadata_ids = {str(item["package_id"]) for item in metadata}
    for row in resolution.itertuples(index=False):
        lifecycle_refs = _refs(row.lifecycle_package_ids)
        metadata_refs = _refs(row.listing_metadata_package_ids)
        if not lifecycle_refs.issubset(lifecycle_ids) or not metadata_refs.issubset(metadata_ids):
            raise DataValidationError("SECURITY_LIFECYCLE_D4E_BATCH_PACKAGE_REFERENCE_INVALID")
        if row.resolution_status == "VERIFIED_RESOLVED" and not (lifecycle_refs or metadata_refs):
            raise DataValidationError("SECURITY_LIFECYCLE_D4E_BATCH_RESOLUTION_UNPROVEN")


def _lifecycle_package(path: Path) -> JsonObject:
    try:
        manifest = _read_json(path / "manifest.json", "SECURITY_LIFECYCLE_D4E_PACKAGE_INVALID")
        if manifest.get("artifact_name") == "security_lifecycle_bulk_official_source":
            source, _ = validate_bulk_official_source_package(path)
            return {**source, "package_dir": "security_lifecycle_bulk_source"}
        evidence = validate_official_evidence_package(path)
        return {**evidence, "package_dir": "security_lifecycle_evidence"}
    except DataValidationError:
        raise


def _package_index(packages: list[JsonObject]) -> list[JsonObject]:
    return [
        {
            "package_id": item["package_id"],
            "package_hash": item["package_hash"],
            **({"package_dir": item["package_dir"]} if "package_dir" in item else {}),
        }
        for item in packages
    ]


def _package_list(payload: JsonObject) -> list[JsonObject]:
    packages = payload.get("packages")
    if not isinstance(packages, list) or not all(isinstance(item, dict) for item in packages):
        raise DataValidationError("SECURITY_LIFECYCLE_D4E_BATCH_PACKAGE_INDEX_INVALID")
    return cast(list[JsonObject], packages)


def _refs(value: object) -> set[str]:
    return {item for item in str(value).split(",") if item}


def _frame_hash(frame: DataFrame) -> str:
    columns = sorted(frame.columns)
    working = frame.loc[:, columns].astype(object).where(pd.notna(frame.loc[:, columns]), None)
    rows = working.sort_values(columns, key=lambda values: values.astype(str)).to_dict("records")
    return canonical_payload_hash({"columns": columns, "rows": rows})


def _read_json(path: Path, error_code: str) -> JsonObject:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(error_code) from error
    if not isinstance(value, dict):
        raise DataValidationError(error_code)
    return cast(JsonObject, value)


def _report(batch_id: str, counts: JsonObject) -> str:
    return "\n".join(
        [
            "# D.4E Evidence Batch",
            "",
            f"- batch_id: `{batch_id}`",
            f"- input requests: {counts['input_requests']}",
            f"- resolved: {counts['resolved']}",
            f"- partial: {counts['partial']}",
            f"- unresolved: {counts['unresolved']}",
            f"- retrieval failed: {counts['retrieval_failed']}",
            "",
        ]
    )


def _final_queue_report(queue_id: str, counts: JsonObject, rows: DataFrame) -> str:
    lines = [
        "# D.4E Final Residual Queue",
        "",
        f"Queue: `{queue_id}`",
        "",
        f"Intervals: {counts['intervals']}",
        f"Sessions: {counts['sessions']}",
        f"Securities: {counts['securities']}",
        "",
    ]
    for row in rows.sort_values(["canonical_ts_code", "gap_start"]).itertuples(index=False):
        lines.extend(
            [
                f"## {row.canonical_ts_code} {row.gap_start}..{row.gap_end}",
                "",
                f"- Missing fact: {row.reason}",
                f"- Sources searched: {row.official_sources_searched}",
                f"- Documents reviewed: {row.documents_reviewed}",
                f"- Insufficient because: {row.why_evidence_insufficient}",
                f"- Next action: {row.next_operator_action}",
                f"- Classification: {row.final_classification}",
                "",
            ]
        )
    return "\n".join(lines)
