"""Authoritative evidence packages and D.3 lifecycle residual resolution."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_identity import SecurityIdentityResolver
from ashare_quant.data.security_lifecycle import SecurityLifecycleResolver
from ashare_quant.data.security_lifecycle_audit import (
    LifecycleAuditPolicy,
    canonical_payload_hash,
    file_sha256,
    validate_security_lifecycle_artifact,
)
from ashare_quant.data.security_lifecycle_source_probe import (
    validate_lifecycle_source_probe_artifact,
)
from ashare_quant.data.security_lifecycle_triage import (
    validate_security_lifecycle_triage_artifact,
)
from ashare_quant.utils.manifest import atomic_write_json

type DataFrame = pd.DataFrame
type JsonObject = dict[str, Any]
type EvidenceStatus = Literal["CANDIDATE", "VERIFIED", "REJECTED", "INSUFFICIENT"]

RESOLUTION_SCHEMA_VERSION = 2
RESOLUTION_CONTRACT_VERSION = 3
EVIDENCE_PACKAGE_SCHEMA_VERSION = 1
RESOLUTION_ARTIFACT_NAME = "security_lifecycle_resolution"
EVIDENCE_ARTIFACT_NAME = "security_lifecycle_official_evidence"
RESOLUTION_FILES = frozenset(
    {
        "summary.json",
        "provider_probe_manifest.json",
        "provider_resolution.parquet",
        "official_evidence_index.parquet",
        "interval_resolution.parquet",
        "repair_plan.parquet",
        "lifecycle_event_plan.parquet",
        "unresolved.parquet",
        "report.md",
    }
)
OFFICIAL_HOST_SUFFIXES = ("sse.com.cn", "szse.cn", "bse.cn")


@dataclass(frozen=True, slots=True)
class LifecycleResolutionResult:
    """Immutable D.3 resolution publication."""

    resolution_id: str
    output_dir: Path
    counts: JsonObject
    idempotent: bool


def publish_official_evidence_package(
    *, payload: JsonObject, document: Path, reports_root: Path
) -> Path:
    """Validate and freeze one operator-reviewed official lifecycle document."""

    normalized = _validate_evidence_payload(payload, document)
    logical = _evidence_logical_identity(normalized)
    package_id = f"security_lifecycle_evidence_{canonical_payload_hash(logical)[:24]}"
    output = reports_root / "security_lifecycle_evidence" / package_id
    if output.exists():
        validate_official_evidence_package(output)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{package_id}.staging-"))
    try:
        document_name = f"document{document.suffix.lower()}"
        documents = staging / "documents"
        documents.mkdir()
        shutil.copyfile(document, documents / document_name)
        frozen_payload = {**normalized, "document_path": f"documents/{document_name}"}
        atomic_write_json(staging / "evidence.json", frozen_payload)
        hashes = {
            "evidence.json": file_sha256(staging / "evidence.json"),
            f"documents/{document_name}": file_sha256(documents / document_name),
        }
        atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": EVIDENCE_PACKAGE_SCHEMA_VERSION,
                "artifact_name": EVIDENCE_ARTIFACT_NAME,
                "package_id": package_id,
                "logical_identity": logical,
                "artifact_hashes": hashes,
                "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    validate_official_evidence_package(output)
    return output


def validate_official_evidence_package(path: Path) -> JsonObject:
    """Validate one official evidence package through its document hash."""

    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        evidence = json.loads((path / "evidence.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError("SECURITY_LIFECYCLE_EVIDENCE_PACKAGE_INVALID") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != EVIDENCE_PACKAGE_SCHEMA_VERSION
        or manifest.get("artifact_name") != EVIDENCE_ARTIFACT_NAME
        or manifest.get("package_id") != path.name
        or not isinstance(evidence, dict)
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_EVIDENCE_PACKAGE_INVALID")
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict) or "evidence.json" not in hashes or len(hashes) != 2:
        raise DataValidationError("SECURITY_LIFECYCLE_EVIDENCE_PACKAGE_SET_INVALID")
    for relative, expected in hashes.items():
        child = (path / str(relative)).resolve()
        if path.resolve() not in child.parents or not child.is_file():
            raise DataValidationError("SECURITY_LIFECYCLE_EVIDENCE_PACKAGE_PATH_INVALID")
        if not isinstance(expected, str) or file_sha256(child) != expected:
            raise DataValidationError("SECURITY_LIFECYCLE_EVIDENCE_PACKAGE_HASH_MISMATCH")
    document = path / str(evidence.get("document_path", ""))
    normalized = _validate_evidence_payload(cast(JsonObject, evidence), document)
    if manifest.get("logical_identity") != _evidence_logical_identity(normalized):
        raise DataValidationError("SECURITY_LIFECYCLE_EVIDENCE_PACKAGE_IDENTITY_MISMATCH")
    return {
        **cast(JsonObject, evidence),
        "package_id": path.name,
        "package_hash": file_sha256(path / "manifest.json"),
    }


class SecurityLifecycleResolutionService:
    """Reconcile every D.2 interval without mutating source or lifecycle data."""

    def __init__(
        self,
        *,
        lifecycle_scan_manifest: Path,
        triage_manifest: Path,
        provider_probe_manifest: Path,
        reports_root: Path,
        identity_resolver: SecurityIdentityResolver,
        lifecycle_policy: LifecycleAuditPolicy,
        lifecycle_evidence: SecurityLifecycleResolver,
        official_evidence_packages: tuple[Path, ...] = (),
    ) -> None:
        self.scan_manifest_path = lifecycle_scan_manifest
        self.triage_manifest_path = triage_manifest
        self.probe_manifest_path = provider_probe_manifest
        self.reports_root = reports_root
        self.identity = identity_resolver
        self.policy = lifecycle_policy
        self.lifecycle = lifecycle_evidence
        self.official_paths = official_evidence_packages

    def run(self) -> LifecycleResolutionResult:
        """Publish an exact one-row-per-D.2-interval resolution table."""

        scan = validate_security_lifecycle_artifact(self.scan_manifest_path.parent)
        triage = validate_security_lifecycle_triage_artifact(self.triage_manifest_path.parent)
        probe = validate_lifecycle_source_probe_artifact(self.probe_manifest_path.parent)
        self._validate_lineage(scan, triage, probe)
        triage_rows = pd.read_parquet(
            self.triage_manifest_path.parent / "unresolved_triage.parquet"
        )
        triage_rows["parent_interval_id"] = triage_rows.apply(_interval_id, axis=1)
        provider = pd.read_parquet(self.probe_manifest_path.parent / "comparison.parquet")
        official = [validate_official_evidence_package(path) for path in self.official_paths]
        official_index = _official_index(official)
        resolved = _resolve_intervals(triage_rows, provider, official_index, self.lifecycle)
        _validate_resolution_reconciliation(triage_rows, resolved)
        repair = _repair_plan(resolved)
        event_plan = _event_plan(resolved, official_index)
        unresolved = resolved[resolved["blocking_after_d3"].astype(bool)].copy()
        counts = _resolution_counts(resolved, provider, official_index)
        logical: JsonObject = {
            "resolution_schema_version": RESOLUTION_SCHEMA_VERSION,
            "resolution_contract_version": RESOLUTION_CONTRACT_VERSION,
            "source_scan_id": scan["scan_id"],
            "source_scan_manifest_hash": file_sha256(self.scan_manifest_path),
            "source_triage_id": triage["triage_id"],
            "source_triage_manifest_hash": file_sha256(self.triage_manifest_path),
            "provider_probe_id": probe["probe_id"],
            "provider_probe_manifest_hash": file_sha256(self.probe_manifest_path),
            "security_identity_mapping_hash": self.identity.mapping_hash,
            "lifecycle_policy_hash": self.policy.policy_hash,
            "lifecycle_evidence_hash": self.lifecycle.policy_hash,
            "official_evidence_package_hashes": sorted(
                str(item["package_hash"]) for item in official
            ),
        }
        resolution_id = f"security_lifecycle_resolution_{canonical_payload_hash(logical)[:24]}"
        output = self.reports_root / "security_lifecycle_resolution" / resolution_id
        if output.exists():
            manifest = validate_lifecycle_resolution_artifact(output)
            return LifecycleResolutionResult(
                resolution_id, output, cast(JsonObject, manifest["counts"]), True
            )
        summary: JsonObject = {
            "resolution_id": resolution_id,
            "counts": counts,
            "evidence_resolution_complete": bool(
                counts["still_unresolved"] == 0
                and counts["probe_failed"] == 0
                and counts["unverified_required_evidence"] == 0
            ),
            "lifecycle_preflight_status": "BLOCKED",
            "notice": "Resolution is not repair; no source or universe data was changed.",
        }
        manifest = self._publish(
            output,
            logical,
            summary,
            probe,
            provider,
            official_index,
            resolved,
            repair,
            event_plan,
            unresolved,
        )
        return LifecycleResolutionResult(
            resolution_id, output, cast(JsonObject, manifest["counts"]), False
        )

    def _validate_lineage(self, scan: JsonObject, triage: JsonObject, probe: JsonObject) -> None:
        if triage.get("source_scan_id") != scan.get("scan_id"):
            raise DataValidationError("SECURITY_LIFECYCLE_RESOLUTION_SCAN_MISMATCH")
        if probe.get("triage_id") != triage.get("triage_id"):
            raise DataValidationError("SECURITY_LIFECYCLE_RESOLUTION_TRIAGE_MISMATCH")
        logical = cast(JsonObject, triage.get("logical_identity", {}))
        expected = {
            "security_identity_mapping_hash": self.identity.mapping_hash,
            "lifecycle_policy_hash": self.policy.policy_hash,
            "lifecycle_evidence_hash": self.lifecycle.policy_hash,
        }
        if any(logical.get(key) != value for key, value in expected.items()):
            raise DataValidationError("SECURITY_LIFECYCLE_RESOLUTION_LINEAGE_MISMATCH")

    def _publish(
        self,
        output: Path,
        logical: JsonObject,
        summary: JsonObject,
        probe: JsonObject,
        provider: DataFrame,
        official: DataFrame,
        resolved: DataFrame,
        repair: DataFrame,
        event_plan: DataFrame,
        unresolved: DataFrame,
    ) -> JsonObject:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
        try:
            atomic_write_json(staging / "summary.json", summary)
            atomic_write_json(staging / "provider_probe_manifest.json", probe)
            provider.to_parquet(staging / "provider_resolution.parquet", index=False)
            official.to_parquet(staging / "official_evidence_index.parquet", index=False)
            resolved.to_parquet(staging / "interval_resolution.parquet", index=False)
            repair.to_parquet(staging / "repair_plan.parquet", index=False)
            event_plan.to_parquet(staging / "lifecycle_event_plan.parquet", index=False)
            unresolved.to_parquet(staging / "unresolved.parquet", index=False)
            (staging / "report.md").write_text(_report(summary), encoding="utf-8")
            hashes = {name: file_sha256(staging / name) for name in sorted(RESOLUTION_FILES)}
            manifest: JsonObject = {
                "schema_version": RESOLUTION_SCHEMA_VERSION,
                "artifact_name": RESOLUTION_ARTIFACT_NAME,
                "resolution_id": summary["resolution_id"],
                "logical_identity": logical,
                "counts": summary["counts"],
                "artifact_hashes": hashes,
                "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            atomic_write_json(staging / "manifest.json", manifest)
            os.replace(staging, output)
            return manifest
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def validate_lifecycle_resolution_artifact(path: Path) -> JsonObject:
    """Validate a D.3 resolution artifact and its exact child set."""

    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError("SECURITY_LIFECYCLE_RESOLUTION_MANIFEST_INVALID") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != RESOLUTION_SCHEMA_VERSION
        or manifest.get("artifact_name") != RESOLUTION_ARTIFACT_NAME
        or manifest.get("resolution_id") != path.name
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_RESOLUTION_MANIFEST_INVALID")
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict) or set(hashes) != RESOLUTION_FILES:
        raise DataValidationError("SECURITY_LIFECYCLE_RESOLUTION_ARTIFACT_SET_INVALID")
    for name, expected in hashes.items():
        if not isinstance(expected, str) or file_sha256(path / name) != expected:
            raise DataValidationError(f"SECURITY_LIFECYCLE_RESOLUTION_HASH_MISMATCH: {name}")
    return cast(JsonObject, manifest)


def _validate_evidence_payload(payload: JsonObject, document: Path) -> JsonObject:
    required = {
        "canonical_ts_code",
        "event_type",
        "effective_start",
        "effective_end",
        "exchange",
        "official_source_type",
        "official_url",
        "official_document_id",
        "publication_date",
        "effective_date",
        "retrieved_at",
        "document_hash",
        "reviewed_fact",
        "evidence_status",
        "document_security_codes",
        "document_effective_dates",
    }
    if not required.issubset(payload):
        raise DataValidationError("SECURITY_LIFECYCLE_OFFICIAL_EVIDENCE_INCOMPLETE")
    status = str(payload["evidence_status"]).upper()
    if status not in {"CANDIDATE", "VERIFIED", "REJECTED", "INSUFFICIENT"}:
        raise DataValidationError("SECURITY_LIFECYCLE_OFFICIAL_EVIDENCE_STATUS_INVALID")
    host = (urlparse(str(payload["official_url"])).hostname or "").lower()
    official = any(
        host == suffix or host.endswith(f".{suffix}") for suffix in OFFICIAL_HOST_SUFFIXES
    )
    if status == "VERIFIED" and not official:
        raise DataValidationError("SECURITY_LIFECYCLE_OFFICIAL_SOURCE_INVALID")
    code = str(payload["canonical_ts_code"]).upper()
    codes = {str(value).upper() for value in cast(list[object], payload["document_security_codes"])}
    dates = {str(value) for value in cast(list[object], payload["document_effective_dates"])}
    if status == "VERIFIED" and (code not in codes or str(payload["effective_date"]) not in dates):
        raise DataValidationError("SECURITY_LIFECYCLE_OFFICIAL_EVIDENCE_FACT_MISMATCH")
    if not document.is_file() or file_sha256(document) != payload["document_hash"]:
        raise DataValidationError("SECURITY_LIFECYCLE_OFFICIAL_EVIDENCE_HASH_MISMATCH")
    return {key: payload[key] for key in sorted(required)}


def _evidence_logical_identity(payload: JsonObject) -> JsonObject:
    return {key: value for key, value in payload.items() if key != "retrieved_at"}


def _official_index(items: list[JsonObject]) -> DataFrame:
    columns = [
        "package_id",
        "package_hash",
        "canonical_ts_code",
        "event_type",
        "effective_start",
        "effective_end",
        "official_source_type",
        "official_url",
        "official_document_id",
        "document_hash",
        "reviewed_fact",
        "evidence_status",
    ]
    rows = [{key: item.get(key) for key in columns} for item in items]
    return pd.DataFrame(rows, columns=columns)


def _resolve_intervals(
    triage: DataFrame,
    provider: DataFrame,
    official: DataFrame,
    lifecycle: SecurityLifecycleResolver,
) -> DataFrame:
    provider_by_id = (
        provider.set_index("parent_interval_id", drop=False) if not provider.empty else None
    )
    rows: list[JsonObject] = []
    for item in triage.sort_values(["canonical_ts_code", "gap_start", "gap_end"]).itertuples(
        index=False
    ):
        interval_id = str(item.parent_interval_id)
        code, start, end = str(item.canonical_ts_code), str(item.gap_start), str(item.gap_end)
        verified = _overlapping_official(official, code, start, end)
        existing = any(
            event.canonical_ts_code == code
            and event.effective_from <= end
            and event.effective_to >= start
            for event in lifecycle.events()
        )
        source_status = None
        repair_action = None
        probe_failed = False
        provider_evidence_hash = None
        expected_provider_rows = 0
        current_local_rows = 0
        source_ts_codes = ""
        provider_explained_sessions = 0
        if provider_by_id is not None and interval_id in provider_by_id.index:
            record = provider_by_id.loc[interval_id]
            if isinstance(record, pd.DataFrame):
                raise DataValidationError("SECURITY_LIFECYCLE_RESOLUTION_DUPLICATE_PROVIDER_ROW")
            source_status = str(record["source_completeness_status"])
            probe_failed = bool(record["probe_failed"])
            provider_evidence_hash = canonical_payload_hash(
                {
                    "provider_suspend_hash": record["provider_suspend_hash"],
                    "provider_daily_hash": record["provider_daily_hash"],
                }
            )
            expected_provider_rows = int(record["provider_full_day_s_rows"]) + int(
                record["provider_valid_daily_rows"]
            )
            current_local_rows = int(record["local_full_day_s_rows"]) + int(
                record["local_valid_daily_rows"]
            )
            provider_explained_sessions = min(
                int(str(item.session_count)),
                int(record["provider_full_day_s_rows"]) + int(record["provider_valid_daily_rows"]),
            )
            missing_rows = json.loads(str(record["missing_suspend_rows"])) + json.loads(
                str(record["missing_daily_rows"])
            )
            source_ts_codes = ";".join(
                sorted(
                    {
                        str(row.get("source_ts_code", ""))
                        for row in missing_rows
                        if row.get("source_ts_code")
                    }
                )
            )
        if not verified.empty:
            event_type = str(verified.iloc[0]["event_type"])
            resolution = (
                "LIFECYCLE_SCHEMA_EXTENSION_REQUIRED"
                if event_type not in {"LISTING_SUSPENDED", "ORDINARY_SUSPENSION"}
                else "RESOLVED_NEW_OFFICIAL_EVIDENCE"
            )
        elif existing:
            resolution = "RESOLVED_EXISTING_V3_EVIDENCE"
        elif source_status in {"LOCAL_SUSPEND_D_INCOMPLETE", "LOCAL_BOTH_INCOMPLETE"}:
            repair_action = (
                "REINGEST_BOTH"
                if source_status == "LOCAL_BOTH_INCOMPLETE"
                else "REINGEST_SUSPEND_D"
            )
            resolution = (
                "RESOLVED_LOCAL_SUSPEND_D_GAP"
                if provider_explained_sessions == int(str(item.session_count))
                else "STILL_UNRESOLVED"
            )
        elif source_status == "LOCAL_DAILY_INCOMPLETE":
            repair_action = "REINGEST_DAILY"
            resolution = (
                "RESOLVED_LOCAL_DAILY_GAP"
                if provider_explained_sessions == int(str(item.session_count))
                else "STILL_UNRESOLVED"
            )
        elif source_status == "PROVIDER_HAS_NO_SUSPEND_EVIDENCE":
            resolution = "CONFIRMED_PROVIDER_NO_SUSPEND_EVIDENCE"
        else:
            resolution = "STILL_UNRESOLVED"
        blocking = resolution not in {
            "RESOLVED_EXISTING_V3_EVIDENCE",
            "RESOLVED_NEW_OFFICIAL_EVIDENCE",
        }
        rows.append(
            {
                "parent_interval_id": interval_id,
                "canonical_ts_code": code,
                "gap_start": start,
                "gap_end": end,
                "session_count": int(str(item.session_count)),
                "triage_category": str(item.triage_category),
                "source_completeness_status": source_status,
                "resolution": resolution,
                "repair_action": repair_action,
                "repair_required": repair_action is not None,
                "expected_provider_rows": expected_provider_rows,
                "current_local_rows": current_local_rows,
                "provider_evidence_hash": provider_evidence_hash,
                "source_ts_codes": source_ts_codes,
                "provider_explained_sessions": provider_explained_sessions,
                "unexplained_sessions_after_probe": max(
                    0, int(str(item.session_count)) - provider_explained_sessions
                ),
                "probe_failed": probe_failed,
                "unverified_required_evidence": resolution
                in {"CONFIRMED_PROVIDER_NO_SUSPEND_EVIDENCE", "STILL_UNRESOLVED"},
                "blocking_after_d3": blocking,
                "official_package_ids": (
                    ";".join(verified["package_id"].astype(str)) if "package_id" in verified else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def _overlapping_official(frame: DataFrame, code: str, start: str, end: str) -> DataFrame:
    if frame.empty:
        return frame.copy()
    return frame[
        frame["canonical_ts_code"].astype(str).eq(code)
        & frame["evidence_status"].astype(str).eq("VERIFIED")
        & frame["effective_start"].astype(str).le(end)
        & frame["effective_end"].astype(str).ge(start)
    ].copy()


def _validate_resolution_reconciliation(source: DataFrame, resolved: DataFrame) -> None:
    if (
        len(source) != len(resolved)
        or resolved["parent_interval_id"].duplicated().any()
        or set(source["parent_interval_id"]) != set(resolved["parent_interval_id"])
        or int(source["session_count"].sum()) != int(resolved["session_count"].sum())
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_RESOLUTION_RECONCILIATION_FAILED")


def _repair_plan(resolved: DataFrame) -> DataFrame:
    columns = [
        "parent_interval_id",
        "dataset",
        "canonical_ts_code",
        "source_ts_codes",
        "start_date",
        "end_date",
        "session_count",
        "expected_provider_rows",
        "current_local_rows",
        "provider_evidence_hash",
        "provider_explained_sessions",
        "unexplained_sessions_after_probe",
        "repair_action",
        "source_completeness_status",
    ]
    rows = []
    for row in resolved[resolved["repair_required"].astype(bool)].itertuples(index=False):
        dataset = (
            "suspend_d,daily"
            if row.repair_action == "REINGEST_BOTH"
            else str(row.repair_action).removeprefix("REINGEST_").lower()
        )
        rows.append(
            {
                "parent_interval_id": row.parent_interval_id,
                "dataset": dataset,
                "canonical_ts_code": row.canonical_ts_code,
                "source_ts_codes": row.source_ts_codes,
                "start_date": row.gap_start,
                "end_date": row.gap_end,
                "session_count": row.session_count,
                "expected_provider_rows": row.expected_provider_rows,
                "current_local_rows": row.current_local_rows,
                "provider_evidence_hash": row.provider_evidence_hash,
                "provider_explained_sessions": row.provider_explained_sessions,
                "unexplained_sessions_after_probe": row.unexplained_sessions_after_probe,
                "repair_action": row.repair_action,
                "source_completeness_status": row.source_completeness_status,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _event_plan(resolved: DataFrame, official: DataFrame) -> DataFrame:
    columns = [
        "package_id",
        "canonical_ts_code",
        "event_type",
        "effective_start",
        "effective_end",
        "package_hash",
        "new_event_required",
    ]
    if official.empty:
        return pd.DataFrame(columns=columns)
    used = {
        item for value in resolved["official_package_ids"] for item in str(value).split(";") if item
    }
    result = official[official["package_id"].astype(str).isin(used)].copy()
    result["new_event_required"] = True
    return result.reindex(columns=columns)


def _resolution_counts(resolved: DataFrame, provider: DataFrame, official: DataFrame) -> JsonObject:
    resolutions = resolved["resolution"].astype(str)
    statuses = (
        provider["source_completeness_status"].astype(str)
        if not provider.empty
        else pd.Series(dtype="string")
    )
    return {
        "intervals": int(len(resolved)),
        "sessions": int(resolved["session_count"].sum()),
        "securities": int(resolved["canonical_ts_code"].nunique()),
        "by_resolution": {
            str(key): int(value) for key, value in resolutions.value_counts().sort_index().items()
        },
        "by_provider_status": {
            str(key): int(value) for key, value in statuses.value_counts().sort_index().items()
        },
        "repair_required": int(resolved["repair_required"].astype(bool).sum()),
        "probe_failed": int(resolved["probe_failed"].astype(bool).sum()),
        "verified_official_evidence": int(
            official["evidence_status"].astype(str).eq("VERIFIED").sum()
        )
        if not official.empty
        else 0,
        "still_unresolved": int(resolutions.eq("STILL_UNRESOLVED").sum()),
        "unverified_required_evidence": int(
            resolved["unverified_required_evidence"].astype(bool).sum()
        ),
        "blocking_after_d3": int(resolved["blocking_after_d3"].astype(bool).sum()),
    }


def _interval_id(row: pd.Series) -> str:
    identity = {
        "canonical_ts_code": str(row["canonical_ts_code"]),
        "gap_start": str(row["gap_start"]),
        "gap_end": str(row["gap_end"]),
        "session_count": int(str(row["session_count"])),
    }
    return f"interval_{canonical_payload_hash(identity)[:24]}"


def _report(summary: JsonObject) -> str:
    counts = cast(JsonObject, summary["counts"])
    return "\n".join(
        [
            "# Security Lifecycle D.3 Resolution",
            "",
            f"- resolution_id: `{summary['resolution_id']}`",
            f"- intervals: `{counts['intervals']}`",
            f"- sessions: `{counts['sessions']}`",
            f"- evidence_resolution_complete: `{summary['evidence_resolution_complete']}`",
            "- lifecycle_preflight_status: `BLOCKED`",
            "",
            "Resolution records source/evidence truth only. It does not repair source data.",
            "",
        ]
    )
