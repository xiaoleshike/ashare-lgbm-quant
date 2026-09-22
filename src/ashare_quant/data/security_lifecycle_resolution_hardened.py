"""D.3.1 lifecycle resolution with session-exact evidence segmentation."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_identity import SecurityIdentityResolver
from ashare_quant.data.security_lifecycle_audit import canonical_payload_hash, file_sha256
from ashare_quant.data.security_lifecycle_official_index import (
    validate_official_lifecycle_index,
)
from ashare_quant.data.security_lifecycle_resolution import (
    validate_lifecycle_resolution_artifact,
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

SCHEMA_VERSION = 1
CONTRACT_VERSION = 2
ARTIFACT_NAME = "security_lifecycle_resolution_hardened"
ARTIFACT_FILES = frozenset(
    {
        "summary.json",
        "evidence_coverage_matrix.parquet",
        "parent_reconciliation.parquet",
        "source_repair_plan.parquet",
        "unresolved_evidence_requests.parquet",
        "official_index_manifest.json",
        "report.md",
    }
)
SOURCE_NO_SUSPEND = "PROVIDER_HAS_NO_SUSPEND_EVIDENCE"
UNRESOLVED = "STILL_UNRESOLVED"


@dataclass(frozen=True, slots=True)
class HardenedResolutionResult:
    """Published D.3.1 session-exact resolution artifact."""

    resolution_id: str
    output_dir: Path
    counts: JsonObject
    idempotent: bool


class HardenedLifecycleResolutionService:
    """Separate source evidence from lifecycle evidence and segment partial repairs."""

    def __init__(
        self,
        *,
        triage_manifest: Path,
        provider_probe_manifest: Path,
        d3_resolution_manifest: Path,
        official_index_manifest: Path,
        raw_root: Path,
        reports_root: Path,
        identity_resolver: SecurityIdentityResolver,
    ) -> None:
        self.triage_manifest_path = triage_manifest
        self.probe_manifest_path = provider_probe_manifest
        self.d3_manifest_path = d3_resolution_manifest
        self.index_manifest_path = official_index_manifest
        self.raw_root = raw_root
        self.reports_root = reports_root
        self.identity = identity_resolver

    def run(self) -> HardenedResolutionResult:
        """Validate lineage, reconcile every session and publish an immutable artifact."""

        triage_manifest = validate_security_lifecycle_triage_artifact(
            self.triage_manifest_path.parent
        )
        probe_manifest = validate_lifecycle_source_probe_artifact(self.probe_manifest_path.parent)
        d3_manifest = validate_lifecycle_resolution_artifact(self.d3_manifest_path.parent)
        index_manifest = validate_official_lifecycle_index(self.index_manifest_path.parent)
        self._validate_lineage(triage_manifest, probe_manifest, d3_manifest, index_manifest)

        triage = pd.read_parquet(self.triage_manifest_path.parent / "unresolved_triage.parquet")
        triage["parent_interval_id"] = triage.apply(_parent_interval_id, axis=1)
        provider = pd.read_parquet(self.probe_manifest_path.parent / "comparison.parquet")
        events = pd.read_parquet(self.index_manifest_path.parent / "official_events.parquet")
        calendar = _open_sessions(self.raw_root / "trade_cal")
        segments, reconciliation = build_resolution_segments(triage, provider, events, calendar)
        repair = _source_repair_plan(segments, provider)
        unresolved = _unresolved_requests(segments, triage)
        counts = _counts(triage, segments, repair, unresolved)

        logical: JsonObject = {
            "schema_version": SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "triage_id": triage_manifest["triage_id"],
            "triage_manifest_hash": file_sha256(self.triage_manifest_path),
            "provider_probe_id": probe_manifest["probe_id"],
            "provider_probe_manifest_hash": file_sha256(self.probe_manifest_path),
            "d3_resolution_id": d3_manifest["resolution_id"],
            "d3_resolution_manifest_hash": file_sha256(self.d3_manifest_path),
            "official_index_id": index_manifest["index_id"],
            "official_index_manifest_hash": file_sha256(self.index_manifest_path),
            "open_session_calendar_hash": _relevant_calendar_hash(triage, calendar),
            "security_identity_mapping_version": self.identity.mapping_version,
            "security_identity_mapping_hash": self.identity.mapping_hash,
        }
        resolution_id = (
            f"security_lifecycle_resolution_hardened_{canonical_payload_hash(logical)[:24]}"
        )
        output = self.reports_root / "security_lifecycle_resolution_hardened" / resolution_id
        if output.exists():
            manifest = validate_hardened_resolution_artifact(output)
            return HardenedResolutionResult(
                resolution_id, output, cast(JsonObject, manifest["counts"]), True
            )
        self._publish(
            output,
            logical,
            counts,
            segments,
            reconciliation,
            repair,
            unresolved,
            index_manifest,
        )
        return HardenedResolutionResult(resolution_id, output, counts, False)

    def _validate_lineage(
        self,
        triage: JsonObject,
        probe: JsonObject,
        d3: JsonObject,
        index: JsonObject,
    ) -> None:
        if probe.get("triage_id") != triage.get("triage_id"):
            raise DataValidationError("SECURITY_LIFECYCLE_HARDENED_TRIAGE_MISMATCH")
        d3_logical = cast(JsonObject, d3.get("logical_identity", {}))
        if d3_logical.get("source_triage_id") != triage.get("triage_id") or d3_logical.get(
            "provider_probe_id"
        ) != probe.get("probe_id"):
            raise DataValidationError("SECURITY_LIFECYCLE_HARDENED_D3_MISMATCH")
        index_logical = cast(JsonObject, index.get("logical_identity", {}))
        if index_logical.get("security_identity_mapping_hash") != self.identity.mapping_hash:
            raise DataValidationError("SECURITY_LIFECYCLE_HARDENED_IDENTITY_MISMATCH")

    def _publish(
        self,
        output: Path,
        logical: JsonObject,
        counts: JsonObject,
        segments: DataFrame,
        reconciliation: DataFrame,
        repair: DataFrame,
        unresolved: DataFrame,
        index_manifest: JsonObject,
    ) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
        try:
            summary = {
                "resolution_id": output.name,
                "counts": counts,
                "authoritative_evidence_ready": bool(
                    counts["still_unresolved"] == 0
                    and counts["evidence_retrieval_failed"] == 0
                    and counts["unverified_required_evidence"] == 0
                ),
                "lifecycle_preflight_status": "BLOCKED",
            }
            atomic_write_json(staging / "summary.json", summary)
            segments.to_parquet(staging / "evidence_coverage_matrix.parquet", index=False)
            reconciliation.to_parquet(staging / "parent_reconciliation.parquet", index=False)
            repair.to_parquet(staging / "source_repair_plan.parquet", index=False)
            unresolved.to_parquet(staging / "unresolved_evidence_requests.parquet", index=False)
            atomic_write_json(staging / "official_index_manifest.json", index_manifest)
            (staging / "report.md").write_text(_report(summary), encoding="utf-8")
            hashes = {name: file_sha256(staging / name) for name in sorted(ARTIFACT_FILES)}
            atomic_write_json(
                staging / "manifest.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_name": ARTIFACT_NAME,
                    "resolution_id": output.name,
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


def build_resolution_segments(
    triage: DataFrame,
    provider: DataFrame,
    official_events: DataFrame,
    open_sessions: tuple[str, ...],
) -> tuple[DataFrame, DataFrame]:
    """Resolve individual sessions, then compress adjacent equal-resolution sessions."""

    calendar = tuple(sorted(set(open_sessions)))
    position = {date: index for index, date in enumerate(calendar)}
    provider_by_id = {str(row.parent_interval_id): row for row in provider.itertuples(index=False)}
    event_coverage = _official_event_coverage(official_events, calendar)
    segment_rows: list[JsonObject] = []
    reconciliation_rows: list[JsonObject] = []
    for parent in triage.sort_values(["canonical_ts_code", "gap_start", "gap_end"]).itertuples(
        index=False
    ):
        parent_id = str(parent.parent_interval_id)
        sessions = tuple(
            date for date in calendar if str(parent.gap_start) <= date <= str(parent.gap_end)
        )
        if len(sessions) != int(str(parent.session_count)):
            raise DataValidationError(
                f"SECURITY_LIFECYCLE_HARDENED_PARENT_SESSION_MISMATCH: {parent_id}"
            )
        provider_row = provider_by_id.get(parent_id)
        source_by_date = _source_evidence_by_date(provider_row)
        daily_rows: list[JsonObject] = []
        for trade_date in sessions:
            source = source_by_date.get(trade_date, _default_source_resolution(provider_row))
            official = event_coverage.get((str(parent.canonical_ts_code), trade_date))
            lifecycle = _lifecycle_resolution(source, official)
            daily_rows.append(
                {
                    "parent_interval_id": parent_id,
                    "canonical_ts_code": str(parent.canonical_ts_code),
                    "exchange": str(parent.exchange),
                    "trade_date": trade_date,
                    "triage_category": str(parent.triage_category),
                    "source_resolution": source["source_resolution"],
                    "lifecycle_resolution": lifecycle,
                    "repair_required": bool(source["repair_required"]),
                    "repair_action": source["repair_action"],
                    "source_ts_code": source["source_ts_code"],
                    "official_index_match": official["row_identity"] if official else "",
                    "official_evidence_package": official["package_id"] if official else "",
                    "evidence_source_category": (
                        official["evidence_source_category"] if official else ""
                    ),
                    "schema_extension_required": (
                        lifecycle == "LIFECYCLE_SCHEMA_EXTENSION_REQUIRED"
                    ),
                    "final_blocking": lifecycle
                    in {"STILL_UNRESOLVED", "LIFECYCLE_SCHEMA_EXTENSION_REQUIRED"},
                }
            )
        compressed = _compress_parent(daily_rows, position)
        segment_rows.extend(compressed)
        reconciliation_rows.append(
            {
                "parent_interval_id": parent_id,
                "canonical_ts_code": str(parent.canonical_ts_code),
                "parent_sessions": len(sessions),
                "child_segments": len(compressed),
                "child_sessions": sum(int(row["session_count"]) for row in compressed),
                "reconciled": sum(int(row["session_count"]) for row in compressed) == len(sessions),
            }
        )
    segments = pd.DataFrame(segment_rows)
    reconciliation = pd.DataFrame(reconciliation_rows)
    _validate_session_reconciliation(triage, segments, reconciliation)
    return segments, reconciliation


def validate_hardened_resolution_artifact(path: Path) -> JsonObject:
    """Validate the exact immutable D.3.1 artifact child set and hashes."""

    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError("SECURITY_LIFECYCLE_HARDENED_MANIFEST_INVALID") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_name") != ARTIFACT_NAME
        or manifest.get("resolution_id") != path.name
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_HARDENED_MANIFEST_INVALID")
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict) or set(hashes) != ARTIFACT_FILES:
        raise DataValidationError("SECURITY_LIFECYCLE_HARDENED_ARTIFACT_SET_INVALID")
    for name, expected in hashes.items():
        if not isinstance(expected, str) or file_sha256(path / name) != expected:
            raise DataValidationError(f"SECURITY_LIFECYCLE_HARDENED_HASH_MISMATCH: {name}")
    return cast(JsonObject, manifest)


def _source_evidence_by_date(provider: object | None) -> dict[str, JsonObject]:
    if provider is None:
        return {}
    result: dict[str, JsonObject] = {}
    for field, resolution, action in (
        ("missing_suspend_rows", "LOCAL_SUSPEND_D_INCOMPLETE", "REINGEST_SUSPEND_D"),
        ("missing_daily_rows", "LOCAL_DAILY_INCOMPLETE", "REINGEST_DAILY"),
    ):
        for row in json.loads(str(getattr(provider, field))):
            date = str(row["trade_date"])
            candidate = {
                "source_resolution": resolution,
                "repair_required": True,
                "repair_action": action,
                "source_ts_code": str(row.get("source_ts_code", "")),
            }
            existing = result.get(date)
            if existing and existing["source_resolution"] != resolution:
                candidate["source_resolution"] = "LOCAL_BOTH_INCOMPLETE"
                candidate["repair_action"] = "REINGEST_BOTH"
                candidate["source_ts_code"] = ";".join(
                    sorted(
                        {
                            str(existing["source_ts_code"]),
                            str(candidate["source_ts_code"]),
                        }
                    )
                )
            result[date] = candidate
    return result


def _default_source_resolution(provider: object | None) -> JsonObject:
    provider_row = cast(Any, provider)
    status = "NOT_PROBED" if provider is None else str(provider_row.source_completeness_status)
    if status.startswith("LOCAL_"):
        status = "PROVIDER_EVIDENCE_ABSENT_FOR_SESSION"
    return {
        "source_resolution": status,
        "repair_required": False,
        "repair_action": "",
        "source_ts_code": "",
    }


def _lifecycle_resolution(source: JsonObject, official: JsonObject | None) -> str:
    if official is not None:
        event_type = str(official["event_type"])
        if event_type == "ORDINARY_FULL_DAY_SUSPENSION":
            return "VERIFIED_ORDINARY_SUSPENSION"
        if event_type in {"FORMAL_LISTING_SUSPENSION_START", "FORMAL_LISTING_RESUMPTION"}:
            return "VERIFIED_LISTING_SUSPENSION"
        if event_type == "TERMINAL_DELISTING":
            return "VERIFIED_TERMINAL_LIFECYCLE"
        if event_type == "MERGER_TERMINATION":
            return "VERIFIED_CORPORATE_ACTION"
        if event_type == "SECURITY_CODE_TRANSITION":
            return "VERIFIED_SECURITY_IDENTITY"
        return "LIFECYCLE_SCHEMA_EXTENSION_REQUIRED"
    source_resolution = str(source["source_resolution"])
    if source_resolution in {"LOCAL_SUSPEND_D_INCOMPLETE", "LOCAL_BOTH_INCOMPLETE"}:
        return "RESOLVED_LOCAL_SUSPEND_D_GAP"
    if source_resolution == "LOCAL_DAILY_INCOMPLETE":
        return "RESOLVED_LOCAL_DAILY_GAP"
    return UNRESOLVED


def _official_event_coverage(
    events: DataFrame, calendar: tuple[str, ...]
) -> dict[tuple[str, str], JsonObject]:
    if events.empty:
        return {}
    priority = {
        "EXISTING_V3": 6,
        "FORMAL_LISTING_SUSPENSION_START": 5,
        "TERMINAL_DELISTING": 4,
        "MERGER_TERMINATION": 3,
        "SECURITY_CODE_TRANSITION": 3,
        "ORDINARY_FULL_DAY_SUSPENSION": 2,
        "OTHER_LIFECYCLE_EVENT": 1,
    }
    rows = cast(list[JsonObject], events.to_dict("records"))
    closures: dict[str, list[str]] = {}
    for row in rows:
        if row["event_type"] in {"FORMAL_LISTING_RESUMPTION", "TERMINAL_DELISTING"}:
            closures.setdefault(str(row["canonical_ts_code"]), []).append(
                str(row["effective_start"])
            )
    result: dict[tuple[str, str], JsonObject] = {}
    for row in rows:
        if str(row.get("status")) != "VERIFIED":
            continue
        code = str(row["canonical_ts_code"])
        start, end = str(row["effective_start"]), str(row["effective_end"])
        event_type = str(row["event_type"])
        if event_type == "FORMAL_LISTING_RESUMPTION":
            continue
        if event_type == "FORMAL_LISTING_SUSPENSION_START" and end == start:
            next_closures = sorted(date for date in closures.get(code, []) if date > start)
            if not next_closures:
                continue
            end = next_closures[0]
            selected_dates = [date for date in calendar if start <= date < end]
        else:
            selected_dates = [date for date in calendar if start <= date <= end]
        for date in selected_dates:
            key = (code, date)
            existing = result.get(key)
            row_priority = priority.get(str(row.get("official_source_type")), 0) + priority.get(
                event_type, 0
            )
            existing_priority = -1
            if existing:
                existing_priority = priority.get(
                    str(existing.get("official_source_type")), 0
                ) + priority.get(str(existing["event_type"]), 0)
                if existing_priority == row_priority and (
                    existing["event_type"] != event_type
                    or existing["row_identity"] != row["row_identity"]
                ):
                    raise DataValidationError(
                        f"SECURITY_LIFECYCLE_OFFICIAL_INDEX_COVERAGE_CONFLICT: {code} {date}"
                    )
            if row_priority > existing_priority:
                result[key] = row
    return result


def _compress_parent(rows: list[JsonObject], position: dict[str, int]) -> list[JsonObject]:
    if not rows:
        return []
    identity_fields = (
        "source_resolution",
        "lifecycle_resolution",
        "repair_required",
        "repair_action",
        "source_ts_code",
        "official_index_match",
        "official_evidence_package",
        "evidence_source_category",
        "schema_extension_required",
        "final_blocking",
    )
    groups: list[list[JsonObject]] = []
    for row in rows:
        if not groups:
            groups.append([row])
            continue
        previous = groups[-1][-1]
        same = all(previous[field] == row[field] for field in identity_fields)
        adjacent = position[str(row["trade_date"])] == position[str(previous["trade_date"])] + 1
        if same and adjacent:
            groups[-1].append(row)
        else:
            groups.append([row])
    result = []
    for group in groups:
        first, last = group[0], group[-1]
        identity = {
            "parent_interval_id": first["parent_interval_id"],
            "segment_start": first["trade_date"],
            "segment_end": last["trade_date"],
            "source_resolution": first["source_resolution"],
            "lifecycle_resolution": first["lifecycle_resolution"],
        }
        result.append(
            {
                **{key: value for key, value in first.items() if key != "trade_date"},
                "resolution_segment_id": (
                    f"resolution_segment_{canonical_payload_hash(identity)[:24]}"
                ),
                "segment_start": first["trade_date"],
                "segment_end": last["trade_date"],
                "session_count": len(group),
            }
        )
    return result


def _validate_session_reconciliation(
    triage: DataFrame, segments: DataFrame, reconciliation: DataFrame
) -> None:
    parent_ids = set(triage["parent_interval_id"].astype(str))
    if (
        set(segments["parent_interval_id"].astype(str)) != parent_ids
        or set(reconciliation["parent_interval_id"].astype(str)) != parent_ids
        or segments["resolution_segment_id"].duplicated().any()
        or not reconciliation["reconciled"].astype(bool).all()
        or int(triage["session_count"].sum()) != int(segments["session_count"].sum())
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_HARDENED_RECONCILIATION_FAILED")


def _source_repair_plan(segments: DataFrame, provider: DataFrame) -> DataFrame:
    provider_hashes: dict[str, str] = {}
    for row in provider.itertuples(index=False):
        provider_hashes[str(row.parent_interval_id)] = canonical_payload_hash(
            {
                "provider_suspend_hash": row.provider_suspend_hash,
                "provider_daily_hash": row.provider_daily_hash,
            }
        )
    rows = []
    repair = segments[segments["repair_required"].astype(bool)]
    for row in repair.itertuples(index=False):
        rows.append(
            {
                "parent_interval_id": row.parent_interval_id,
                "resolution_segment_id": row.resolution_segment_id,
                "dataset": str(row.repair_action).removeprefix("REINGEST_").lower(),
                "canonical_ts_code": row.canonical_ts_code,
                "source_ts_code": row.source_ts_code,
                "start_date": row.segment_start,
                "end_date": row.segment_end,
                "session_count": row.session_count,
                "repair_action": row.repair_action,
                "provider_evidence_hash": provider_hashes.get(str(row.parent_interval_id), ""),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "parent_interval_id",
            "resolution_segment_id",
            "dataset",
            "canonical_ts_code",
            "source_ts_code",
            "start_date",
            "end_date",
            "session_count",
            "repair_action",
            "provider_evidence_hash",
        ],
    )


def _unresolved_requests(segments: DataFrame, triage: DataFrame) -> DataFrame:
    triage_by_id = triage.set_index("parent_interval_id")
    rows = []
    for segment in segments[segments["final_blocking"].astype(bool)].itertuples(index=False):
        parent = triage_by_id.loc[str(segment.parent_interval_id)]
        rows.append(
            {
                "parent_interval_id": segment.parent_interval_id,
                "resolution_segment_id": segment.resolution_segment_id,
                "canonical_ts_code": segment.canonical_ts_code,
                "segment_start": segment.segment_start,
                "segment_end": segment.segment_end,
                "session_count": segment.session_count,
                "known": (
                    f"source={segment.source_resolution}; previous_quote="
                    f"{parent['previous_valid_quote']}; next_quote={parent['next_valid_quote']}"
                ),
                "missing": "verified official lifecycle fact covering every segment session",
                "recommended_official_source": f"{segment.exchange} official disclosures",
                "suggested_search_terms": (
                    f"{segment.canonical_ts_code} {parent['source_aliases']} "
                    "停牌 复牌 暂停上市 恢复上市 终止上市 退市 吸收合并 换股 重大资产重组"
                ).strip(),
                "date_window": f"{segment.segment_start}..{segment.segment_end}",
                "historical_aliases": str(parent["source_aliases"]),
                "lifecycle_resolution": segment.lifecycle_resolution,
            }
        )
    return pd.DataFrame(rows)


def _counts(
    triage: DataFrame,
    segments: DataFrame,
    repair: DataFrame,
    unresolved: DataFrame,
) -> JsonObject:
    by_resolution = {
        str(key): {
            "segments": int(len(group)),
            "sessions": int(group["session_count"].sum()),
            "securities": int(group["canonical_ts_code"].nunique()),
        }
        for key, group in segments.groupby("lifecycle_resolution", sort=True)
    }
    still = segments["lifecycle_resolution"].astype(str).eq(UNRESOLVED)
    return {
        "parent_intervals": int(len(triage)),
        "parent_sessions": int(triage["session_count"].sum()),
        "child_segments": int(len(segments)),
        "child_sessions": int(segments["session_count"].sum()),
        "parents_split": int(segments.groupby("parent_interval_id").size().astype(int).gt(1).sum()),
        "by_lifecycle_resolution": by_resolution,
        "source_repair_segments": int(len(repair)),
        "source_repair_sessions": int(repair["session_count"].sum()) if not repair.empty else 0,
        "still_unresolved": int(still.sum()),
        "still_unresolved_sessions": int(segments.loc[still, "session_count"].sum()),
        "unverified_required_evidence": int(len(unresolved)),
        "evidence_retrieval_failed": 0,
    }


def _open_sessions(path: Path) -> tuple[str, ...]:
    files = sorted(path.glob("**/*.parquet"))
    if not files:
        raise DataValidationError("SECURITY_LIFECYCLE_HARDENED_TRADE_CAL_MISSING")
    escaped = str(path / "**" / "*.parquet").replace("'", "''")
    query = f"""SELECT DISTINCT CAST(cal_date AS VARCHAR) AS trade_date
        FROM read_parquet('{escaped}', union_by_name=true, hive_partitioning=false)
        WHERE CAST(is_open AS INTEGER)=1 ORDER BY trade_date"""  # noqa: S608
    with duckdb.connect() as connection:
        rows = connection.execute(query).fetchall()
    return tuple(str(row[0]) for row in rows)


def _relevant_calendar_hash(triage: DataFrame, calendar: tuple[str, ...]) -> str:
    start = str(triage["gap_start"].astype(str).min())
    end = str(triage["gap_end"].astype(str).max())
    relevant = [date for date in calendar if start <= date <= end]
    return canonical_payload_hash({"start": start, "end": end, "open_sessions": relevant})


def _parent_interval_id(row: pd.Series) -> str:
    identity = {
        "canonical_ts_code": str(row["canonical_ts_code"]),
        "gap_start": str(row["gap_start"]),
        "gap_end": str(row["gap_end"]),
        "session_count": int(row["session_count"]),
    }
    return f"interval_{canonical_payload_hash(identity)[:24]}"


def _report(summary: JsonObject) -> str:
    counts = cast(JsonObject, summary["counts"])
    return "\n".join(
        [
            "# Security Lifecycle D.3.1 Hardened Resolution",
            "",
            f"- resolution_id: `{summary['resolution_id']}`",
            f"- parent_intervals: `{counts['parent_intervals']}`",
            f"- parent_sessions: `{counts['parent_sessions']}`",
            f"- child_segments: `{counts['child_segments']}`",
            f"- authoritative_evidence_ready: `{summary['authoritative_evidence_ready']}`",
            "- lifecycle_preflight_status: `BLOCKED`",
            "",
            "Source completeness and lifecycle resolution are independent fields.",
            "",
        ]
    )
