"""D.3.2 residual official-evidence closure over an immutable D.3.1 queue."""

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
from ashare_quant.data.research_source_snapshot import snapshot_contract
from ashare_quant.data.security_lifecycle_audit import canonical_payload_hash, file_sha256
from ashare_quant.data.security_lifecycle_official_index import (
    validate_official_lifecycle_index,
)
from ashare_quant.data.security_lifecycle_resolution_hardened import (
    _open_sessions,
    _parent_interval_id,
    validate_hardened_resolution_artifact,
)
from ashare_quant.data.security_lifecycle_triage import (
    validate_security_lifecycle_triage_artifact,
)
from ashare_quant.utils.manifest import atomic_write_json

type DataFrame = pd.DataFrame
type JsonObject = dict[str, Any]

SCHEMA_VERSION = 2
CONTRACT_VERSION = "security_lifecycle_evidence_closure_v2_blocking_reachability"
ARTIFACT_NAME = "security_lifecycle_evidence_closure"
ARTIFACT_FILES = frozenset(
    {
        "summary.json",
        "evidence_coverage_matrix.parquet",
        "unresolved.parquet",
        "distribution.parquet",
        "official_source_coverage.parquet",
        "h5_reachability.parquet",
        "source_repair_plan.parquet",
        "research_source_snapshot_contract.json",
        "report.md",
    }
)


@dataclass(frozen=True, slots=True)
class EvidenceClosureResult:
    """Published D.3.2 residual evidence result."""

    closure_id: str
    output_dir: Path
    counts: JsonObject
    idempotent: bool


class SecurityLifecycleEvidenceClosureService:
    """Reconcile new evidence against the exact immutable D.3.1 unresolved queue."""

    def __init__(
        self,
        *,
        baseline_hardened_manifest: Path,
        current_hardened_manifest: Path,
        official_index_manifest: Path,
        triage_manifest: Path,
        raw_root: Path,
        reports_root: Path,
    ) -> None:
        self.baseline_manifest_path = baseline_hardened_manifest
        self.current_manifest_path = current_hardened_manifest
        self.index_manifest_path = official_index_manifest
        self.triage_manifest_path = triage_manifest
        self.raw_root = raw_root
        self.reports_root = reports_root

    def run(self) -> EvidenceClosureResult:
        """Validate lineage, reconcile all input sessions and publish immutable output."""

        baseline_manifest = validate_hardened_resolution_artifact(
            self.baseline_manifest_path.parent
        )
        current_manifest = validate_hardened_resolution_artifact(self.current_manifest_path.parent)
        index_manifest = validate_official_lifecycle_index(self.index_manifest_path.parent)
        triage_manifest = validate_security_lifecycle_triage_artifact(
            self.triage_manifest_path.parent
        )
        self._validate_lineage(baseline_manifest, current_manifest, index_manifest, triage_manifest)
        baseline = pd.read_parquet(
            self.baseline_manifest_path.parent / "evidence_coverage_matrix.parquet"
        )
        queue = baseline[baseline["lifecycle_resolution"].eq("STILL_UNRESOLVED")].copy()
        current = pd.read_parquet(
            self.current_manifest_path.parent / "evidence_coverage_matrix.parquet"
        )
        calendar = _open_sessions(self.raw_root / "trade_cal")
        coverage = reconcile_evidence_queue(queue, current, calendar)
        triage = pd.read_parquet(self.triage_manifest_path.parent / "unresolved_triage.parquet")
        if "parent_interval_id" not in triage.columns:
            triage["parent_interval_id"] = triage.apply(_parent_interval_id, axis=1)
        events = pd.read_parquet(self.index_manifest_path.parent / "official_events.parquet")
        sources = json.loads(
            (self.index_manifest_path.parent / "source_inventory.json").read_text(encoding="utf-8")
        )
        distribution = unresolved_distribution(queue)
        reachability = h5_reachability_diagnostic(
            coverage[coverage["final_blocking"].astype(bool)].copy(), triage
        )
        unresolved = unresolved_evidence_requests(coverage, triage, sources)
        source_coverage = official_source_coverage(events)
        repair = pd.read_parquet(self.current_manifest_path.parent / "source_repair_plan.parquet")
        counts = closure_counts(queue, coverage, unresolved, reachability)
        logical: JsonObject = {
            "schema_version": SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "baseline_resolution_id": baseline_manifest["resolution_id"],
            "baseline_manifest_hash": file_sha256(self.baseline_manifest_path),
            "current_resolution_id": current_manifest["resolution_id"],
            "current_manifest_hash": file_sha256(self.current_manifest_path),
            "official_index_id": index_manifest["index_id"],
            "official_index_manifest_hash": file_sha256(self.index_manifest_path),
            "triage_id": triage_manifest["triage_id"],
            "triage_manifest_hash": file_sha256(self.triage_manifest_path),
            "input_queue_hash": _frame_hash(queue),
            "coverage_hash": _frame_hash(coverage),
            "source_snapshot_contract": snapshot_contract(),
        }
        closure_id = f"security_lifecycle_evidence_closure_{canonical_payload_hash(logical)[:24]}"
        output = self.reports_root / "security_lifecycle_evidence_closure" / closure_id
        if output.exists():
            manifest = validate_evidence_closure_artifact(output)
            return EvidenceClosureResult(
                closure_id, output, cast(JsonObject, manifest["counts"]), True
            )
        self._publish(
            output,
            logical,
            counts,
            coverage,
            unresolved,
            distribution,
            source_coverage,
            reachability,
            repair,
        )
        return EvidenceClosureResult(closure_id, output, counts, False)

    def _validate_lineage(
        self,
        baseline: JsonObject,
        current: JsonObject,
        index: JsonObject,
        triage: JsonObject,
    ) -> None:
        baseline_logical = cast(JsonObject, baseline.get("logical_identity", {}))
        current_logical = cast(JsonObject, current.get("logical_identity", {}))
        if (
            baseline_logical.get("triage_id") != triage.get("triage_id")
            or current_logical.get("triage_id") != triage.get("triage_id")
            or current_logical.get("official_index_id") != index.get("index_id")
            or baseline_logical.get("provider_probe_id") != current_logical.get("provider_probe_id")
        ):
            raise DataValidationError("SECURITY_LIFECYCLE_EVIDENCE_CLOSURE_LINEAGE_MISMATCH")

    def _publish(
        self,
        output: Path,
        logical: JsonObject,
        counts: JsonObject,
        coverage: DataFrame,
        unresolved: DataFrame,
        distribution: DataFrame,
        source_coverage: DataFrame,
        reachability: DataFrame,
        repair: DataFrame,
    ) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
        try:
            summary = {
                "closure_id": output.name,
                "counts": counts,
                "d4_readiness": (
                    "AUTHORITATIVE_EVIDENCE_READY"
                    if counts["still_unresolved"] == 0
                    else "AUTHORITATIVE_EVIDENCE_BLOCKED"
                ),
                "lifecycle_preflight_status": "BLOCKED",
                "v4_created": False,
                "v4_completeness_claim": False,
            }
            atomic_write_json(staging / "summary.json", summary)
            coverage.to_parquet(staging / "evidence_coverage_matrix.parquet", index=False)
            unresolved.to_parquet(staging / "unresolved.parquet", index=False)
            distribution.to_parquet(staging / "distribution.parquet", index=False)
            source_coverage.to_parquet(staging / "official_source_coverage.parquet", index=False)
            reachability.to_parquet(staging / "h5_reachability.parquet", index=False)
            repair.to_parquet(staging / "source_repair_plan.parquet", index=False)
            atomic_write_json(
                staging / "research_source_snapshot_contract.json", snapshot_contract()
            )
            (staging / "report.md").write_text(_report(summary), encoding="utf-8")
            hashes = {name: file_sha256(staging / name) for name in sorted(ARTIFACT_FILES)}
            atomic_write_json(
                staging / "manifest.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_name": ARTIFACT_NAME,
                    "closure_id": output.name,
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
        validate_evidence_closure_artifact(output)


def reconcile_evidence_queue(
    queue: DataFrame, current: DataFrame, calendar: tuple[str, ...]
) -> DataFrame:
    """Map every immutable input session exactly once to current child evidence."""

    rows: list[JsonObject] = []
    sessions = tuple(sorted(set(calendar)))
    current_by_parent = {
        str(parent): group.copy()
        for parent, group in current.groupby("parent_interval_id", sort=False)
    }
    for source in queue.sort_values("resolution_segment_id").itertuples(index=False):
        input_dates = [
            date
            for date in sessions
            if str(source.segment_start) <= date <= str(source.segment_end)
        ]
        if len(input_dates) != int(cast(int, source.session_count)):
            raise DataValidationError("SECURITY_LIFECYCLE_EVIDENCE_QUEUE_SESSION_MISMATCH")
        candidates = current_by_parent.get(str(source.parent_interval_id))
        if candidates is None:
            raise DataValidationError("SECURITY_LIFECYCLE_EVIDENCE_QUEUE_PARENT_MISSING")
        for date in input_dates:
            match = candidates[
                candidates["segment_start"].astype(str).le(date)
                & candidates["segment_end"].astype(str).ge(date)
            ]
            if len(match) != 1:
                raise DataValidationError("SECURITY_LIFECYCLE_EVIDENCE_QUEUE_OVERLAP")
            item = cast(JsonObject, match.iloc[0].to_dict())
            rows.append(
                {
                    "input_segment_id": str(source.resolution_segment_id),
                    "parent_interval_id": str(source.parent_interval_id),
                    "current_segment_id": str(item["resolution_segment_id"]),
                    "canonical_ts_code": str(source.canonical_ts_code),
                    "exchange": str(source.exchange),
                    "trade_date": date,
                    "source_resolution": str(item["source_resolution"]),
                    "lifecycle_resolution": str(item["lifecycle_resolution"]),
                    "official_evidence_package": str(item["official_evidence_package"]),
                    "evidence_source_category": str(item["evidence_source_category"]),
                    "repair_required": bool(item["repair_required"]),
                    "final_blocking": bool(item["final_blocking"]),
                }
            )
    daily = pd.DataFrame(rows)
    if (
        len(daily) != int(queue["session_count"].sum())
        or daily.duplicated(["input_segment_id", "trade_date"]).any()
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_EVIDENCE_QUEUE_RECONCILIATION_FAILED")
    group_fields = [
        "input_segment_id",
        "parent_interval_id",
        "current_segment_id",
        "canonical_ts_code",
        "exchange",
        "source_resolution",
        "lifecycle_resolution",
        "official_evidence_package",
        "evidence_source_category",
        "repair_required",
        "final_blocking",
    ]
    result = (
        daily.groupby(group_fields, dropna=False, sort=True)
        .agg(
            segment_start=("trade_date", "min"),
            segment_end=("trade_date", "max"),
            session_count=("trade_date", "size"),
        )
        .reset_index()
    )
    if int(result["session_count"].sum()) != int(queue["session_count"].sum()):
        raise DataValidationError("SECURITY_LIFECYCLE_EVIDENCE_QUEUE_RECONCILIATION_FAILED")
    return result


def unresolved_distribution(queue: DataFrame) -> DataFrame:
    """Describe the immutable work queue before external evidence matching."""

    work = queue.copy()
    count = pd.to_numeric(work["session_count"], errors="raise")
    work["duration_bucket"] = pd.cut(
        count,
        [0, 1, 5, 20, 60, 120, 252, float("inf")],
        labels=["1", "2-5", "6-20", "21-60", "61-120", "121-252", ">252"],
        include_lowest=True,
    ).astype(str)
    year = work["segment_start"].astype(str).str[:4].astype(int)
    work["calendar_year"] = year
    work["regulatory_era"] = pd.cut(
        year,
        [0, 2011, 2020, 9999],
        labels=["HISTORICAL_PRE_2012", "TRANSITION_2012_2020", "POST_REFORM"],
        include_lowest=True,
    ).astype(str)
    return work[
        [
            "resolution_segment_id",
            "canonical_ts_code",
            "exchange",
            "segment_start",
            "segment_end",
            "session_count",
            "duration_bucket",
            "calendar_year",
            "regulatory_era",
            "source_resolution",
        ]
    ].rename(columns={"resolution_segment_id": "input_segment_id"})


def h5_reachability_diagnostic(queue: DataFrame, triage: DataFrame) -> DataFrame:
    """Conservatively prioritize evidence without using labels or future returns."""

    parent = triage.set_index("parent_interval_id")
    rows: list[JsonObject] = []
    for segment in queue.itertuples(index=False):
        context = parent.loc[str(segment.parent_interval_id)]
        segment_id = getattr(
            segment,
            "input_segment_id",
            getattr(segment, "resolution_segment_id", ""),
        )
        list_date = _optional_date(context.get("list_date"))
        delist_date = _optional_date(context.get("delist_date"))
        before_list = bool(list_date and str(segment.segment_end) < list_date)
        after_delist = bool(delist_date and str(segment.segment_start) >= delist_date)
        previous = _present(context.get("previous_valid_quote"))
        if before_list or after_delist:
            status = "CLEARLY_OUTSIDE_H5_REACHABILITY"
            reason = "outside authoritative list/delist lifecycle"
        elif previous:
            status = "POTENTIALLY_H5_REACHABLE"
            reason = "listed envelope with a prior signal-date quote; no labels consulted"
        else:
            status = "UNKNOWN_REACHABILITY"
            reason = "insufficient signal-date eligibility evidence"
        rows.append(
            {
                "input_segment_id": segment_id,
                "canonical_ts_code": segment.canonical_ts_code,
                "segment_start": segment.segment_start,
                "segment_end": segment.segment_end,
                "session_count": segment.session_count,
                "reachability": status,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def unresolved_evidence_requests(
    coverage: DataFrame, triage: DataFrame, source_inventory: JsonObject
) -> DataFrame:
    """Emit actionable official-source requests for every remaining blocker."""

    parents = triage.set_index("parent_interval_id")
    sources = cast(list[JsonObject], source_inventory.get("sources", []))
    by_exchange: dict[str, list[str]] = {}
    for source in sources:
        by_exchange.setdefault(str(source.get("exchange", "")), []).append(
            str(source.get("package_id", ""))
        )
    rows: list[JsonObject] = []
    blocked = coverage[coverage["final_blocking"].astype(bool)]
    for segment in blocked.itertuples(index=False):
        context = parents.loc[str(segment.parent_interval_id)]
        exchange = str(segment.exchange)
        aliases = str(context.get("source_aliases", ""))
        query = (
            f"{segment.canonical_ts_code} {aliases} 停牌 复牌 暂停上市 恢复上市 "
            "终止上市 退市 吸收合并 换股 重大资产重组"
        ).strip()
        rows.append(
            {
                "input_segment_id": segment.input_segment_id,
                "parent_interval_id": segment.parent_interval_id,
                "canonical_ts_code": segment.canonical_ts_code,
                "historical_aliases": aliases,
                "historical_names": "",
                "segment_start": segment.segment_start,
                "segment_end": segment.segment_end,
                "session_count": segment.session_count,
                "exchange": exchange,
                "source_resolution": segment.source_resolution,
                "official_sources_searched": ";".join(sorted(set(by_exchange.get(exchange, [])))),
                "queries_attempted": query,
                "documents_inspected": ";".join(sorted(set(by_exchange.get(exchange, [])))),
                "insufficiency_reason": "no VERIFIED indexed event covers every segment session",
                "next_official_action": (
                    f"query {exchange} official company disclosures for exact effective boundaries"
                ),
                "lifecycle_resolution": "STILL_UNRESOLVED",
            }
        )
    return pd.DataFrame(rows)


def official_source_coverage(events: DataFrame) -> DataFrame:
    """Aggregate verified event rows by frozen evidence source."""

    if events.empty:
        return pd.DataFrame(columns=["exchange", "source", "package_id", "records", "securities"])
    return (
        events.groupby(
            ["exchange", "evidence_source_category", "package_id"], sort=True, dropna=False
        )
        .agg(records=("row_identity", "nunique"), securities=("canonical_ts_code", "nunique"))
        .reset_index()
        .rename(columns={"evidence_source_category": "source"})
    )


def closure_counts(
    queue: DataFrame, coverage: DataFrame, unresolved: DataFrame, reachability: DataFrame
) -> JsonObject:
    """Return exact queue, lifecycle and informational reachability counts."""

    by_resolution = {
        str(name): {
            "segments": int(len(group)),
            "sessions": int(group["session_count"].sum()),
            "securities": int(group["canonical_ts_code"].nunique()),
            "blocking": bool(group["final_blocking"].astype(bool).any()),
        }
        for name, group in coverage.groupby("lifecycle_resolution", sort=True)
    }
    reach = {
        str(name): {
            "segments": int(len(group)),
            "sessions": int(group["session_count"].sum()),
            "securities": int(group["canonical_ts_code"].nunique()),
        }
        for name, group in reachability.groupby("reachability", sort=True)
    }
    still = coverage["lifecycle_resolution"].eq("STILL_UNRESOLVED")
    return {
        "input_segments": int(len(queue)),
        "input_sessions": int(queue["session_count"].sum()),
        "input_securities": int(queue["canonical_ts_code"].nunique()),
        "output_segments": int(len(coverage)),
        "output_sessions": int(coverage["session_count"].sum()),
        "by_lifecycle_resolution": by_resolution,
        "still_unresolved": int(still.sum()),
        "still_unresolved_sessions": int(coverage.loc[still, "session_count"].sum()),
        "unverified_required_evidence": int(len(unresolved)),
        "evidence_retrieval_failed": 0,
        "h5_reachability": reach,
    }


def validate_evidence_closure_artifact(path: Path) -> JsonObject:
    """Validate exact child set and hashes for one D.3.2 artifact."""

    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError("SECURITY_LIFECYCLE_EVIDENCE_CLOSURE_INVALID") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_name") != ARTIFACT_NAME
        or manifest.get("closure_id") != path.name
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_EVIDENCE_CLOSURE_INVALID")
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict) or set(hashes) != ARTIFACT_FILES:
        raise DataValidationError("SECURITY_LIFECYCLE_EVIDENCE_CLOSURE_SET_INVALID")
    for name, expected in hashes.items():
        if not isinstance(expected, str) or file_sha256(path / name) != expected:
            raise DataValidationError("SECURITY_LIFECYCLE_EVIDENCE_CLOSURE_HASH_MISMATCH")
    return cast(JsonObject, manifest)


def _present(value: object) -> bool:
    return value is not None and str(value).strip() not in {"", "None", "<NA>", "NaT", "nan"}


def _optional_date(value: object) -> str | None:
    return str(value) if _present(value) else None


def _frame_hash(frame: DataFrame) -> str:
    columns = sorted(frame.columns)
    work = frame[columns].astype(object).where(pd.notna(frame[columns]), None)
    rows = work.sort_values(columns, key=lambda values: values.astype(str)).to_dict("records")
    return canonical_payload_hash({"columns": columns, "rows": rows})


def _report(summary: JsonObject) -> str:
    counts = cast(JsonObject, summary["counts"])
    return "\n".join(
        [
            "# Security Lifecycle Evidence Closure",
            "",
            f"- closure_id: `{summary['closure_id']}`",
            f"- D.4 readiness: `{summary['d4_readiness']}`",
            f"- input segments: `{counts['input_segments']}`",
            f"- input sessions: `{counts['input_sessions']}`",
            f"- remaining unresolved: `{counts['still_unresolved']}`",
            "- H5 reachability is informational and does not relax the lifecycle gate.",
            "- No raw or processed source was modified.",
            "",
        ]
    )
