"""D.3.3 closure of lifecycle gaps explained by verified code transitions."""

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
from ashare_quant.data.security_identity_transition import (
    SecurityIdentityTransitionResolver,
    canonical_payload_hash,
    file_sha256,
)
from ashare_quant.data.security_lifecycle_evidence_closure import (
    validate_evidence_closure_artifact,
)
from ashare_quant.data.security_lifecycle_resolution_hardened import _open_sessions
from ashare_quant.utils.manifest import atomic_write_json

type DataFrame = pd.DataFrame
type JsonObject = dict[str, Any]

SCHEMA_VERSION = 1
CONTRACT_VERSION = "security_lifecycle_transition_closure_v1"
ARTIFACT_NAME = "security_lifecycle_transition_closure"
ARTIFACT_FILES = frozenset(
    {
        "summary.json",
        "evidence_coverage_matrix.parquet",
        "unresolved.parquet",
        "transition_impact.parquet",
        "h5_reachability.parquet",
        "report.md",
    }
)


@dataclass(frozen=True, slots=True)
class TransitionClosureResult:
    """Published D.3.3 transition closure result."""

    closure_id: str
    output_dir: Path
    counts: JsonObject
    idempotent: bool


class SecurityLifecycleTransitionClosureService:
    """Resolve only post-transition old-code expectations in a D.3.2 queue."""

    def __init__(
        self,
        *,
        d32_closure_manifest: Path,
        identity_transition_artifact: Path,
        raw_root: Path,
        reports_root: Path,
    ) -> None:
        self.d32_manifest_path = d32_closure_manifest
        self.transition_path = identity_transition_artifact
        self.raw_root = raw_root
        self.reports_root = reports_root

    def run(self) -> TransitionClosureResult:
        """Validate inputs, split segments and publish exact session reconciliation."""

        upstream = validate_evidence_closure_artifact(self.d32_manifest_path.parent)
        transitions = SecurityIdentityTransitionResolver.from_path(self.transition_path)
        coverage = pd.read_parquet(
            self.d32_manifest_path.parent / "evidence_coverage_matrix.parquet"
        )
        queue = coverage[coverage["lifecycle_resolution"].eq("STILL_UNRESOLVED")].copy()
        expected = cast(JsonObject, upstream["counts"])
        if len(queue) != int(expected["still_unresolved"]) or int(
            queue["session_count"].sum()
        ) != int(expected["still_unresolved_sessions"]):
            raise DataValidationError("SECURITY_LIFECYCLE_TRANSITION_QUEUE_MISMATCH")
        calendar = _open_sessions(self.raw_root / "trade_cal")
        resolved = resolve_transition_queue(queue, calendar, transitions)
        original_reachability = pd.read_parquet(
            self.d32_manifest_path.parent / "h5_reachability.parquet"
        )
        reachability = _reachability(resolved, original_reachability)
        unresolved = resolved[resolved["final_blocking"].astype(bool)].copy()
        impact = _transition_impact(resolved, transitions, original_reachability)
        counts = _counts(queue, resolved, reachability)
        logical = {
            "schema_version": SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "d32_closure_id": upstream["closure_id"],
            "d32_closure_manifest_hash": file_sha256(self.d32_manifest_path),
            "identity_transition_version": transitions.artifact_version,
            "identity_transition_hash": transitions.artifact_hash,
            "input_queue_hash": _frame_hash(queue),
            "output_coverage_hash": _frame_hash(resolved),
        }
        closure_id = f"security_lifecycle_transition_closure_{canonical_payload_hash(logical)[:24]}"
        output = self.reports_root / "security_lifecycle_transition_closure" / closure_id
        if output.exists():
            manifest = validate_transition_closure_artifact(output)
            return TransitionClosureResult(
                closure_id, output, cast(JsonObject, manifest["counts"]), True
            )
        self._publish(output, logical, counts, resolved, unresolved, impact, reachability)
        return TransitionClosureResult(closure_id, output, counts, False)

    def _publish(
        self,
        output: Path,
        logical: JsonObject,
        counts: JsonObject,
        coverage: DataFrame,
        unresolved: DataFrame,
        impact: DataFrame,
        reachability: DataFrame,
    ) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
        try:
            readiness = (
                "AUTHORITATIVE_EVIDENCE_READY"
                if counts["still_unresolved"] == 0
                else "AUTHORITATIVE_EVIDENCE_BLOCKED"
            )
            summary = {
                "closure_id": output.name,
                "counts": counts,
                "d4_readiness": readiness,
                "lifecycle_preflight_status": "BLOCKED",
                "gate_split_recommendation": "KEEP_SINGLE_GLOBAL_GATE",
            }
            atomic_write_json(staging / "summary.json", summary)
            coverage.to_parquet(staging / "evidence_coverage_matrix.parquet", index=False)
            unresolved.to_parquet(staging / "unresolved.parquet", index=False)
            impact.to_parquet(staging / "transition_impact.parquet", index=False)
            reachability.to_parquet(staging / "h5_reachability.parquet", index=False)
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
        validate_transition_closure_artifact(output)


def resolve_transition_queue(
    queue: DataFrame,
    calendar: tuple[str, ...],
    transitions: SecurityIdentityTransitionResolver,
) -> DataFrame:
    """Split every input segment into exact unresolved/resolved session descendants."""

    dates = tuple(sorted(set(calendar)))
    rows: list[JsonObject] = []
    for source in queue.sort_values("input_segment_id").itertuples(index=False):
        segment_dates = [
            date for date in dates if str(source.segment_start) <= date <= str(source.segment_end)
        ]
        if len(segment_dates) != int(cast(int, source.session_count)):
            raise DataValidationError("SECURITY_LIFECYCLE_TRANSITION_SESSION_MISMATCH")
        daily: list[JsonObject] = []
        for date in segment_dates:
            transition = transitions.transition_for(str(source.canonical_ts_code), date)
            resolved = transition is not None
            daily.append(
                {
                    "trade_date": date,
                    "lifecycle_resolution": (
                        "VERIFIED_SECURITY_CODE_TRANSITION" if resolved else "STILL_UNRESOLVED"
                    ),
                    "official_evidence_package": (
                        transition.evidence_package_id if transition is not None else ""
                    ),
                    "identity_transition_successor": (
                        transition.successor_ts_code if transition is not None else ""
                    ),
                    "identity_transition_effective_date": (
                        transition.effective_date if transition is not None else ""
                    ),
                    "execution_supported": (
                        transition.execution_supported if transition is not None else False
                    ),
                    "final_blocking": not resolved,
                }
            )
        start = 0
        for index in range(1, len(daily) + 1):
            if index < len(daily) and _resolution_key(daily[index]) == _resolution_key(
                daily[start]
            ):
                continue
            group = daily[start:index]
            first = group[0]
            segment_identity = {
                "input_segment_id": str(source.input_segment_id),
                "start": group[0]["trade_date"],
                "end": group[-1]["trade_date"],
                "resolution": first["lifecycle_resolution"],
            }
            rows.append(
                {
                    "input_segment_id": str(source.input_segment_id),
                    "parent_interval_id": str(source.parent_interval_id),
                    "transition_segment_id": (
                        f"transition_segment_{canonical_payload_hash(segment_identity)[:24]}"
                    ),
                    "canonical_ts_code": str(source.canonical_ts_code),
                    "exchange": str(source.exchange),
                    "source_resolution": str(source.source_resolution),
                    "lifecycle_resolution": first["lifecycle_resolution"],
                    "official_evidence_package": first["official_evidence_package"],
                    "identity_transition_successor": first["identity_transition_successor"],
                    "identity_transition_effective_date": first[
                        "identity_transition_effective_date"
                    ],
                    "execution_supported": first["execution_supported"],
                    "repair_required": bool(source.repair_required),
                    "final_blocking": first["final_blocking"],
                    "segment_start": group[0]["trade_date"],
                    "segment_end": group[-1]["trade_date"],
                    "session_count": len(group),
                }
            )
            start = index
    result = pd.DataFrame(rows)
    if (
        int(result["session_count"].sum()) != int(queue["session_count"].sum())
        or result["transition_segment_id"].duplicated().any()
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_TRANSITION_RECONCILIATION_FAILED")
    by_input = result.groupby("input_segment_id")["session_count"].sum()
    expected = queue.set_index("input_segment_id")["session_count"].astype(int)
    if not by_input.astype(int).sort_index().equals(expected.sort_index()):
        raise DataValidationError("SECURITY_LIFECYCLE_TRANSITION_RECONCILIATION_FAILED")
    return result.sort_values(["canonical_ts_code", "segment_start"]).reset_index(drop=True)


def validate_transition_closure_artifact(path: Path) -> JsonObject:
    """Validate immutable D.3.3 closure children and logical identity."""

    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError("SECURITY_LIFECYCLE_TRANSITION_CLOSURE_INVALID") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_name") != ARTIFACT_NAME
        or manifest.get("closure_id") != path.name
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_TRANSITION_CLOSURE_INVALID")
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict) or set(hashes) != ARTIFACT_FILES:
        raise DataValidationError("SECURITY_LIFECYCLE_TRANSITION_CLOSURE_SET_INVALID")
    if {item.name for item in path.iterdir()} != ARTIFACT_FILES | {"manifest.json"}:
        raise DataValidationError("SECURITY_LIFECYCLE_TRANSITION_CLOSURE_SET_INVALID")
    for name, expected in hashes.items():
        if not isinstance(expected, str) or file_sha256(path / name) != expected:
            raise DataValidationError("SECURITY_LIFECYCLE_TRANSITION_CLOSURE_HASH_MISMATCH")
    logical = manifest.get("logical_identity")
    if not isinstance(logical, dict):
        raise DataValidationError("SECURITY_LIFECYCLE_TRANSITION_CLOSURE_INVALID")
    expected_id = f"security_lifecycle_transition_closure_{canonical_payload_hash(logical)[:24]}"
    if expected_id != path.name:
        raise DataValidationError("SECURITY_LIFECYCLE_TRANSITION_CLOSURE_IDENTITY_MISMATCH")
    return cast(JsonObject, manifest)


def _resolution_key(row: JsonObject) -> tuple[object, ...]:
    return (
        row["lifecycle_resolution"],
        row["official_evidence_package"],
        row["identity_transition_successor"],
        row["final_blocking"],
    )


def _reachability(coverage: DataFrame, original: DataFrame) -> DataFrame:
    source = original.set_index("input_segment_id")
    rows: list[JsonObject] = []
    for item in coverage[coverage["final_blocking"].astype(bool)].itertuples(index=False):
        previous = source.loc[str(item.input_segment_id)]
        rows.append(
            {
                "transition_segment_id": item.transition_segment_id,
                "input_segment_id": item.input_segment_id,
                "canonical_ts_code": item.canonical_ts_code,
                "segment_start": item.segment_start,
                "segment_end": item.segment_end,
                "session_count": item.session_count,
                "reachability": str(previous["reachability"]),
                "reason": str(previous["reason"]),
            }
        )
    return pd.DataFrame(rows)


def _transition_impact(
    coverage: DataFrame,
    transitions: SecurityIdentityTransitionResolver,
    original_reachability: DataFrame,
) -> DataFrame:
    reach = original_reachability.set_index("input_segment_id")
    rows: list[JsonObject] = []
    for transition in transitions.transition_records():
        matched = coverage[
            coverage["canonical_ts_code"].eq(transition.predecessor_ts_code)
            & coverage["lifecycle_resolution"].eq("VERIFIED_SECURITY_CODE_TRANSITION")
        ]
        input_ids = tuple(sorted(matched["input_segment_id"].astype(str).unique()))
        h5_statuses = sorted(
            {str(reach.loc[item]["reachability"]) for item in input_ids if item in reach.index}
        )
        rows.append(
            {
                "predecessor_ts_code": transition.predecessor_ts_code,
                "successor_ts_code": transition.successor_ts_code,
                "effective_date": transition.effective_date,
                "continuity_type": transition.continuity_type,
                "share_conversion_ratio": transition.share_conversion_ratio,
                "evidence_package_id": transition.evidence_package_id,
                "segments_resolved": int(len(matched)),
                "sessions_resolved": int(matched["session_count"].sum()),
                "h5_reachability": ";".join(h5_statuses),
                "execution_supported": transition.execution_supported,
                "execution_status": "CORPORATE_ACTION_EXECUTION_UNSUPPORTED",
            }
        )
    return pd.DataFrame(rows)


def _counts(queue: DataFrame, coverage: DataFrame, reachability: DataFrame) -> JsonObject:
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
    blocked = coverage[coverage["final_blocking"].astype(bool)]
    return {
        "input_segments": int(len(queue)),
        "input_sessions": int(queue["session_count"].sum()),
        "input_securities": int(queue["canonical_ts_code"].nunique()),
        "output_segments": int(len(coverage)),
        "output_sessions": int(coverage["session_count"].sum()),
        "parents_split": int((coverage.groupby("input_segment_id").size().astype(int) > 1).sum()),
        "by_lifecycle_resolution": by_resolution,
        "still_unresolved": int(len(blocked)),
        "still_unresolved_sessions": int(blocked["session_count"].sum()),
        "still_unresolved_securities": int(blocked["canonical_ts_code"].nunique()),
        "evidence_retrieval_failed": 0,
        "unverified_required_evidence": int(len(blocked)),
        "h5_reachability": reach,
    }


def _frame_hash(frame: DataFrame) -> str:
    columns = sorted(frame.columns)
    work = frame[columns].astype(object).where(pd.notna(frame[columns]), None)
    rows = work.sort_values(columns, key=lambda values: values.astype(str)).to_dict("records")
    return canonical_payload_hash({"columns": columns, "rows": rows})


def _report(summary: JsonObject) -> str:
    counts = cast(JsonObject, summary["counts"])
    return "\n".join(
        [
            "# Security Lifecycle Transition Closure",
            "",
            f"- closure_id: `{summary['closure_id']}`",
            f"- input sessions: `{counts['input_sessions']}`",
            f"- remaining unresolved: `{counts['still_unresolved']}`",
            f"- D.4 readiness: `{summary['d4_readiness']}`",
            "- Code-transition identity resolution does not implement position conversion.",
            "- H5 reachability remains informational; the global lifecycle gate is unchanged.",
            "",
        ]
    )
