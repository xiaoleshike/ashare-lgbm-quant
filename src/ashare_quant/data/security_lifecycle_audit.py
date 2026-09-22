"""Offline full-market security lifecycle completeness audit."""

# ruff: noqa: S608 -- SQL interpolates only validated dates and escaped local paths.

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import duckdb
import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_identity import SecurityIdentityResolver
from ashare_quant.data.security_identity_transition import SecurityIdentityTransitionResolver
from ashare_quant.data.security_lifecycle import SecurityLifecycleResolver
from ashare_quant.utils.manifest import atomic_write_json

type DataFrame = pd.DataFrame
type JsonObject = dict[str, Any]
type LifecycleClassification = Literal[
    "ORDINARY_SUSPENSION",
    "LISTING_SUSPENSION",
    "TERMINAL_DELISTING",
    "SECURITY_ALIAS",
    "MISSING_RAW_DATA",
    "UNRESOLVED",
]

SCANNER_SCHEMA_VERSION = 5
LEGACY_SCANNER_SCHEMA_VERSIONS = frozenset({1, 2, 3, 4})
CLASSIFICATION_CONTRACT_VERSION = 4
ARTIFACT_NAME = "security_lifecycle_scan"
REQUIRED_ARTIFACTS = frozenset(
    {
        "summary.json",
        "lifecycle_intervals.parquet",
        "classified_gaps.parquet",
        "boundary_checks.parquet",
        "raw_data_gaps.parquet",
        "unresolved.parquet",
        "same_day_sr_diagnostics.parquet",
        "old_vs_new_changes.parquet",
        "source_inventory.json",
        "supersession_comparison.json",
        "report.md",
    }
)
LEGACY_REQUIRED_ARTIFACTS = frozenset(
    {
        "summary.json",
        "lifecycle_intervals.parquet",
        "classified_gaps.parquet",
        "boundary_checks.parquet",
        "raw_data_gaps.parquet",
        "unresolved.parquet",
        "source_inventory.json",
        "report.md",
    }
)


@dataclass(frozen=True, slots=True)
class LifecycleAuditPolicy:
    """Versioned interpretation rules, separate from stock-level evidence."""

    schema_version: int
    policy_version: str
    policy_hash: str
    classification_contract_version: int
    payload: JsonObject
    path: Path

    @classmethod
    def from_path(cls, path: Path) -> LifecycleAuditPolicy:
        """Load one immutable lifecycle interpretation policy."""

        payload = _read_json(path, "lifecycle policy")
        if (
            payload.get("schema_version") != 2
            or payload.get("artifact_name") != "security_lifecycle_policy"
            or payload.get("classification_contract_version") != CLASSIFICATION_CONTRACT_VERSION
        ):
            raise DataValidationError("SECURITY_LIFECYCLE_POLICY_INVALID")
        version = payload.get("policy_version")
        if not isinstance(version, str) or not version:
            raise DataValidationError("SECURITY_LIFECYCLE_POLICY_INVALID")
        required_precedence = [
            "LISTING_SUSPENSION",
            "ORDINARY_SUSPENSION",
            "TERMINAL_DELISTING",
            "SECURITY_ALIAS",
            "MISSING_RAW_DATA",
            "UNRESOLVED",
        ]
        if payload.get("classification_precedence") != required_precedence:
            raise DataValidationError("SECURITY_LIFECYCLE_POLICY_INVALID: precedence")
        return cls(
            schema_version=2,
            policy_version=version,
            policy_hash=file_sha256(path),
            classification_contract_version=CLASSIFICATION_CONTRACT_VERSION,
            payload=payload,
            path=path,
        )


@dataclass(frozen=True, slots=True)
class SecurityLifecycleScanResult:
    """Published lifecycle scan outcome."""

    scan_id: str
    status: Literal["PASS", "BLOCKED"]
    output_dir: Path
    counts: JsonObject
    idempotent: bool


def normalize_ordinary_suspension_intervals(
    suspend_events: DataFrame,
    open_sessions: tuple[str, ...],
    *,
    scan_start: str,
    scan_end: str,
    valid_quote_keys: frozenset[tuple[str, str]] = frozenset(),
    stronger_evidence_keys: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[DataFrame, DataFrame]:
    """Compress explicit full-day S snapshots; R is boundary evidence only."""

    interval_columns = [
        "canonical_ts_code",
        "state",
        "effective_start",
        "effective_end",
        "start_source",
        "end_source",
        "evidence_hash",
    ]
    boundary_columns = [
        "canonical_ts_code",
        "trade_date",
        "boundary_type",
        "status",
        "blocking",
        "reason",
        "evidence",
    ]
    if suspend_events.empty:
        return pd.DataFrame(columns=interval_columns), pd.DataFrame(columns=boundary_columns)
    required = {"ts_code", "trade_date", "suspend_type"}
    if not required.issubset(suspend_events.columns):
        raise DataValidationError("SECURITY_LIFECYCLE_SOURCE_INVALID: suspend_d schema")
    calendar = tuple(sorted(set(open_sessions)))
    positions = {date: index for index, date in enumerate(calendar)}
    working = suspend_events.copy()
    working["ts_code"] = working["ts_code"].astype(str).str.strip().str.upper()
    working["trade_date"] = working["trade_date"].astype(str)
    working["suspend_type"] = working["suspend_type"].fillna("").astype(str).str.upper()
    if "suspend_timing" not in working:
        working["suspend_timing"] = pd.NA
    timing = working["suspend_timing"].astype("string").fillna("").str.strip()
    working["is_timed"] = timing.ne("")
    working = working[
        working["trade_date"].between(scan_start, scan_end)
        & working["trade_date"].isin(positions)
        & working["suspend_type"].isin(["S", "R"])
    ]
    working = working.drop_duplicates(
        subset=["ts_code", "trade_date", "suspend_type", "suspend_timing"]
    )
    intervals: list[JsonObject] = []
    boundaries: list[JsonObject] = []
    for code, code_events in working.groupby("ts_code", sort=True):
        explicit_suspended_dates: list[str] = []
        grouped = code_events.groupby("trade_date", sort=True)
        for trade_date, daily_events in grouped:
            code_text = str(code)
            date_text = str(trade_date)
            key = (code_text, date_text)
            event_types = set(daily_events["suspend_type"].astype(str))
            full_s = bool(((daily_events["suspend_type"] == "S") & ~daily_events["is_timed"]).any())
            timed_s = bool(((daily_events["suspend_type"] == "S") & daily_events["is_timed"]).any())
            has_r = "R" in event_types
            if full_s and has_r:
                has_quote = key in valid_quote_keys
                has_stronger_evidence = key in stronger_evidence_keys
                boundaries.append(
                    _boundary(
                        code_text,
                        date_text,
                        "SAME_DAY_SR_AMBIGUITY",
                        (
                            "WARNING"
                            if has_quote or has_stronger_evidence
                            else "BOUNDARY_INCONSISTENCY"
                        ),
                        (
                            "same-day full-day S/R has a valid quote or stronger lifecycle "
                            "evidence; it is not treated as full-day suspension"
                            if has_quote or has_stronger_evidence
                            else "same-day full-day S/R cannot explain the missing quote"
                        ),
                        blocking=not (has_quote or has_stronger_evidence),
                    )
                )
                continue
            if timed_s and has_r:
                boundaries.append(
                    _boundary(
                        code_text,
                        date_text,
                        "INTRADAY_SUSPEND_RESUME",
                        "INFO" if key in valid_quote_keys else "WARNING",
                        "timed S/R is intraday evidence and not a full-day suspension",
                        blocking=False,
                    )
                )
                continue
            if full_s:
                explicit_suspended_dates.append(date_text)
            if has_r:
                previous_index = positions[date_text] - 1
                prior_s = (
                    previous_index >= 0 and calendar[previous_index] in explicit_suspended_dates
                )
                boundaries.append(
                    _boundary(
                        code_text,
                        date_text,
                        "RESUME_DAY_CONSISTENCY",
                        "INFO" if prior_s else "WARNING",
                        (
                            "R follows an explicit full-day S snapshot"
                            if prior_s
                            else "R has no explicit full-day S snapshot on the prior open session"
                        ),
                        blocking=False,
                    )
                )
        if explicit_suspended_dates:
            start = explicit_suspended_dates[0]
            end = start
            for current in explicit_suspended_dates[1:]:
                if positions[current] == positions[end] + 1:
                    end = current
                    continue
                intervals.append(_ordinary_interval(str(code), start, end))
                start = current
                end = current
            intervals.append(_ordinary_interval(str(code), start, end))
    return (
        pd.DataFrame(intervals, columns=interval_columns),
        pd.DataFrame(boundaries, columns=boundary_columns),
    )


class SecurityLifecycleScanner:
    """Read-only full-market lifecycle scanner with immutable publication."""

    def __init__(
        self,
        *,
        raw_root: Path,
        processed_root: Path,
        reports_root: Path,
        identity_resolver: SecurityIdentityResolver,
        lifecycle_evidence: SecurityLifecycleResolver,
        lifecycle_policy: LifecycleAuditPolicy,
        identity_transitions: SecurityIdentityTransitionResolver,
        supersedes_scan_manifest: Path | None = None,
        supersession_reason: str | None = None,
    ) -> None:
        self.raw_root = raw_root
        self.processed_root = processed_root
        self.reports_root = reports_root
        self.identity = identity_resolver
        self.evidence = lifecycle_evidence
        self.policy = lifecycle_policy
        self.identity_transitions = identity_transitions
        self.identity_transitions.validate_alias_coexistence(identity_resolver)
        self.supersedes_scan_manifest = supersedes_scan_manifest
        self.supersession_reason = supersession_reason

    def scan(self, start_date: str, end_date: str) -> SecurityLifecycleScanResult:
        """Scan exact source snapshots and publish all findings, including blockers."""

        _validate_date(start_date, "start_date")
        _validate_date(end_date, "end_date")
        if start_date > end_date:
            raise DataValidationError("SECURITY_LIFECYCLE_SCAN_RANGE_INVALID")
        paths = self._required_paths()
        source_inventory = _build_source_inventory(paths)
        superseded = self._superseded_manifest()
        logical_identity = {
            "scanner_schema_version": SCANNER_SCHEMA_VERSION,
            "classification_contract_version": self.policy.classification_contract_version,
            "start_date": start_date,
            "end_date": end_date,
            "source_inventory_hash": canonical_payload_hash(source_inventory),
            "security_identity_mapping_version": self.identity.mapping_version,
            "security_identity_mapping_hash": self.identity.mapping_hash,
            "security_identity_transition_version": self.identity_transitions.artifact_version,
            "security_identity_transition_hash": self.identity_transitions.artifact_hash,
            "lifecycle_policy_version": self.policy.policy_version,
            "lifecycle_policy_hash": self.policy.policy_hash,
            "lifecycle_evidence_version": self.evidence.policy_version,
            "lifecycle_evidence_hash": self.evidence.policy_hash,
            "supersedes_scan_id": None if superseded is None else superseded["scan_id"],
            "supersession_reason": self.supersession_reason,
        }
        scan_id = f"security_lifecycle_{canonical_payload_hash(logical_identity)[:24]}"
        output = self.reports_root / "security_lifecycle" / scan_id
        if output.exists():
            manifest = validate_security_lifecycle_artifact(output)
            return SecurityLifecycleScanResult(
                scan_id=scan_id,
                status=cast(Literal["PASS", "BLOCKED"], manifest["status"]),
                output_dir=output,
                counts=cast(JsonObject, manifest["counts"]),
                idempotent=True,
            )
        frames, counts = self._scan_sources(start_date, end_date)
        comparison, changes = self._supersession_comparison(
            superseded=superseded,
            scan_id=scan_id,
            counts=counts,
            frames=frames,
        )
        frames["old_vs_new_changes"] = changes
        summary = {
            "scan_id": scan_id,
            "status": _scan_status(counts),
            "start_date": start_date,
            "end_date": end_date,
            "counts": counts,
            "policy_version": self.policy.policy_version,
            "policy_hash": self.policy.policy_hash,
            "lifecycle_evidence_version": self.evidence.policy_version,
            "lifecycle_evidence_hash": self.evidence.policy_hash,
            "security_identity_mapping_version": self.identity.mapping_version,
            "security_identity_mapping_hash": self.identity.mapping_hash,
            "security_identity_transition_version": self.identity_transitions.artifact_version,
            "security_identity_transition_hash": self.identity_transitions.artifact_hash,
            "supersedes_scan_id": comparison.get("old_scan_id"),
            "supersession_reason": comparison.get("supersession_reason"),
            "supersession_comparison": comparison,
        }
        manifest = self._publish(
            output=output,
            logical_identity=logical_identity,
            source_inventory=source_inventory,
            summary=summary,
            frames=frames,
            comparison=comparison,
        )
        return SecurityLifecycleScanResult(
            scan_id=scan_id,
            status=cast(Literal["PASS", "BLOCKED"], manifest["status"]),
            output_dir=output,
            counts=counts,
            idempotent=False,
        )

    def _superseded_manifest(self) -> JsonObject | None:
        if self.supersedes_scan_manifest is None:
            if self.supersession_reason is not None:
                raise DataValidationError("SECURITY_LIFECYCLE_SUPERSESSION_INVALID")
            return None
        if self.supersession_reason != "SUSPEND_D_SEMANTICS_RECALIBRATION":
            raise DataValidationError("SECURITY_LIFECYCLE_SUPERSESSION_INVALID")
        if self.supersedes_scan_manifest.name != "manifest.json":
            raise DataValidationError("SECURITY_LIFECYCLE_SUPERSESSION_INVALID")
        return validate_security_lifecycle_artifact(self.supersedes_scan_manifest.parent)

    def _supersession_comparison(
        self,
        *,
        superseded: JsonObject | None,
        scan_id: str,
        counts: JsonObject,
        frames: dict[str, DataFrame],
    ) -> tuple[JsonObject, DataFrame]:
        change_columns = ["canonical_ts_code", "old_classifications", "new_classifications"]
        if superseded is None or self.supersedes_scan_manifest is None:
            return (
                {
                    "old_scan_id": None,
                    "new_scan_id": scan_id,
                    "supersession_reason": None,
                    "count_delta": {},
                    "boundary_delta": {},
                    "same_day_sr": _same_day_summary(frames["same_day_sr_diagnostics"]),
                    "old_unresolved_security_outcomes": {},
                    "changed_security_count": 0,
                },
                pd.DataFrame(columns=change_columns),
            )
        old_root = self.supersedes_scan_manifest.parent
        old_classified = pd.read_parquet(old_root / "classified_gaps.parquet")
        old_boundaries = pd.read_parquet(old_root / "boundary_checks.parquet")
        old_unresolved = pd.read_parquet(old_root / "unresolved.parquet")
        new_classified = frames["classified_gaps"]
        new_unresolved = frames["unresolved"]
        tracked_counts = (
            "missing_price_candidates",
            "ordinary_suspension",
            "listing_suspension",
            "terminal_delisting",
            "unresolved",
            "policy_collisions",
            "boundary_inconsistencies",
        )
        old_counts = cast(JsonObject, superseded["counts"])
        count_delta = {
            key: {
                "old": int(old_counts.get(key, 0)),
                "new": int(counts.get(key, 0)),
                "delta": int(counts.get(key, 0)) - int(old_counts.get(key, 0)),
            }
            for key in tracked_counts
        }
        boundary_types = (
            "SUSPENSION_STATE",
            "ORDINARY_SUSPENSION_DAILY_STATE",
            "TERMINAL_BOUNDARY",
            "TERMINAL_TRANSITION",
            "SAME_DAY_FULL_SESSION_S_R",
            "SAME_DAY_SR_AMBIGUITY",
        )
        boundary_delta = {
            name: {
                "old": int(old_boundaries["boundary_type"].astype(str).eq(name).sum()),
                "new": int(frames["boundary_checks"]["boundary_type"].astype(str).eq(name).sum()),
            }
            for name in boundary_types
        }
        for values in boundary_delta.values():
            values["delta"] = values["new"] - values["old"]
        old_sets = _classification_sets(old_classified)
        new_sets = _classification_sets(new_classified)
        changed_rows = [
            {
                "canonical_ts_code": code,
                "old_classifications": ",".join(sorted(old_sets.get(code, set()))),
                "new_classifications": ",".join(sorted(new_sets.get(code, set()))),
            }
            for code in sorted(set(old_sets) | set(new_sets))
            if old_sets.get(code, set()) != new_sets.get(code, set())
        ]
        outcomes = _old_unresolved_outcomes(old_unresolved, new_classified, new_unresolved)
        return (
            {
                "old_scan_id": superseded["scan_id"],
                "new_scan_id": scan_id,
                "supersession_reason": self.supersession_reason,
                "count_delta": count_delta,
                "boundary_delta": boundary_delta,
                "same_day_sr": _same_day_summary(frames["same_day_sr_diagnostics"]),
                "old_unresolved_security_outcomes": outcomes,
                "changed_security_count": len(changed_rows),
            },
            pd.DataFrame(changed_rows, columns=change_columns),
        )

    def _required_paths(self) -> dict[str, Path]:
        paths = {
            "trade_cal": self.raw_root / "trade_cal",
            "stock_basic": self.raw_root / "stock_basic",
            "daily": self.raw_root / "daily",
            "suspend_d": self.raw_root / "suspend_d",
            "universe_daily": self.processed_root / "universe_daily",
        }
        missing = [name for name, path in paths.items() if not any(path.glob("**/*.parquet"))]
        if missing:
            raise DataValidationError(
                f"SECURITY_LIFECYCLE_SOURCE_MISSING: datasets={sorted(missing)}"
            )
        return paths

    def _scan_sources(
        self, start_date: str, end_date: str
    ) -> tuple[dict[str, DataFrame], JsonObject]:
        con = duckdb.connect()
        try:
            con.execute("SET memory_limit='3GB'")
            self._prepare_tables(con, start_date, end_date)
            calendar = tuple(
                con.execute("SELECT trade_date FROM open_sessions ORDER BY session_index")
                .fetchdf()["trade_date"]
                .astype(str)
            )
            coverage = con.execute(
                "SELECT min(cal_date),max(cal_date) FROM calendar_coverage"
            ).fetchone()
            if (
                not calendar
                or coverage is None
                or str(coverage[0]) > start_date
                or str(coverage[1]) < end_date
            ):
                raise DataValidationError("SECURITY_LIFECYCLE_COVERAGE_INSUFFICIENT: trade_cal")
            suspend = con.execute(
                "SELECT canonical_ts_code AS ts_code, trade_date, suspend_type, "
                "suspend_timing FROM suspend_events ORDER BY 1,2,3"
            ).fetchdf()
            listing = pd.DataFrame(
                [
                    {
                        "canonical_ts_code": event.canonical_ts_code,
                        "state": "LISTING_SUSPENSION",
                        "effective_start": event.effective_from,
                        "effective_end": event.effective_to,
                        "start_source": "VERIFIED_LIFECYCLE_EVIDENCE",
                        "end_source": "VERIFIED_LIFECYCLE_EVIDENCE",
                        "evidence_hash": canonical_payload_hash(
                            {
                                "source": event.evidence_source,
                                "reference": event.evidence_reference,
                            }
                        ),
                    }
                    for event in self.evidence.events()
                    if event.event_type == "LISTING_SUSPENDED"
                    and event.effective_to >= start_date
                    and event.effective_from <= end_date
                ]
            )
            interval_columns = [
                "canonical_ts_code",
                "state",
                "effective_start",
                "effective_end",
                "start_source",
                "end_source",
                "evidence_hash",
            ]
            if listing.empty:
                listing = pd.DataFrame(columns=interval_columns)
            valid_quote_keys = frozenset(
                (str(row.canonical_ts_code), str(row.trade_date))
                for row in con.execute(
                    "SELECT d.canonical_ts_code,d.trade_date FROM daily_canonical d "
                    "JOIN (SELECT DISTINCT canonical_ts_code,trade_date FROM suspend_events) s "
                    "USING(canonical_ts_code,trade_date) WHERE d.valid_quote"
                )
                .fetchdf()
                .itertuples(index=False)
            )
            stronger_evidence_keys = frozenset(
                (str(event.canonical_ts_code), date)
                for event in self.evidence.events()
                for date in calendar
                if event.effective_from <= date <= event.effective_to
            )
            ordinary, event_boundaries = normalize_ordinary_suspension_intervals(
                suspend,
                calendar,
                scan_start=start_date,
                scan_end=end_date,
                valid_quote_keys=valid_quote_keys,
                stronger_evidence_keys=stronger_evidence_keys,
            )
            verified_ordinary = pd.DataFrame(
                [
                    {
                        "canonical_ts_code": event.canonical_ts_code,
                        "state": "ORDINARY_SUSPENSION",
                        "effective_start": max(event.effective_from, start_date),
                        "effective_end": min(event.effective_to, end_date),
                        "start_source": "VERIFIED_LIFECYCLE_EVIDENCE",
                        "end_source": "VERIFIED_LIFECYCLE_EVIDENCE",
                        "evidence_hash": canonical_payload_hash(
                            {
                                "source": event.evidence_source,
                                "reference": event.evidence_reference,
                            }
                        ),
                    }
                    for event in self.evidence.events()
                    if event.event_type == "ORDINARY_SUSPENSION"
                    and event.effective_to >= start_date
                    and event.effective_from <= end_date
                ],
                columns=interval_columns,
            )
            ordinary = pd.concat([ordinary, verified_ordinary], ignore_index=True).drop_duplicates()
            intervals = pd.concat([ordinary, listing], ignore_index=True)
            con.register("lifecycle_intervals_input", intervals)
            classified_daily = self._classify_daily(con)
            classified = _compress_classified(classified_daily)
            consistency = self._boundary_checks(con, intervals)
            boundaries = pd.concat([event_boundaries, consistency], ignore_index=True)
            same_day_sr = self._same_day_sr_diagnostics(con, intervals, boundaries)
            raw_gaps = con.execute(
                "SELECT 'daily' AS dataset, trade_date, partition, 'FULL_MARKET_DATE' AS scope, "
                "'open trade_cal session has no daily rows' AS reason, TRUE AS blocking, "
                "'trade_cal/daily anti-join' AS evidence FROM raw_gap_dates ORDER BY trade_date"
            ).fetchdf()
            unresolved = classified[classified["blocking"].astype(bool)].copy()
            candidate_counts = con.execute(
                "SELECT (SELECT count(*) FROM expected_listed) AS listed_session_candidates, "
                "(SELECT count(*) FROM expected_listed e JOIN daily_canonical d USING "
                "(canonical_ts_code, trade_date) WHERE d.valid_quote) AS valid_price_rows, "
                "(SELECT count(*) FROM missing_candidates) AS missing_price_candidates, "
                "(SELECT count(*) FROM securities) AS canonical_securities, "
                "(SELECT count(*) FROM open_sessions) AS open_sessions"
            ).fetchone()
            assert candidate_counts is not None
            counts: JsonObject = {
                "canonical_securities": int(candidate_counts[3]),
                "open_sessions": int(candidate_counts[4]),
                "listed_session_candidates": int(candidate_counts[0]),
                "valid_price_rows": int(candidate_counts[1]),
                "missing_price_candidates": int(candidate_counts[2]),
            }
            for name in (
                "ORDINARY_SUSPENSION",
                "LISTING_SUSPENSION",
                "TERMINAL_DELISTING",
                "SECURITY_ALIAS",
                "MISSING_RAW_DATA",
                "UNRESOLVED",
            ):
                selected = classified[classified["classification"] == name]
                counts[name.lower()] = (
                    int(selected["session_count"].sum()) if not selected.empty else 0
                )
                counts[f"{name.lower()}_intervals"] = int(len(selected))
            counts["policy_collisions"] = int(
                boundaries["status"].astype(str).eq("POLICY_COLLISION").sum()
            )
            counts["boundary_inconsistencies"] = int(
                (
                    boundaries["status"].astype(str).eq("BOUNDARY_INCONSISTENCY")
                    & boundaries["blocking"].fillna(False).astype(bool)
                ).sum()
            )
            counts["boundary_warnings"] = int(boundaries["status"].astype(str).eq("WARNING").sum())
            counts["boundary_info"] = int(boundaries["status"].astype(str).eq("INFO").sum())
            counts["blocking_raw_data_gaps"] = int(
                raw_gaps["blocking"].fillna(False).astype(bool).sum()
            )
            counts["v3_events_preserved"] = self.evidence.event_count
            counts["new_verified_lifecycle_events"] = 0
            return (
                {
                    "lifecycle_intervals": intervals,
                    "classified_gaps": classified,
                    "boundary_checks": boundaries,
                    "raw_data_gaps": raw_gaps,
                    "unresolved": unresolved,
                    "same_day_sr_diagnostics": same_day_sr,
                },
                counts,
            )
        finally:
            con.close()

    def _prepare_tables(self, con: duckdb.DuckDBPyConnection, start: str, end: str) -> None:
        aliases = pd.DataFrame(
            [
                {
                    "source_code": alias.source_code,
                    "canonical_code": alias.canonical_code,
                    "effective_from": alias.effective_from,
                    "effective_to": alias.effective_to,
                }
                for alias in self.identity.alias_records()
            ],
            columns=["source_code", "canonical_code", "effective_from", "effective_to"],
        )
        con.register("security_aliases", aliases)
        transitions = pd.DataFrame(
            [
                {
                    "predecessor_ts_code": item.predecessor_ts_code,
                    "successor_ts_code": item.successor_ts_code,
                    "effective_date": item.effective_date,
                    "evidence_package_id": item.evidence_package_id,
                }
                for item in self.identity_transitions.transition_records()
            ],
            columns=[
                "predecessor_ts_code",
                "successor_ts_code",
                "effective_date",
                "evidence_package_id",
            ],
        )
        con.register("security_identity_transitions", transitions)
        trade_cal = _parquet_glob(self.raw_root / "trade_cal")
        stock_basic = _parquet_glob(self.raw_root / "stock_basic")
        daily = _parquet_glob(self.raw_root / "daily")
        suspend = _parquet_glob(self.raw_root / "suspend_d")
        universe = _parquet_glob(self.processed_root / "universe_daily")
        con.execute(  # noqa: S608 - validated local paths and YYYYMMDD bounds only
            f"""CREATE TEMP TABLE calendar_coverage AS
            SELECT CAST(cal_date AS VARCHAR) AS cal_date
            FROM read_parquet('{trade_cal}', union_by_name=true, hive_partitioning=false)
            WHERE exchange='SSE'"""
        )
        con.execute(  # noqa: S608 - validated local paths and YYYYMMDD bounds only
            f"""CREATE TEMP TABLE open_sessions AS
            SELECT trade_date, row_number() OVER (ORDER BY trade_date) - 1 AS session_index
            FROM (
              SELECT DISTINCT CAST(cal_date AS VARCHAR) AS trade_date
              FROM read_parquet('{trade_cal}', union_by_name=true, hive_partitioning=false)
              WHERE exchange='SSE' AND is_open=1
                AND cal_date BETWEEN '{start}' AND '{end}'
            ) ORDER BY trade_date"""
        )
        con.execute(  # noqa: S608 - validated local paths and YYYYMMDD bounds only
            f"""CREATE TEMP TABLE daily_raw AS
            SELECT upper(trim(CAST(d.ts_code AS VARCHAR))) AS source_ts_code,
                   coalesce(a.canonical_code, upper(trim(CAST(d.ts_code AS VARCHAR))))
                     AS canonical_ts_code,
                   CAST(d.trade_date AS VARCHAR) AS trade_date,
                   d.open,d.close
            FROM read_parquet('{daily}', union_by_name=true, hive_partitioning=false) d
            LEFT JOIN security_aliases a
              ON upper(trim(CAST(d.ts_code AS VARCHAR)))=a.source_code
             AND (a.effective_from IS NULL OR CAST(d.trade_date AS VARCHAR)>=a.effective_from)
             AND (a.effective_to IS NULL OR CAST(d.trade_date AS VARCHAR)<=a.effective_to)
            WHERE d.trade_date BETWEEN '{start}' AND '{end}'"""
        )
        con.execute(  # noqa: S608 - validated local paths only
            """CREATE TEMP TABLE daily_canonical AS
            SELECT canonical_ts_code, trade_date,
                   bool_or(open IS NOT NULL AND isfinite(open) AND open>0
                     AND close IS NOT NULL AND isfinite(close) AND close>0)
                     AND count(DISTINCT struct_pack(open := open, close := close))=1 AS valid_quote,
                   string_agg(DISTINCT source_ts_code, ',' ORDER BY source_ts_code)
                     AS source_ts_codes,
                   bool_or(source_ts_code=canonical_ts_code AND open IS NOT NULL
                     AND isfinite(open) AND open>0 AND close IS NOT NULL
                     AND isfinite(close) AND close>0) AS canonical_source_valid,
                   bool_or(source_ts_code<>canonical_ts_code AND open IS NOT NULL
                     AND isfinite(open) AND open>0 AND close IS NOT NULL
                     AND isfinite(close) AND close>0) AS alias_source_valid,
                   count(DISTINCT struct_pack(open := open, close := close))>1 AS quote_conflict
            FROM daily_raw GROUP BY 1,2"""
        )
        con.execute(  # noqa: S608 - validated local paths and YYYYMMDD bounds only
            f"""CREATE TEMP TABLE stock_source AS
            SELECT coalesce(a.canonical_code, upper(trim(CAST(s.ts_code AS VARCHAR))))
                     AS canonical_ts_code,
                   nullif(trim(CAST(s.list_date AS VARCHAR)), '') AS list_date,
                   nullif(trim(CAST(s.delist_date AS VARCHAR)), '') AS delist_date
            FROM read_parquet('{stock_basic}', union_by_name=true, hive_partitioning=false) s
            LEFT JOIN security_aliases a
              ON upper(trim(CAST(s.ts_code AS VARCHAR)))=a.source_code
             AND a.effective_from IS NULL AND a.effective_to IS NULL"""
        )
        con.execute(
            """CREATE TEMP TABLE daily_security_bounds AS
            SELECT canonical_ts_code,min(trade_date) AS first_daily_date,
                   max(trade_date) AS last_daily_date
            FROM daily_canonical GROUP BY 1"""
        )
        con.execute(  # noqa: S608 - validated local paths and YYYYMMDD bounds only
            """CREATE TEMP TABLE securities AS
            WITH stock AS (
              SELECT canonical_ts_code, min(list_date) AS list_date,
                     max(delist_date) AS delist_date
              FROM stock_source GROUP BY 1
            )
            SELECT canonical_ts_code,list_date,delist_date FROM stock
            WHERE list_date IS NOT NULL"""
        )
        con.execute(
            """CREATE TEMP TABLE expected_listed AS
            SELECT s.canonical_ts_code, c.trade_date, c.session_index,
                   s.list_date, s.delist_date
            FROM securities s JOIN open_sessions c
              ON c.trade_date>=s.list_date
             AND (s.delist_date IS NULL OR c.trade_date<s.delist_date)
            LEFT JOIN security_identity_transitions t
              ON t.predecessor_ts_code=s.canonical_ts_code
            WHERE t.effective_date IS NULL OR c.trade_date<t.effective_date"""
        )
        con.execute(
            """CREATE TEMP TABLE missing_candidates AS
            SELECT e.*, d.source_ts_codes
            FROM expected_listed e LEFT JOIN daily_canonical d USING(canonical_ts_code,trade_date)
            WHERE coalesce(d.valid_quote,false)=false"""
        )
        con.execute(
            """CREATE TEMP TABLE raw_gap_dates AS
            SELECT c.trade_date,
                   'year='||substr(c.trade_date,1,4)||'/month='||substr(c.trade_date,5,2)
                     AS partition
            FROM open_sessions c LEFT JOIN
              (SELECT DISTINCT trade_date FROM daily_raw) d USING(trade_date)
            WHERE d.trade_date IS NULL"""
        )
        con.execute(
            f"""CREATE TEMP TABLE universe_state AS
            SELECT upper(trim(CAST(ts_code AS VARCHAR))) AS canonical_ts_code,
                   CAST(trade_date AS VARCHAR) AS trade_date,
                   bool_or(coalesce(is_listed,false)) AS is_listed,
                   bool_or(coalesce(is_suspended,false)) AS is_suspended
            FROM read_parquet('{universe}', union_by_name=true, hive_partitioning=false)
            WHERE trade_date BETWEEN '{start}' AND '{end}' GROUP BY 1,2"""
        )
        con.execute(
            f"""CREATE TEMP TABLE suspend_events AS
            SELECT coalesce(a.canonical_code, upper(trim(CAST(s.ts_code AS VARCHAR))))
                     AS canonical_ts_code,
                   CAST(s.trade_date AS VARCHAR) AS trade_date,
                   upper(trim(CAST(s.suspend_type AS VARCHAR))) AS suspend_type,
                   CAST(s.suspend_timing AS VARCHAR) AS suspend_timing
            FROM read_parquet('{suspend}', union_by_name=true, hive_partitioning=false) s
            LEFT JOIN security_aliases a
              ON upper(trim(CAST(s.ts_code AS VARCHAR)))=a.source_code
             AND (a.effective_from IS NULL OR CAST(s.trade_date AS VARCHAR)>=a.effective_from)
             AND (a.effective_to IS NULL OR CAST(s.trade_date AS VARCHAR)<=a.effective_to)
            WHERE s.trade_date BETWEEN '{start}' AND '{end}'"""
        )

    def _classify_daily(self, con: duckdb.DuckDBPyConnection) -> DataFrame:
        return con.execute(
            """WITH ordinary AS (
              SELECT * FROM lifecycle_intervals_input WHERE state='ORDINARY_SUSPENSION'
            ), listing AS (
              SELECT * FROM lifecycle_intervals_input WHERE state='LISTING_SUSPENSION'
            ), missing AS (
              SELECT m.*, u.is_listed AS universe_is_listed,
                     u.is_suspended AS universe_is_suspended,
                     l.effective_start AS listing_start,l.effective_end AS listing_end,
                     l.evidence_hash AS listing_evidence_hash,
                     o.effective_start AS ordinary_start,o.effective_end AS ordinary_end,
                     o.evidence_hash AS ordinary_evidence_hash,
                     r.trade_date IS NOT NULL AS raw_gap
              FROM missing_candidates m
              LEFT JOIN universe_state u ON u.canonical_ts_code=m.canonical_ts_code
                AND u.trade_date=m.trade_date
              LEFT JOIN listing l ON l.canonical_ts_code=m.canonical_ts_code
                AND m.trade_date BETWEEN l.effective_start AND l.effective_end
              LEFT JOIN ordinary o ON o.canonical_ts_code=m.canonical_ts_code
                AND m.trade_date BETWEEN o.effective_start AND o.effective_end
              LEFT JOIN raw_gap_dates r ON r.trade_date=m.trade_date
            ), gap_rows AS (
              SELECT canonical_ts_code, coalesce(source_ts_codes,'') AS source_ts_codes,
                     trade_date, session_index, list_date, delist_date,
                CASE WHEN listing_start IS NOT NULL THEN 'LISTING_SUSPENSION'
                     WHEN ordinary_start IS NOT NULL THEN 'ORDINARY_SUSPENSION'
                     WHEN raw_gap THEN 'MISSING_RAW_DATA' ELSE 'UNRESOLVED' END classification,
                CASE WHEN listing_start IS NOT NULL OR ordinary_start IS NOT NULL
                       THEN 'RESOLVED_NONTRADING'
                     WHEN raw_gap THEN 'BLOCKING_RAW_DATA'
                     ELSE 'BLOCKING_UNRESOLVED' END disposition,
                NOT (listing_start IS NOT NULL OR ordinary_start IS NOT NULL) AS blocking,
                coalesce(listing_start,ordinary_start) AS evidence_start,
                coalesce(listing_end,ordinary_end) AS evidence_end,
                coalesce(listing_evidence_hash,ordinary_evidence_hash,
                  CASE WHEN raw_gap THEN 'trade_cal/daily anti-join' ELSE '' END) AS evidence_ids,
                CASE WHEN listing_start IS NOT NULL THEN 'verified listing-suspension interval'
                     WHEN ordinary_start IS NOT NULL THEN 'authoritative suspend_d interval'
                     WHEN raw_gap THEN 'entire open-session daily data is absent'
                     ELSE 'listed security has no valid quote or authoritative lifecycle evidence'
                END reason
              FROM missing
            ), aliases AS (
              SELECT e.canonical_ts_code,d.source_ts_codes,e.trade_date,e.session_index,
                     e.list_date,e.delist_date,'SECURITY_ALIAS' AS classification,
                     'RESOLVED_IDENTITY' AS disposition,FALSE AS blocking,
                     e.trade_date AS evidence_start,e.trade_date AS evidence_end,
                     d.source_ts_codes AS evidence_ids,
                     'explicit identity mapping supplies canonical quote' AS reason
              FROM expected_listed e JOIN daily_canonical d USING(canonical_ts_code,trade_date)
              WHERE d.valid_quote AND d.alias_source_valid AND NOT d.canonical_source_valid
            ), terminal AS (
              SELECT s.canonical_ts_code,'' AS source_ts_codes,c.trade_date,c.session_index,
                     s.list_date,s.delist_date,'TERMINAL_DELISTING' AS classification,
                     'RESOLVED_TERMINAL' AS disposition,FALSE AS blocking,
                     s.delist_date AS evidence_start,s.delist_date AS evidence_end,
                     'stock_basic.delist_date' AS evidence_ids,
                     'first open session at or after authoritative delist_date' AS reason
              FROM securities s JOIN open_sessions c ON c.trade_date>=s.delist_date
              QUALIFY row_number() OVER(PARTITION BY s.canonical_ts_code ORDER BY c.trade_date)=1
            )
            SELECT * FROM gap_rows UNION ALL SELECT * FROM aliases UNION ALL SELECT * FROM terminal
            ORDER BY canonical_ts_code,trade_date,classification"""
        ).fetchdf()

    def _boundary_checks(self, con: duckdb.DuckDBPyConnection, intervals: DataFrame) -> DataFrame:
        columns = [
            "canonical_ts_code",
            "trade_date",
            "boundary_type",
            "status",
            "blocking",
            "reason",
            "evidence",
        ]
        con.register("all_intervals", intervals)
        checks = con.execute(
            """WITH calendar_bounds AS (
              SELECT min(trade_date) AS first_date,max(trade_date) AS last_date FROM open_sessions
            ), ordinary_mismatch AS (
              SELECT DISTINCT i.canonical_ts_code,c.trade_date,
                     'ORDINARY_SUSPENSION_DAILY_STATE' boundary_type,
                     'BOUNDARY_INCONSISTENCY' status,TRUE blocking,
                     'explicit full-day S snapshot is absent from universe_daily' reason,
                     'suspend_d:S' evidence
              FROM all_intervals i JOIN open_sessions c
                ON c.trade_date BETWEEN i.effective_start AND i.effective_end
              LEFT JOIN universe_state u ON u.canonical_ts_code=i.canonical_ts_code
                AND u.trade_date=c.trade_date
              WHERE i.state='ORDINARY_SUSPENSION'
                AND coalesce(u.is_suspended,false)=false
            ), listing_boundary_mismatch AS (
              SELECT i.canonical_ts_code,b.trade_date,b.boundary_type,
                     'BOUNDARY_INCONSISTENCY',TRUE,
                     'verified listing-suspension boundary is absent from universe_daily',
                     'verified lifecycle evidence'
              FROM (
                SELECT i.canonical_ts_code,i.effective_start,i.effective_end,
                       min(c.trade_date) AS first_open_session,
                       max(c.trade_date) AS last_open_session
                FROM all_intervals i JOIN open_sessions c
                  ON c.trade_date BETWEEN i.effective_start AND i.effective_end
                WHERE i.state='LISTING_SUSPENSION' GROUP BY 1,2,3
              ) i
              CROSS JOIN LATERAL (VALUES
                (i.first_open_session,'LISTING_SUSPENSION_START'),
                (i.last_open_session,'LISTING_SUSPENSION_END')
              ) b(trade_date,boundary_type)
              LEFT JOIN universe_state u ON u.canonical_ts_code=i.canonical_ts_code
                AND u.trade_date=b.trade_date
              WHERE coalesce(u.is_suspended,false)=false
            ), resume_days AS (
              SELECT e.canonical_ts_code,e.trade_date,e.has_same_day_s,
                     prior.trade_date AS prior_open_session
              FROM (
                SELECT canonical_ts_code,trade_date,
                       bool_or(suspend_type='S') AS has_same_day_s
                FROM suspend_events GROUP BY 1,2 HAVING bool_or(suspend_type='R')
              ) e
              JOIN open_sessions c USING(trade_date)
              LEFT JOIN open_sessions prior ON prior.session_index=c.session_index-1
            ), resume_checks AS (
              SELECT r.canonical_ts_code,r.trade_date,'RESUME_DAY_CONSISTENCY',
                CASE WHEN r.has_same_day_s THEN 'WARNING'
                     WHEN coalesce(u.is_suspended,false) THEN 'BOUNDARY_INCONSISTENCY'
                     WHEN NOT coalesce(d.valid_quote,false) THEN 'WARNING'
                     WHEN p.canonical_ts_code IS NULL THEN 'WARNING' ELSE 'INFO' END,
                coalesce(u.is_suspended,false) AND NOT r.has_same_day_s,
                CASE WHEN r.has_same_day_s
                       THEN 'same-day S/R is handled by its dedicated provider-shape diagnostic'
                     WHEN coalesce(u.is_suspended,false)
                       THEN 'R resume date remains suspended in universe_daily'
                     WHEN NOT coalesce(d.valid_quote,false)
                       THEN 'R resume date has no valid executable daily quote'
                     WHEN p.canonical_ts_code IS NULL
                       THEN 'R has no explicit full-day S snapshot on prior open session'
                     ELSE 'R follows an explicit full-day S snapshot and has a valid quote' END,
                'suspend_d:R'
              FROM resume_days r
              LEFT JOIN universe_state u USING(canonical_ts_code,trade_date)
              LEFT JOIN daily_canonical d USING(canonical_ts_code,trade_date)
              LEFT JOIN (
                SELECT DISTINCT canonical_ts_code,trade_date FROM suspend_events
                WHERE suspend_type='S' AND coalesce(trim(suspend_timing),'')=''
              ) p ON p.canonical_ts_code=r.canonical_ts_code
                AND p.trade_date=r.prior_open_session
            ), terminal_mismatch AS (
              SELECT b.canonical_ts_code,b.trade_date,'TERMINAL_TRANSITION',
                     'BOUNDARY_INCONSISTENCY',TRUE,
                     'universe_daily remains listed on authoritative delist_date',
                     'stock_basic.delist_date'
              FROM (
                SELECT s.canonical_ts_code,min(c.trade_date) AS trade_date
                FROM securities s JOIN open_sessions c ON c.trade_date>=s.delist_date
                CROSS JOIN calendar_bounds bounds
                WHERE s.delist_date IS NOT NULL
                  AND s.delist_date BETWEEN bounds.first_date AND bounds.last_date GROUP BY 1
              ) b
              LEFT JOIN universe_state u ON u.canonical_ts_code=b.canonical_ts_code
                AND u.trade_date=b.trade_date
              WHERE coalesce(u.is_listed,true)=true
            ), post_terminal AS (
              SELECT s.canonical_ts_code,min(d.trade_date) AS trade_date,'POST_TERMINAL_QUOTE',
                     'BOUNDARY_INCONSISTENCY',TRUE,
                     'valid quote exists on or after authoritative terminal boundary',
                     string_agg(DISTINCT d.source_ts_codes,',' ORDER BY d.source_ts_codes)
              FROM securities s JOIN daily_canonical d USING(canonical_ts_code)
              WHERE s.delist_date IS NOT NULL AND d.trade_date>=s.delist_date AND d.valid_quote
              GROUP BY 1
            ), listing_start_mismatch AS (
              SELECT b.canonical_ts_code,b.trade_date,'LISTING_START',
                     'BOUNDARY_INCONSISTENCY',TRUE,
                     'universe_daily is not listed on first expected listed session',
                     'stock_basic.list_date/daily first date'
              FROM (
                SELECT s.canonical_ts_code,min(c.trade_date) AS trade_date
                FROM securities s JOIN open_sessions c ON c.trade_date>=s.list_date
                CROSS JOIN calendar_bounds bounds
                WHERE s.list_date BETWEEN bounds.first_date AND bounds.last_date GROUP BY 1
              ) b
              LEFT JOIN universe_state u ON u.canonical_ts_code=b.canonical_ts_code
                AND u.trade_date=b.trade_date
              WHERE coalesce(u.is_listed,false)=false
            ), quote_conflicts AS (
              SELECT canonical_ts_code,trade_date,'DAILY_QUOTE_CONFLICT',
                     'BOUNDARY_INCONSISTENCY',TRUE,
                     'canonical daily rows contain conflicting executable prices',source_ts_codes
              FROM daily_canonical WHERE quote_conflict
            ), transition_successor_coverage AS (
              SELECT t.successor_ts_code AS canonical_ts_code,t.effective_date AS trade_date,
                     'IDENTITY_SUCCESSOR_COVERAGE',
                     'BOUNDARY_INCONSISTENCY',TRUE,
                     'identity transition successor lacks authoritative stock_basic metadata',
                     t.evidence_package_id
              FROM security_identity_transitions t
              LEFT JOIN stock_source s ON s.canonical_ts_code=t.successor_ts_code
              WHERE s.canonical_ts_code IS NULL
            ), transition_double_count AS (
              SELECT t.predecessor_ts_code AS canonical_ts_code,d1.trade_date,
                     'IDENTITY_TRANSITION_DOUBLE_QUOTE',
                     'BOUNDARY_INCONSISTENCY',TRUE,
                     'predecessor and successor both have valid canonical quotes after transition',
                     t.evidence_package_id
              FROM security_identity_transitions t
              JOIN daily_canonical d1 ON d1.canonical_ts_code=t.predecessor_ts_code
                AND d1.trade_date>=t.effective_date AND d1.valid_quote
              JOIN daily_canonical d2 ON d2.canonical_ts_code=t.successor_ts_code
                AND d2.trade_date=d1.trade_date AND d2.valid_quote
            ), listing_metadata_missing AS (
              SELECT d.canonical_ts_code,d.first_daily_date AS trade_date,
                     'LISTING_METADATA_MISSING','BOUNDARY_INCONSISTENCY',TRUE,
                     'daily history exists without authoritative stock_basic listing metadata',
                     'stock_basic/daily identity accounting'
              FROM daily_security_bounds d
              LEFT JOIN stock_source s USING(canonical_ts_code)
              WHERE s.canonical_ts_code IS NULL
            )
            SELECT * FROM ordinary_mismatch
            UNION ALL SELECT * FROM listing_boundary_mismatch
            UNION ALL SELECT * FROM resume_checks
            UNION ALL SELECT * FROM terminal_mismatch
            UNION ALL SELECT * FROM post_terminal
            UNION ALL SELECT * FROM listing_start_mismatch
            UNION ALL SELECT * FROM quote_conflicts
            UNION ALL SELECT * FROM transition_successor_coverage
            UNION ALL SELECT * FROM transition_double_count
            UNION ALL SELECT * FROM listing_metadata_missing
            ORDER BY 1,2,3"""
        ).fetchdf()
        checks.columns = columns
        return checks

    def _same_day_sr_diagnostics(
        self,
        con: duckdb.DuckDBPyConnection,
        intervals: DataFrame,
        boundaries: DataFrame,
    ) -> DataFrame:
        columns = [
            "canonical_ts_code",
            "trade_date",
            "has_full_day_s",
            "has_intraday_s",
            "has_r",
            "valid_daily_quote",
            "universe_is_suspended",
            "overlaps_listing_suspension",
            "terminal_effective",
            "boundary_type",
            "severity",
            "blocking",
        ]
        con.register("same_day_boundary_input", boundaries)
        con.register("same_day_interval_input", intervals)
        frame = con.execute(
            """WITH grouped AS (
              SELECT canonical_ts_code,trade_date,
                     bool_or(suspend_type='S' AND coalesce(trim(suspend_timing),'')='')
                       AS has_full_day_s,
                     bool_or(suspend_type='S' AND coalesce(trim(suspend_timing),'')<>'')
                       AS has_intraday_s,
                     bool_or(suspend_type='R') AS has_r
              FROM suspend_events GROUP BY 1,2
            )
            SELECT g.canonical_ts_code,g.trade_date,g.has_full_day_s,g.has_intraday_s,g.has_r,
                   coalesce(d.valid_quote,false) AS valid_daily_quote,
                   coalesce(u.is_suspended,false) AS universe_is_suspended,
                   bool_or(coalesce(i.state='LISTING_SUSPENSION',false))
                     AS overlaps_listing_suspension,
                   bool_or(coalesce(s.delist_date IS NOT NULL
                     AND g.trade_date>=s.delist_date,false)) AS terminal_effective,
                   max(b.boundary_type) AS boundary_type,
                   max(b.status) AS severity,
                   bool_or(coalesce(b.blocking,false)) AS blocking
            FROM grouped g
            LEFT JOIN daily_canonical d USING(canonical_ts_code,trade_date)
            LEFT JOIN universe_state u USING(canonical_ts_code,trade_date)
            LEFT JOIN same_day_interval_input i ON i.canonical_ts_code=g.canonical_ts_code
              AND g.trade_date BETWEEN i.effective_start AND i.effective_end
            LEFT JOIN securities s USING(canonical_ts_code)
            LEFT JOIN same_day_boundary_input b ON b.canonical_ts_code=g.canonical_ts_code
              AND b.trade_date=g.trade_date
              AND b.boundary_type IN ('SAME_DAY_SR_AMBIGUITY','INTRADAY_SUSPEND_RESUME')
            WHERE g.has_r AND (g.has_full_day_s OR g.has_intraday_s)
            GROUP BY 1,2,3,4,5,6,7 ORDER BY 1,2"""
        ).fetchdf()
        if frame.empty:
            return pd.DataFrame(columns=columns)
        frame.columns = columns
        return frame

    def _publish(
        self,
        *,
        output: Path,
        logical_identity: JsonObject,
        source_inventory: JsonObject,
        summary: JsonObject,
        frames: dict[str, DataFrame],
        comparison: JsonObject,
    ) -> JsonObject:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
        try:
            atomic_write_json(staging / "summary.json", summary)
            atomic_write_json(staging / "source_inventory.json", source_inventory)
            atomic_write_json(staging / "supersession_comparison.json", comparison)
            for name, frame in frames.items():
                frame.to_parquet(staging / f"{name}.parquet", index=False)
            (staging / "report.md").write_text(_report(summary), encoding="utf-8")
            artifact_hashes = {
                name: file_sha256(staging / name) for name in sorted(REQUIRED_ARTIFACTS)
            }
            manifest: JsonObject = {
                "schema_version": SCANNER_SCHEMA_VERSION,
                "artifact_name": ARTIFACT_NAME,
                "scan_id": summary["scan_id"],
                "status": summary["status"],
                "start_date": summary["start_date"],
                "end_date": summary["end_date"],
                "policy_version": self.policy.policy_version,
                "policy_hash": self.policy.policy_hash,
                "security_identity_mapping_version": self.identity.mapping_version,
                "security_identity_mapping_hash": self.identity.mapping_hash,
                "security_identity_transition_version": self.identity_transitions.artifact_version,
                "security_identity_transition_hash": self.identity_transitions.artifact_hash,
                "lifecycle_evidence_version": self.evidence.policy_version,
                "lifecycle_evidence_hash": self.evidence.policy_hash,
                "source_inventory_hash": canonical_payload_hash(source_inventory),
                "supersedes_scan_id": summary.get("supersedes_scan_id"),
                "supersession_reason": summary.get("supersession_reason"),
                "logical_identity": logical_identity,
                "counts": summary["counts"],
                "artifact_hashes": artifact_hashes,
                "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            atomic_write_json(staging / "manifest.json", manifest)
            if output.exists():
                raise DataValidationError(f"SECURITY_LIFECYCLE_IMMUTABLE_ARTIFACT_EXISTS: {output}")
            os.replace(staging, output)
            return manifest
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def validate_security_lifecycle_artifact(path: Path) -> JsonObject:
    """Validate a complete lifecycle scan root through every child hash."""

    manifest = _read_json(path / "manifest.json", "lifecycle scan manifest")
    if (
        manifest.get("schema_version")
        not in LEGACY_SCANNER_SCHEMA_VERSIONS | {SCANNER_SCHEMA_VERSION}
        or manifest.get("artifact_name") != ARTIFACT_NAME
        or manifest.get("status") not in {"PASS", "BLOCKED"}
        or path.name != manifest.get("scan_id")
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_MANIFEST_INVALID")
    schema_version = int(manifest["schema_version"])
    required_artifacts = (
        REQUIRED_ARTIFACTS
        if schema_version in {3, SCANNER_SCHEMA_VERSION}
        else LEGACY_REQUIRED_ARTIFACTS
    )
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict) or set(hashes) != required_artifacts:
        raise DataValidationError("SECURITY_LIFECYCLE_ARTIFACT_SET_MISMATCH")
    if {item.name for item in path.iterdir()} != required_artifacts | {"manifest.json"}:
        raise DataValidationError("SECURITY_LIFECYCLE_ARTIFACT_SET_MISMATCH")
    for name, expected in hashes.items():
        child = path / str(name)
        if not isinstance(expected, str) or file_sha256(child) != expected:
            raise DataValidationError(f"SECURITY_LIFECYCLE_ARTIFACT_HASH_MISMATCH: {name}")
    inventory = _read_json(path / "source_inventory.json", "lifecycle source inventory")
    if canonical_payload_hash(inventory) != manifest.get("source_inventory_hash"):
        raise DataValidationError("SECURITY_LIFECYCLE_SOURCE_INVENTORY_HASH_MISMATCH")
    logical = manifest.get("logical_identity")
    if not isinstance(logical, dict):
        raise DataValidationError("SECURITY_LIFECYCLE_MANIFEST_INVALID")
    expected_id = f"security_lifecycle_{canonical_payload_hash(logical)[:24]}"
    if expected_id != manifest.get("scan_id"):
        raise DataValidationError("SECURITY_LIFECYCLE_SCAN_ID_MISMATCH")
    if schema_version == SCANNER_SCHEMA_VERSION:
        _validate_lifecycle_business_contents(path, manifest)
    return manifest


def _validate_lifecycle_business_contents(path: Path, manifest: JsonObject) -> None:
    """Recompute evidence-grade gate semantics from immutable child artifacts."""

    summary = _read_json(path / "summary.json", "lifecycle scan summary")
    counts = manifest.get("counts")
    required_counts = {
        "canonical_securities",
        "open_sessions",
        "listed_session_candidates",
        "valid_price_rows",
        "missing_price_candidates",
        "ordinary_suspension",
        "ordinary_suspension_intervals",
        "listing_suspension",
        "listing_suspension_intervals",
        "terminal_delisting",
        "terminal_delisting_intervals",
        "security_alias",
        "security_alias_intervals",
        "missing_raw_data",
        "missing_raw_data_intervals",
        "unresolved",
        "unresolved_intervals",
        "policy_collisions",
        "boundary_inconsistencies",
        "boundary_warnings",
        "boundary_info",
        "blocking_raw_data_gaps",
    }
    if (
        not isinstance(counts, dict)
        or not required_counts.issubset(counts)
        or any(
            isinstance(counts[key], bool) or not isinstance(counts[key], int)
            for key in required_counts
        )
        or any(int(counts[key]) < 0 for key in required_counts)
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_COUNTS_INVALID")
    if summary.get("counts") != counts or summary.get("status") != manifest.get("status"):
        raise DataValidationError("SECURITY_LIFECYCLE_STATUS_MISMATCH")
    for key in (
        "scan_id",
        "start_date",
        "end_date",
        "policy_version",
        "policy_hash",
        "lifecycle_evidence_version",
        "lifecycle_evidence_hash",
        "security_identity_mapping_version",
        "security_identity_mapping_hash",
        "security_identity_transition_version",
        "security_identity_transition_hash",
    ):
        if summary.get(key) != manifest.get(key):
            raise DataValidationError("SECURITY_LIFECYCLE_SUMMARY_MISMATCH")

    classified = pd.read_parquet(path / "classified_gaps.parquet")
    unresolved = pd.read_parquet(path / "unresolved.parquet")
    boundaries = pd.read_parquet(path / "boundary_checks.parquet")
    raw_gaps = pd.read_parquet(path / "raw_data_gaps.parquet")
    required_gap_columns = {
        "canonical_ts_code",
        "gap_start",
        "gap_end",
        "session_count",
        "classification",
        "blocking",
    }
    if not required_gap_columns.issubset(classified.columns):
        raise DataValidationError("SECURITY_LIFECYCLE_CLASSIFIED_GAPS_INVALID")
    if classified.duplicated(
        ["canonical_ts_code", "gap_start", "gap_end", "classification"], keep=False
    ).any():
        raise DataValidationError("SECURITY_LIFECYCLE_CLASSIFIED_GAPS_DUPLICATE")
    for row in classified.itertuples(index=False):
        if (
            not _is_real_date(str(row.gap_start))
            or not _is_real_date(str(row.gap_end))
            or str(row.gap_start) > str(row.gap_end)
            or isinstance(row.session_count, bool)
            or int(cast(Any, row).session_count) <= 0
        ):
            raise DataValidationError("SECURITY_LIFECYCLE_CLASSIFIED_GAPS_INVALID")
    for _, group in classified.groupby("canonical_ts_code", sort=False):
        ordered = group.sort_values(["gap_start", "gap_end"])
        previous_end: str | None = None
        for row in ordered.itertuples(index=False):
            if previous_end is not None and str(row.gap_start) <= previous_end:
                raise DataValidationError("SECURITY_LIFECYCLE_CLASSIFIED_GAPS_OVERLAP")
            previous_end = str(row.gap_end)

    recomputed: dict[str, int] = {}
    for classification in (
        "ORDINARY_SUSPENSION",
        "LISTING_SUSPENSION",
        "TERMINAL_DELISTING",
        "SECURITY_ALIAS",
        "MISSING_RAW_DATA",
        "UNRESOLVED",
    ):
        selected = classified[classified["classification"].astype(str).eq(classification)]
        key = classification.lower()
        recomputed[key] = int(selected["session_count"].astype(int).sum())
        recomputed[f"{key}_intervals"] = int(len(selected))
    gap_classifications = {
        "ORDINARY_SUSPENSION",
        "LISTING_SUSPENSION",
        "MISSING_RAW_DATA",
        "UNRESOLVED",
    }
    recomputed["missing_price_candidates"] = int(
        classified.loc[
            classified["classification"].astype(str).isin(gap_classifications),
            "session_count",
        ]
        .astype(int)
        .sum()
    )
    recomputed["policy_collisions"] = int(
        boundaries["status"].astype(str).eq("POLICY_COLLISION").sum()
    )
    recomputed["boundary_inconsistencies"] = int(
        (
            boundaries["status"].astype(str).eq("BOUNDARY_INCONSISTENCY")
            & boundaries["blocking"].fillna(False).astype(bool)
        ).sum()
    )
    recomputed["boundary_warnings"] = int(boundaries["status"].astype(str).eq("WARNING").sum())
    recomputed["boundary_info"] = int(boundaries["status"].astype(str).eq("INFO").sum())
    recomputed["blocking_raw_data_gaps"] = int(
        raw_gaps["blocking"].fillna(False).astype(bool).sum()
    )
    if any(int(counts[key]) != value for key, value in recomputed.items()):
        raise DataValidationError("SECURITY_LIFECYCLE_COUNTS_MISMATCH")
    expected_unresolved = classified[classified["blocking"].fillna(False).astype(bool)].reset_index(
        drop=True
    )
    if not _frames_equal_by_value(unresolved, expected_unresolved):
        raise DataValidationError("SECURITY_LIFECYCLE_UNRESOLVED_MISMATCH")
    expected_status = _scan_status(cast(JsonObject, counts))
    if expected_status != manifest.get("status"):
        raise DataValidationError("SECURITY_LIFECYCLE_STATUS_MISMATCH")

    logical = cast(JsonObject, manifest["logical_identity"])
    logical_pairs = {
        "scanner_schema_version": manifest["schema_version"],
        "start_date": manifest["start_date"],
        "end_date": manifest["end_date"],
        "source_inventory_hash": manifest["source_inventory_hash"],
        "security_identity_mapping_version": manifest["security_identity_mapping_version"],
        "security_identity_mapping_hash": manifest["security_identity_mapping_hash"],
        "security_identity_transition_version": manifest["security_identity_transition_version"],
        "security_identity_transition_hash": manifest["security_identity_transition_hash"],
        "lifecycle_policy_version": manifest["policy_version"],
        "lifecycle_policy_hash": manifest["policy_hash"],
        "lifecycle_evidence_version": manifest["lifecycle_evidence_version"],
        "lifecycle_evidence_hash": manifest["lifecycle_evidence_hash"],
    }
    if any(logical.get(key) != value for key, value in logical_pairs.items()):
        raise DataValidationError("SECURITY_LIFECYCLE_LOGICAL_IDENTITY_MISMATCH")


def _frames_equal_by_value(left: DataFrame, right: DataFrame) -> bool:
    if set(left.columns) != set(right.columns) or len(left) != len(right):
        return False
    columns = sorted(left.columns)
    left_rows = (
        left[columns].fillna("<NULL>").astype(str).sort_values(columns).reset_index(drop=True)
    )
    right_rows = (
        right[columns].fillna("<NULL>").astype(str).sort_values(columns).reset_index(drop=True)
    )
    return left_rows.equals(right_rows)


def _is_real_date(value: str) -> bool:
    try:
        return datetime.strptime(value, "%Y%m%d").strftime("%Y%m%d") == value
    except ValueError:
        return False


def validate_pass_lifecycle_scan(
    manifest_path: Path,
    *,
    required_start: str,
    required_end: str,
    raw_root: Path | None = None,
    processed_root: Path | None = None,
    identity_resolver: SecurityIdentityResolver | None = None,
    lifecycle_evidence: SecurityLifecycleResolver | None = None,
    lifecycle_policy: LifecycleAuditPolicy | None = None,
    identity_transitions: SecurityIdentityTransitionResolver | None = None,
) -> JsonObject:
    """Require one intact PASS scan covering the executable data interval."""

    manifest = validate_security_lifecycle_artifact(manifest_path.parent)
    if manifest_path != manifest_path.parent / "manifest.json":
        raise DataValidationError("SECURITY_LIFECYCLE_MANIFEST_INVALID")
    if manifest.get("schema_version") != SCANNER_SCHEMA_VERSION:
        raise DataValidationError("SECURITY_LIFECYCLE_AUDIT_REQUIRED: legacy scan contract")
    if manifest.get("status") != "PASS":
        raise DataValidationError("SECURITY_LIFECYCLE_AUDIT_REQUIRED: scan is not PASS")
    if (
        str(manifest.get("start_date")) > required_start
        or str(manifest.get("end_date")) < required_end
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_COVERAGE_INSUFFICIENT")
    if raw_root is not None or processed_root is not None:
        if raw_root is None or processed_root is None:
            raise DataValidationError("SECURITY_LIFECYCLE_SOURCE_MISMATCH")
        current = _build_source_inventory(
            {
                "trade_cal": raw_root / "trade_cal",
                "stock_basic": raw_root / "stock_basic",
                "daily": raw_root / "daily",
                "suspend_d": raw_root / "suspend_d",
                "universe_daily": processed_root / "universe_daily",
            }
        )
        if canonical_payload_hash(current) != manifest.get("source_inventory_hash"):
            raise DataValidationError("SECURITY_LIFECYCLE_SOURCE_MISMATCH")
    if identity_resolver is not None and (
        manifest.get("security_identity_mapping_version") != identity_resolver.mapping_version
        or manifest.get("security_identity_mapping_hash") != identity_resolver.mapping_hash
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_SOURCE_MISMATCH: identity mapping")
    if identity_transitions is None:
        raise DataValidationError("SECURITY_IDENTITY_TRANSITION_CONTRACT_REQUIRED")
    if (
        manifest.get("security_identity_transition_version")
        != identity_transitions.artifact_version
        or manifest.get("security_identity_transition_hash") != identity_transitions.artifact_hash
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_SOURCE_MISMATCH: identity transitions")
    if lifecycle_evidence is not None and (
        manifest.get("lifecycle_evidence_version") != lifecycle_evidence.policy_version
        or manifest.get("lifecycle_evidence_hash") != lifecycle_evidence.policy_hash
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_SOURCE_MISMATCH: lifecycle evidence")
    if lifecycle_policy is not None and (
        manifest.get("policy_version") != lifecycle_policy.policy_version
        or manifest.get("policy_hash") != lifecycle_policy.policy_hash
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_SOURCE_MISMATCH: lifecycle policy")
    return manifest


def _compress_classified(frame: DataFrame) -> DataFrame:
    columns = [
        "canonical_ts_code",
        "source_ts_codes",
        "trade_date",
        "gap_start",
        "gap_end",
        "session_count",
        "classification",
        "disposition",
        "blocking",
        "list_date",
        "delist_date",
        "suspension_start",
        "suspension_resume",
        "evidence_types",
        "evidence_ids",
        "reason",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    working = frame.sort_values(["canonical_ts_code", "classification", "session_index"]).copy()
    group_fields = [
        "canonical_ts_code",
        "classification",
        "disposition",
        "blocking",
        "evidence_start",
        "evidence_end",
        "evidence_ids",
        "reason",
        "list_date",
        "delist_date",
    ]
    previous = working.groupby(group_fields, dropna=False)["session_index"].shift()
    working["new_interval"] = previous.isna() | working["session_index"].sub(previous).ne(1)
    working["interval_id"] = working.groupby(group_fields, dropna=False)["new_interval"].cumsum()
    rows: list[JsonObject] = []
    for _, group in working.groupby([*group_fields, "interval_id"], dropna=False, sort=True):
        first = group.iloc[0]
        classification = str(first["classification"])
        rows.append(
            {
                "canonical_ts_code": str(first["canonical_ts_code"]),
                "source_ts_codes": ",".join(
                    sorted(
                        {
                            item
                            for value in group["source_ts_codes"].fillna("").astype(str)
                            for item in value.split(",")
                            if item
                        }
                    )
                ),
                "trade_date": str(group["trade_date"].min()),
                "gap_start": str(group["trade_date"].min()),
                "gap_end": str(group["trade_date"].max()),
                "session_count": int(len(group)),
                "classification": classification,
                "disposition": str(first["disposition"]),
                "blocking": bool(first["blocking"]),
                "list_date": _nullable_string(first["list_date"]),
                "delist_date": _nullable_string(first["delist_date"]),
                "suspension_start": (
                    _nullable_string(first["evidence_start"])
                    if classification in {"ORDINARY_SUSPENSION", "LISTING_SUSPENSION"}
                    else None
                ),
                "suspension_resume": (
                    _nullable_string(first["evidence_end"])
                    if classification in {"ORDINARY_SUSPENSION", "LISTING_SUSPENSION"}
                    else None
                ),
                "evidence_types": _evidence_type(classification),
                "evidence_ids": str(first["evidence_ids"]),
                "reason": str(first["reason"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_source_inventory(paths: dict[str, Path]) -> JsonObject:
    datasets: JsonObject = {}
    for name, root in sorted(paths.items()):
        files = sorted(root.glob("**/*.parquet"))
        logical_files = [
            {
                "path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in files
        ]
        manifest_path = root / "_manifest.json"
        datasets[name] = {
            "file_count": len(logical_files),
            "total_bytes": sum(cast(int, item["size"]) for item in logical_files),
            "content_hash": canonical_payload_hash(logical_files),
            "manifest_hash": file_sha256(manifest_path) if manifest_path.is_file() else None,
        }
    return {"schema_version": 1, "datasets": datasets}


def _ordinary_interval(code: str, start: str, end: str) -> JsonObject:
    identity = {"code": code, "start": start, "end": end, "semantics": "DAILY_S_SNAPSHOTS"}
    return {
        "canonical_ts_code": code,
        "state": "ORDINARY_SUSPENSION",
        "effective_start": start,
        "effective_end": end,
        "start_source": "SUSPEND_D_S",
        "end_source": "LAST_CONSECUTIVE_SUSPEND_D_S_SNAPSHOT",
        "evidence_hash": canonical_payload_hash(identity),
    }


def _boundary(
    code: str,
    date: str,
    boundary_type: str,
    status: str,
    reason: str,
    *,
    blocking: bool,
) -> JsonObject:
    return {
        "canonical_ts_code": code,
        "trade_date": date,
        "boundary_type": boundary_type,
        "status": status,
        "blocking": blocking,
        "reason": reason,
        "evidence": "suspend_d",
    }


def _classification_sets(frame: DataFrame) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    if frame.empty:
        return result
    for code, group in frame.groupby("canonical_ts_code", sort=True):
        result[str(code)] = set(group["classification"].astype(str))
    return result


def _old_unresolved_outcomes(
    old_unresolved: DataFrame,
    new_classified: DataFrame,
    new_unresolved: DataFrame,
) -> JsonObject:
    old_codes = sorted(set(old_unresolved["canonical_ts_code"].astype(str)))
    details: dict[str, str] = {}
    for code in old_codes:
        old_rows = old_unresolved[old_unresolved["canonical_ts_code"].astype(str).eq(code)]
        old_periods = tuple(
            (str(row.gap_start), str(row.gap_end)) for row in old_rows.itertuples(index=False)
        )
        new_rows = new_classified[new_classified["canonical_ts_code"].astype(str).eq(code)]
        overlapping = _overlapping_rows(new_rows, old_periods)
        classifications = set(overlapping["classification"].astype(str))
        unresolved_overlap = _overlapping_rows(
            new_unresolved[new_unresolved["canonical_ts_code"].astype(str).eq(code)],
            old_periods,
        )
        if not unresolved_overlap.empty:
            outcome = "STILL_UNRESOLVED"
        elif "MISSING_RAW_DATA" in classifications:
            outcome = "NOW_RAW_DATA_GAP"
        elif "LISTING_SUSPENSION" in classifications:
            outcome = "NOW_LISTING_SUSPENSION"
        elif "TERMINAL_DELISTING" in classifications:
            outcome = "NOW_TERMINAL"
        elif "SECURITY_ALIAS" in classifications:
            outcome = "NOW_SECURITY_ALIAS"
        else:
            outcome = "RESOLVED_BY_SEMANTIC_FIX"
        details[code] = outcome
    counts = {name: list(details.values()).count(name) for name in sorted(set(details.values()))}
    return {"counts": counts, "securities": details}


def _overlapping_rows(frame: DataFrame, periods: tuple[tuple[str, str], ...]) -> DataFrame:
    if frame.empty:
        return frame.copy()
    mask = [
        any(str(row.gap_start) <= end and str(row.gap_end) >= start for start, end in periods)
        for row in frame.itertuples(index=False)
    ]
    return frame.loc[mask]


def _same_day_summary(frame: DataFrame) -> JsonObject:
    if frame.empty:
        return {
            "total": 0,
            "valid_daily_quote": 0,
            "no_daily_quote": 0,
            "intraday_s": 0,
            "full_day_s": 0,
            "overlaps_listing_suspension": 0,
            "terminal": 0,
            "neither": 0,
            "blocking": 0,
            "non_blocking": 0,
        }
    valid = frame["valid_daily_quote"].fillna(False).astype(bool)
    listing = frame["overlaps_listing_suspension"].fillna(False).astype(bool)
    terminal = frame["terminal_effective"].fillna(False).astype(bool)
    blocking = frame["blocking"].fillna(False).astype(bool)
    return {
        "total": int(len(frame)),
        "valid_daily_quote": int(valid.sum()),
        "no_daily_quote": int((~valid).sum()),
        "intraday_s": int(frame["has_intraday_s"].fillna(False).astype(bool).sum()),
        "full_day_s": int(frame["has_full_day_s"].fillna(False).astype(bool).sum()),
        "overlaps_listing_suspension": int(listing.sum()),
        "terminal": int(terminal.sum()),
        "neither": int((~listing & ~terminal).sum()),
        "blocking": int(blocking.sum()),
        "non_blocking": int((~blocking).sum()),
    }


def _scan_status(counts: JsonObject) -> Literal["PASS", "BLOCKED"]:
    blockers = (
        int(counts.get("unresolved", 0)),
        int(counts.get("missing_raw_data", 0)),
        int(counts.get("policy_collisions", 0)),
        int(counts.get("boundary_inconsistencies", 0)),
        int(counts.get("blocking_raw_data_gaps", 0)),
    )
    return "PASS" if not any(blockers) else "BLOCKED"


def _report(summary: JsonObject) -> str:
    counts = cast(JsonObject, summary["counts"])
    lines = [
        "# Security Lifecycle Scan",
        "",
        f"- Scan: `{summary['scan_id']}`",
        f"- Status: **{summary['status']}**",
        f"- Period: `{summary['start_date']}..{summary['end_date']}`",
        f"- Policy: `{summary['policy_version']}`",
        f"- Evidence: `{summary['lifecycle_evidence_version']}`",
        "",
        "## Counts",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(counts.items()))
    comparison = summary.get("supersession_comparison")
    if isinstance(comparison, dict) and comparison.get("old_scan_id"):
        lines.extend(
            [
                "",
                "## Supersession",
                "",
                f"- Supersedes: `{comparison['old_scan_id']}`",
                f"- Reason: `{comparison['supersession_reason']}`",
                "",
                "### Count Delta",
                "",
            ]
        )
        for key, value in sorted(cast(JsonObject, comparison["count_delta"]).items()):
            item = cast(JsonObject, value)
            lines.append(f"- {key}: {item['old']} -> {item['new']} ({item['delta']:+d})")
    lines.extend(
        [
            "",
            "Classification does not imply safe execution: raw-data loss and unresolved "
            "gaps remain blocking.",
        ]
    )
    return "\n".join(lines) + "\n"


def _evidence_type(classification: str) -> str:
    return {
        "ORDINARY_SUSPENSION": "SUSPEND_D",
        "LISTING_SUSPENSION": "OPERATOR_VERIFIED_OFFICIAL_EVIDENCE",
        "TERMINAL_DELISTING": "STOCK_BASIC",
        "SECURITY_ALIAS": "SECURITY_IDENTITY_MAPPING",
        "MISSING_RAW_DATA": "SOURCE_INVENTORY",
        "UNRESOLVED": "INSUFFICIENT_AUTHORITY",
    }[classification]


def _nullable_string(value: object) -> str | None:
    return (
        None
        if value is None or value is pd.NA or str(value) in {"", "None", "<NA>", "nan"}
        else str(value)
    )


def _read_json(path: Path, description: str) -> JsonObject:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"cannot read {description}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"{description} must contain an object")
    return payload


def _parquet_glob(root: Path) -> str:
    return str(root / "**" / "*.parquet").replace("'", "''")


def _validate_date(value: str, name: str) -> None:
    if len(value) != 8 or not value.isdigit():
        raise DataValidationError(f"{name} must be YYYYMMDD")


def canonical_payload_hash(payload: object) -> str:
    """Hash one JSON-compatible logical payload without filesystem metadata."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash one required artifact file."""

    if not path.is_file():
        raise DataValidationError(f"required artifact does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()
