"""Deterministic read-only triage for residual lifecycle scan findings."""

# ruff: noqa: S608 -- SQL paths are escaped local paths and codes are joined as data.

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import duckdb
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
from ashare_quant.universe.builder import add_listing_flags
from ashare_quant.utils.manifest import atomic_write_json

type DataFrame = pd.DataFrame
type JsonObject = dict[str, Any]
type TriageCategory = Literal[
    "LIKELY_FORMAL_LISTING_SUSPENSION",
    "LIKELY_ORDINARY_SUSPENSION_DATA_GAP",
    "LIKELY_PROVIDER_SUSPEND_D_GAP",
    "LIKELY_TERMINAL_TRANSITION",
    "LIKELY_SECURITY_IDENTITY",
    "LIKELY_RAW_DATA_GAP",
    "SAME_DAY_SR_REQUIRES_EVIDENCE",
    "UNKNOWN_REQUIRES_OFFICIAL_EVIDENCE",
]

TRIAGE_SCHEMA_VERSION = 1
TRIAGE_CONTRACT_VERSION = 2
ARTIFACT_NAME = "security_lifecycle_triage"
REQUIRED_ARTIFACTS = frozenset(
    {
        "summary.json",
        "unresolved_triage.parquet",
        "same_day_sr_evidence_requests.parquet",
        "terminal_reconciliation.parquet",
        "listing_suspension_reconciliation.parquet",
        "evidence_requests.json",
    }
)


@dataclass(frozen=True, slots=True)
class SecurityLifecycleTriageResult:
    """Published residual-triage result."""

    triage_id: str
    output_dir: Path
    counts: JsonObject
    idempotent: bool


class SecurityLifecycleTriageService:
    """Enrich residual findings without promoting heuristics to lifecycle evidence."""

    def __init__(
        self,
        *,
        lifecycle_scan_manifest: Path,
        raw_root: Path,
        processed_root: Path,
        reports_root: Path,
        identity_resolver: SecurityIdentityResolver,
        lifecycle_evidence: SecurityLifecycleResolver,
        lifecycle_policy: LifecycleAuditPolicy,
    ) -> None:
        self.scan_manifest_path = lifecycle_scan_manifest
        self.raw_root = raw_root
        self.processed_root = processed_root
        self.reports_root = reports_root
        self.identity = identity_resolver
        self.evidence = lifecycle_evidence
        self.policy = lifecycle_policy

    def run(self) -> SecurityLifecycleTriageResult:
        """Triage every unresolved interval from one validated immutable scan."""

        started = time.perf_counter()
        if self.scan_manifest_path.name != "manifest.json":
            raise DataValidationError("SECURITY_LIFECYCLE_TRIAGE_SOURCE_INVALID")
        scan_root = self.scan_manifest_path.parent
        scan_manifest = validate_security_lifecycle_artifact(scan_root)
        if scan_manifest.get("status") != "BLOCKED":
            raise DataValidationError("SECURITY_LIFECYCLE_TRIAGE_REQUIRES_BLOCKED_SCAN")
        self._validate_lineage(scan_manifest)
        logical_identity: JsonObject = {
            "triage_schema_version": TRIAGE_SCHEMA_VERSION,
            "triage_contract_version": TRIAGE_CONTRACT_VERSION,
            "source_scan_id": scan_manifest["scan_id"],
            "source_scan_manifest_hash": file_sha256(self.scan_manifest_path),
            "security_identity_mapping_version": self.identity.mapping_version,
            "security_identity_mapping_hash": self.identity.mapping_hash,
            "lifecycle_policy_version": self.policy.policy_version,
            "lifecycle_policy_hash": self.policy.policy_hash,
            "lifecycle_evidence_version": self.evidence.policy_version,
            "lifecycle_evidence_hash": self.evidence.policy_hash,
        }
        triage_id = f"security_lifecycle_triage_{canonical_payload_hash(logical_identity)[:24]}"
        output = self.reports_root / "security_lifecycle_triage" / triage_id
        if output.exists():
            manifest = validate_security_lifecycle_triage_artifact(output)
            return SecurityLifecycleTriageResult(
                triage_id=triage_id,
                output_dir=output,
                counts=cast(JsonObject, manifest["counts"]),
                idempotent=True,
            )

        load_started = time.perf_counter()
        unresolved = pd.read_parquet(scan_root / "unresolved.parquet")
        boundaries = pd.read_parquet(scan_root / "boundary_checks.parquet")
        same_day = pd.read_parquet(scan_root / "same_day_sr_diagnostics.parquet")
        source_frames = self._load_relevant_sources(unresolved, boundaries)
        load_seconds = time.perf_counter() - load_started

        classify_started = time.perf_counter()
        triage = self._triage_unresolved(unresolved, same_day, source_frames)
        same_day_requests = self._same_day_requests(same_day, source_frames)
        terminal = self._terminal_reconciliation(boundaries, source_frames)
        listing = self._listing_reconciliation(source_frames)
        classify_seconds = time.perf_counter() - classify_started
        counts = _counts(triage, same_day_requests, terminal, listing)
        evidence_requests = _evidence_requests(triage, same_day_requests)
        summary: JsonObject = {
            "triage_id": triage_id,
            "source_scan_id": scan_manifest["scan_id"],
            "source_scan_manifest_hash": logical_identity["source_scan_manifest_hash"],
            "counts": counts,
            "clusters": _cluster_summary(triage),
            "profile_seconds": {
                "source_validation_and_load": round(load_seconds, 6),
                "classification_and_reconciliation": round(classify_seconds, 6),
                "total_before_publication": round(time.perf_counter() - started, 6),
            },
            "notice": "Triage categories are diagnostic candidates, not lifecycle evidence.",
        }
        manifest = self._publish(
            output=output,
            logical_identity=logical_identity,
            summary=summary,
            frames={
                "unresolved_triage": triage,
                "same_day_sr_evidence_requests": same_day_requests,
                "terminal_reconciliation": terminal,
                "listing_suspension_reconciliation": listing,
            },
            evidence_requests=evidence_requests,
        )
        return SecurityLifecycleTriageResult(
            triage_id=triage_id,
            output_dir=output,
            counts=cast(JsonObject, manifest["counts"]),
            idempotent=False,
        )

    def _validate_lineage(self, manifest: JsonObject) -> None:
        expected = {
            "security_identity_mapping_hash": self.identity.mapping_hash,
            "policy_hash": self.policy.policy_hash,
            "lifecycle_evidence_hash": self.evidence.policy_hash,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise DataValidationError("SECURITY_LIFECYCLE_TRIAGE_LINEAGE_MISMATCH")

    def _load_relevant_sources(
        self, unresolved: DataFrame, boundaries: DataFrame
    ) -> dict[str, DataFrame]:
        codes = set(unresolved["canonical_ts_code"].astype(str))
        blocking = boundaries[boundaries["blocking"].fillna(False).astype(bool)]
        codes.update(blocking["canonical_ts_code"].astype(str))
        codes.update(event.canonical_ts_code for event in self.evidence.events())
        source_codes = self.identity.source_codes_for(codes)
        selected = pd.DataFrame({"source_ts_code": source_codes})
        con = duckdb.connect()
        try:
            con.register("selected_codes", selected)
            daily = self._query_selected(con, "daily", "ts_code,trade_date,open,high,low,close")
            suspend = self._query_selected(
                con, "suspend_d", "ts_code,trade_date,suspend_type,suspend_timing"
            )
            stock = self._query_selected(
                con, "stock_basic", "ts_code,name,market,list_date,delist_date"
            )
            universe = self._query_selected(
                con,
                "universe_daily",
                "ts_code,trade_date,is_listed,is_suspended,can_buy,can_sell",
                processed=True,
            )
            trade_cal = con.execute(
                f"""SELECT DISTINCT cast(cal_date AS VARCHAR) trade_date
                FROM read_parquet('{_parquet_glob(self.raw_root / "trade_cal")}',
                  hive_partitioning=false,union_by_name=true)
                WHERE cast(is_open AS INTEGER)=1 ORDER BY 1"""
            ).fetchdf()
        finally:
            con.close()
        for _name, frame in (("daily", daily), ("suspend_d", suspend), ("stock_basic", stock)):
            if frame.empty:
                continue
            frame["source_ts_code"] = frame["ts_code"].astype(str).str.upper()
            date_column = "trade_date" if "trade_date" in frame.columns else None
            frame["canonical_ts_code"] = [
                self.identity.canonicalize(
                    code,
                    as_of_date=(str(date) if date_column is not None else None),
                )
                for code, date in zip(
                    frame["source_ts_code"],
                    frame[date_column] if date_column is not None else [None] * len(frame),
                    strict=True,
                )
            ]
        if not daily.empty:
            prices = daily[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
            daily["valid_quote"] = prices.notna().all(axis=1) & (prices > 0).all(axis=1)
        universe["canonical_ts_code"] = universe["ts_code"].astype(str).str.upper()
        return {
            "daily": daily,
            "suspend_d": suspend,
            "stock_basic": stock,
            "universe": universe,
            "trade_cal": trade_cal,
        }

    def _query_selected(
        self,
        con: duckdb.DuckDBPyConnection,
        dataset: str,
        columns: str,
        *,
        processed: bool = False,
    ) -> DataFrame:
        root = self.processed_root if processed else self.raw_root
        dataset_root = root / dataset
        if not any(dataset_root.glob("**/*.parquet")):
            raise DataValidationError(f"SECURITY_LIFECYCLE_TRIAGE_SOURCE_MISSING: {dataset}")
        return con.execute(
            f"""SELECT {columns} FROM read_parquet('{_parquet_glob(dataset_root)}',
                hive_partitioning=false,union_by_name=true) d
                JOIN selected_codes s ON cast(d.ts_code AS VARCHAR)=s.source_ts_code"""
        ).fetchdf()

    def _triage_unresolved(
        self, unresolved: DataFrame, same_day: DataFrame, sources: dict[str, DataFrame]
    ) -> DataFrame:
        daily = sources["daily"]
        suspend = sources["suspend_d"]
        stock = sources["stock_basic"]
        same_day_keys = {
            (str(row.canonical_ts_code), str(row.trade_date))
            for row in same_day[same_day["blocking"].fillna(False).astype(bool)].itertuples()
        }
        event_rows = self.evidence.events()
        rows: list[JsonObject] = []
        for item in unresolved.sort_values(
            ["canonical_ts_code", "gap_start", "gap_end"]
        ).itertuples(index=False):
            code = str(item.canonical_ts_code)
            start, end = str(item.gap_start), str(item.gap_end)
            code_daily = daily[
                daily["canonical_ts_code"].astype(str).eq(code)
                & daily["valid_quote"].fillna(False).astype(bool)
            ]
            before = code_daily[code_daily["trade_date"].astype(str) < start]
            after = code_daily[code_daily["trade_date"].astype(str) > end]
            previous_quote = None if before.empty else str(before["trade_date"].astype(str).max())
            next_quote = None if after.empty else str(after["trade_date"].astype(str).min())
            code_suspend = suspend[suspend["canonical_ts_code"].astype(str).eq(code)].copy()
            nearby = code_suspend[
                code_suspend["trade_date"]
                .astype(str)
                .between(
                    _calendar_offset(sources["trade_cal"], start, -5),
                    _calendar_offset(sources["trade_cal"], end, 5),
                )
            ]
            event_types = set(nearby["suspend_type"].fillna("").astype(str).str.upper())
            timing = nearby.get("suspend_timing", pd.Series(dtype="object"))
            has_timed_s = bool(
                (
                    nearby["suspend_type"].fillna("").astype(str).str.upper().eq("S")
                    & timing.fillna("").astype(str).str.strip().ne("")
                ).any()
            )
            overlaps_v3 = any(
                event.canonical_ts_code == code
                and event.effective_from <= end
                and event.effective_to >= start
                for event in event_rows
            )
            aliases = sorted(
                set(
                    daily.loc[
                        daily["canonical_ts_code"].astype(str).eq(code), "source_ts_code"
                    ].astype(str)
                )
                - {code}
            )
            metadata = stock[stock["canonical_ts_code"].astype(str).eq(code)]
            delist_date = (
                None
                if metadata.empty or pd.isna(metadata.iloc[-1].get("delist_date"))
                else str(metadata.iloc[-1]["delist_date"])
            )
            same_day_blocking = any(
                key[0] == code and start <= key[1] <= end for key in same_day_keys
            )
            session_count = int(str(item.session_count))
            category = _triage_category(
                session_count=session_count,
                same_day_blocking=same_day_blocking,
                overlaps_v3=overlaps_v3,
                aliases=aliases,
                delist_date=delist_date,
                gap_end=end,
                next_quote=next_quote,
                has_s="S" in event_types,
                has_r="R" in event_types,
            )
            rows.append(
                {
                    "canonical_ts_code": code,
                    "gap_start": start,
                    "gap_end": end,
                    "session_count": session_count,
                    "triage_category": category,
                    "authoritative_classification_changed": False,
                    "exchange": _exchange(code),
                    "regulatory_period": _regulatory_period(start),
                    "duration_bucket": _duration_bucket(session_count),
                    "quote_resumes": next_quote is not None,
                    "reaches_delist_date": delist_date is not None and end >= delist_date,
                    "previous_valid_quote": previous_quote,
                    "next_valid_quote": next_quote,
                    "nearby_s": "S" in event_types,
                    "nearby_r": "R" in event_types,
                    "nearby_intraday_s": has_timed_s,
                    "existing_v3_evidence": overlaps_v3,
                    "source_aliases": ",".join(aliases),
                    "list_date": getattr(item, "list_date", None),
                    "delist_date": delist_date,
                    "evidence_required": _required_evidence(category),
                }
            )
        return pd.DataFrame(rows)

    def _same_day_requests(self, same_day: DataFrame, sources: dict[str, DataFrame]) -> DataFrame:
        blocking = same_day[same_day["blocking"].fillna(False).astype(bool)].copy()
        triage = pd.DataFrame(
            {
                "canonical_ts_code": blocking["canonical_ts_code"].astype(str),
                "gap_start": blocking["trade_date"].astype(str),
                "gap_end": blocking["trade_date"].astype(str),
                "session_count": 1,
            }
        )
        if triage.empty:
            return pd.DataFrame(
                columns=[
                    "canonical_ts_code",
                    "trade_date",
                    "suspend_rows",
                    "previous_valid_quote",
                    "next_valid_quote",
                    "universe_is_suspended",
                    "exact_missing_information",
                    "evidence_status",
                ]
            )
        enriched = self._triage_unresolved(triage, blocking, sources)
        suspend = sources["suspend_d"]
        rows: list[JsonObject] = []
        for item in enriched.itertuples(index=False):
            selected = suspend[
                suspend["canonical_ts_code"].astype(str).eq(str(item.canonical_ts_code))
                & suspend["trade_date"].astype(str).eq(str(item.gap_start))
            ]
            payload = (
                selected[["source_ts_code", "suspend_type", "suspend_timing"]]
                .fillna("")
                .to_dict("records")
            )
            source = blocking[
                blocking["canonical_ts_code"].astype(str).eq(str(item.canonical_ts_code))
                & blocking["trade_date"].astype(str).eq(str(item.gap_start))
            ].iloc[0]
            rows.append(
                {
                    "canonical_ts_code": item.canonical_ts_code,
                    "trade_date": item.gap_start,
                    "suspend_rows": json.dumps(payload, sort_keys=True, ensure_ascii=True),
                    "previous_valid_quote": item.previous_valid_quote,
                    "next_valid_quote": item.next_valid_quote,
                    "universe_is_suspended": bool(source["universe_is_suspended"]),
                    "exact_missing_information": (
                        "official exchange or issuer evidence resolving whether the same-day "
                        "full-day S/R represents a non-trading session"
                    ),
                    "evidence_status": "OFFICIAL_EVIDENCE_REQUIRED",
                }
            )
        return pd.DataFrame(rows)

    def _terminal_reconciliation(
        self, boundaries: DataFrame, sources: dict[str, DataFrame]
    ) -> DataFrame:
        terminal = boundaries[
            boundaries["boundary_type"].astype(str).eq("TERMINAL_TRANSITION")
            & boundaries["blocking"].fillna(False).astype(bool)
        ][["canonical_ts_code", "trade_date"]].drop_duplicates()
        if terminal.empty:
            return pd.DataFrame(
                columns=[
                    "canonical_ts_code",
                    "trade_date",
                    "old_universe_is_listed",
                    "current_builder_preview_is_listed",
                    "preview_sample",
                    "reconciliation",
                ]
            )
        universe = sources["universe"]
        stock = sources["stock_basic"]
        working = terminal.merge(
            universe[["canonical_ts_code", "trade_date", "is_listed"]].rename(
                columns={"is_listed": "old_universe_is_listed"}
            ),
            on=["canonical_ts_code", "trade_date"],
            how="left",
        )
        metadata = stock[["canonical_ts_code", "list_date", "delist_date"]].drop_duplicates(
            "canonical_ts_code", keep="last"
        )
        working = working.merge(metadata, on="canonical_ts_code", how="left")
        preview = add_listing_flags(
            working[
                [
                    "canonical_ts_code",
                    "trade_date",
                    "list_date",
                    "delist_date",
                ]
            ].rename(columns={"canonical_ts_code": "ts_code"}),
            sorted(sources["trade_cal"]["trade_date"].astype(str).unique()),
        )
        working["current_builder_preview_is_listed"] = preview["is_listed"].to_numpy()
        sample_codes = set(sorted(working["canonical_ts_code"].astype(str).unique())[:10])
        working["preview_sample"] = working["canonical_ts_code"].astype(str).isin(sample_codes)
        old_listed = working["old_universe_is_listed"].fillna(True).astype(bool)
        new_listed = working["current_builder_preview_is_listed"].astype(bool)
        working["reconciliation"] = "CODE_SEMANTICS_REVIEW_REQUIRED"
        working.loc[old_listed & ~new_listed, "reconciliation"] = "STALE_PROCESSED_UNIVERSE"
        return working[
            [
                "canonical_ts_code",
                "trade_date",
                "old_universe_is_listed",
                "current_builder_preview_is_listed",
                "preview_sample",
                "reconciliation",
            ]
        ].sort_values(["canonical_ts_code", "trade_date"])

    def _listing_reconciliation(self, sources: dict[str, DataFrame]) -> DataFrame:
        universe = sources["universe"]
        calendar = sorted(sources["trade_cal"]["trade_date"].astype(str).unique())
        rows: list[JsonObject] = []
        for event in self.evidence.events():
            dates = [
                date for date in calendar if event.effective_from <= date <= event.effective_to
            ]
            if not dates:
                continue
            for boundary_type, date in (
                ("LISTING_SUSPENSION_START", dates[0]),
                ("LISTING_SUSPENSION_END", dates[-1]),
            ):
                current = universe[
                    universe["canonical_ts_code"].astype(str).eq(event.canonical_ts_code)
                    & universe["trade_date"].astype(str).eq(date)
                ]
                rows.append(
                    {
                        "canonical_ts_code": event.canonical_ts_code,
                        "boundary_type": boundary_type,
                        "trade_date": date,
                        "old_universe_is_suspended": (
                            None if current.empty else bool(current.iloc[-1]["is_suspended"])
                        ),
                        "overlay_is_listing_suspended": True,
                        "overlay_is_listed": True,
                        "overlay_can_buy": False,
                        "overlay_can_sell": False,
                        "reconciliation": (
                            "ALREADY_CONSISTENT"
                            if not current.empty and bool(current.iloc[-1]["is_suspended"])
                            else "STALE_PROCESSED_UNIVERSE"
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def _publish(
        self,
        *,
        output: Path,
        logical_identity: JsonObject,
        summary: JsonObject,
        frames: dict[str, DataFrame],
        evidence_requests: JsonObject,
    ) -> JsonObject:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
        try:
            atomic_write_json(staging / "summary.json", summary)
            atomic_write_json(staging / "evidence_requests.json", evidence_requests)
            for name, frame in frames.items():
                frame.to_parquet(staging / f"{name}.parquet", index=False)
            artifact_hashes = {
                name: file_sha256(staging / name) for name in sorted(REQUIRED_ARTIFACTS)
            }
            manifest: JsonObject = {
                "schema_version": TRIAGE_SCHEMA_VERSION,
                "artifact_name": ARTIFACT_NAME,
                "triage_id": summary["triage_id"],
                "source_scan_id": summary["source_scan_id"],
                "source_scan_manifest_hash": summary["source_scan_manifest_hash"],
                "logical_identity": logical_identity,
                "counts": summary["counts"],
                "artifact_hashes": artifact_hashes,
                "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            atomic_write_json(staging / "manifest.json", manifest)
            if output.exists():
                raise DataValidationError(f"SECURITY_LIFECYCLE_TRIAGE_IMMUTABLE: {output}")
            os.replace(staging, output)
            return manifest
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def validate_security_lifecycle_triage_artifact(path: Path) -> JsonObject:
    """Validate a complete immutable triage artifact."""

    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError("SECURITY_LIFECYCLE_TRIAGE_MANIFEST_INVALID") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != TRIAGE_SCHEMA_VERSION
        or manifest.get("artifact_name") != ARTIFACT_NAME
        or manifest.get("triage_id") != path.name
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_TRIAGE_MANIFEST_INVALID")
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict) or set(hashes) != REQUIRED_ARTIFACTS:
        raise DataValidationError("SECURITY_LIFECYCLE_TRIAGE_ARTIFACT_SET_INVALID")
    for name, expected in hashes.items():
        if file_sha256(path / name) != expected:
            raise DataValidationError(f"SECURITY_LIFECYCLE_TRIAGE_HASH_MISMATCH: {name}")
    return cast(JsonObject, manifest)


def _triage_category(
    *,
    session_count: int,
    same_day_blocking: bool,
    overlaps_v3: bool,
    aliases: list[str],
    delist_date: str | None,
    gap_end: str,
    next_quote: str | None,
    has_s: bool,
    has_r: bool,
) -> TriageCategory:
    if same_day_blocking:
        return "SAME_DAY_SR_REQUIRES_EVIDENCE"
    if overlaps_v3:
        return "LIKELY_FORMAL_LISTING_SUSPENSION"
    if aliases:
        return "LIKELY_SECURITY_IDENTITY"
    if delist_date is not None and gap_end >= delist_date:
        return "LIKELY_TERMINAL_TRANSITION"
    if has_s or has_r:
        return "LIKELY_PROVIDER_SUSPEND_D_GAP"
    if next_quote is not None and session_count > 20:
        return "LIKELY_FORMAL_LISTING_SUSPENSION"
    if next_quote is not None and session_count <= 20:
        return "LIKELY_ORDINARY_SUSPENSION_DATA_GAP"
    return "UNKNOWN_REQUIRES_OFFICIAL_EVIDENCE"


def _required_evidence(category: str) -> str:
    return {
        "SAME_DAY_SR_REQUIRES_EVIDENCE": "official full-day trading-status evidence",
        "LIKELY_FORMAL_LISTING_SUSPENSION": "official listing suspension/resumption decision",
        "LIKELY_PROVIDER_SUSPEND_D_GAP": "authoritative suspend_d completeness evidence",
        "LIKELY_ORDINARY_SUSPENSION_DATA_GAP": "official suspension notice or source repair proof",
        "LIKELY_TERMINAL_TRANSITION": "stock_basic terminal contract reconciliation",
        "LIKELY_SECURITY_IDENTITY": "explicit identity mapping validation",
        "LIKELY_RAW_DATA_GAP": "dataset-level ingestion completeness proof",
        "UNKNOWN_REQUIRES_OFFICIAL_EVIDENCE": "official lifecycle evidence or raw-data proof",
    }[category]


def _counts(
    triage: DataFrame, same_day: DataFrame, terminal: DataFrame, listing: DataFrame
) -> JsonObject:
    by_category: JsonObject = {}
    for category, group in triage.groupby("triage_category", sort=True):
        by_category[str(category)] = {
            "intervals": int(len(group)),
            "sessions": int(group["session_count"].sum()),
            "securities": int(group["canonical_ts_code"].nunique()),
        }
    stale_terminal = (
        terminal["reconciliation"].astype(str).eq("STALE_PROCESSED_UNIVERSE")
        if not terminal.empty
        else pd.Series(dtype=bool)
    )
    return {
        "unresolved_intervals": int(len(triage)),
        "unresolved_sessions": int(triage["session_count"].sum()),
        "unresolved_securities": int(triage["canonical_ts_code"].nunique()),
        "by_category": by_category,
        "same_day_sr_evidence_requests": int(len(same_day)),
        "terminal_findings": int(len(terminal)),
        "terminal_stale_processed_universe": int(stale_terminal.sum()),
        "terminal_preview_sample": (
            int(terminal["preview_sample"].sum()) if not terminal.empty else 0
        ),
        "listing_suspension_boundaries": int(len(listing)),
        "listing_stale_processed_universe": int(
            listing["reconciliation"].astype(str).eq("STALE_PROCESSED_UNIVERSE").sum()
        )
        if not listing.empty
        else 0,
    }


def _cluster_summary(triage: DataFrame) -> JsonObject:
    """Return compact deterministic pattern-family aggregates."""

    dimensions = (
        "exchange",
        "regulatory_period",
        "duration_bucket",
        "quote_resumes",
        "reaches_delist_date",
        "nearby_s",
        "nearby_r",
        "nearby_intraday_s",
        "existing_v3_evidence",
    )
    result: JsonObject = {}
    for dimension in dimensions:
        result[dimension] = {
            str(value): {
                "intervals": int(len(group)),
                "sessions": int(group["session_count"].sum()),
                "securities": int(group["canonical_ts_code"].nunique()),
            }
            for value, group in triage.groupby(dimension, dropna=False, sort=True)
        }
    return result


def _evidence_requests(triage: DataFrame, same_day: DataFrame) -> JsonObject:
    requests = triage[
        triage["triage_category"]
        .astype(str)
        .isin(
            {
                "LIKELY_FORMAL_LISTING_SUSPENSION",
                "LIKELY_PROVIDER_SUSPEND_D_GAP",
                "LIKELY_ORDINARY_SUSPENSION_DATA_GAP",
                "UNKNOWN_REQUIRES_OFFICIAL_EVIDENCE",
            }
        )
    ]
    return {
        "schema_version": 1,
        "notice": "Candidate requests do not authorize lifecycle classification.",
        "intervals": requests[
            [
                "canonical_ts_code",
                "gap_start",
                "gap_end",
                "session_count",
                "triage_category",
                "evidence_required",
            ]
        ].to_dict("records"),
        "same_day_sr": same_day.to_dict("records"),
    }


def _duration_bucket(value: int) -> str:
    if value == 1:
        return "1"
    if value <= 5:
        return "2-5"
    if value <= 20:
        return "6-20"
    if value <= 60:
        return "21-60"
    if value <= 120:
        return "61-120"
    return ">120"


def _regulatory_period(date: str) -> str:
    if date < "20210101":
        return "PRE_REFORM_HISTORICAL"
    if date < "20220101":
        return "TRANSITION_2021"
    return "POST_REFORM"


def _exchange(code: str) -> str:
    return {"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}.get(code[-2:], "UNKNOWN")


def _calendar_offset(calendar: DataFrame, date: str, offset: int) -> str:
    dates = sorted(calendar["trade_date"].astype(str).unique())
    if not dates:
        return date
    if date in dates:
        position = dates.index(date)
    else:
        position = next(
            (index for index, value in enumerate(dates) if value >= date), len(dates) - 1
        )
    return str(dates[max(0, min(len(dates) - 1, position + offset))])


def _parquet_glob(root: Path) -> str:
    return str(root / "**" / "*.parquet").replace("'", "''")
