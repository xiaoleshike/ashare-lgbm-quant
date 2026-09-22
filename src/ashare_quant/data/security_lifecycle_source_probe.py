"""Explicit networked provider-completeness probe for lifecycle residuals."""

# ruff: noqa: S608 -- SQL paths are escaped local paths and codes are registered as data.

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import duckdb
import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_identity import SecurityIdentityResolver
from ashare_quant.data.security_lifecycle_audit import canonical_payload_hash, file_sha256
from ashare_quant.data.security_lifecycle_triage import (
    validate_security_lifecycle_triage_artifact,
)
from ashare_quant.utils.manifest import atomic_write_json

type DataFrame = pd.DataFrame
type JsonObject = dict[str, Any]
type SourceCompletenessStatus = Literal[
    "LOCAL_SUSPEND_D_INCOMPLETE",
    "LOCAL_DAILY_INCOMPLETE",
    "LOCAL_BOTH_INCOMPLETE",
    "PROVIDER_HAS_NO_SUSPEND_EVIDENCE",
    "PROVIDER_HAS_NO_DAILY_EVIDENCE",
    "PROVIDER_AND_LOCAL_AGREE",
    "PROVIDER_CONTRADICTS_LOCAL",
    "PROBE_INCONCLUSIVE",
]

PROBE_SCHEMA_VERSION = 1
PROBE_CONTRACT_VERSION = 1
ARTIFACT_NAME = "security_lifecycle_source_probe"
TARGET_CATEGORIES = frozenset(
    {
        "LIKELY_PROVIDER_SUSPEND_D_GAP",
        "LIKELY_ORDINARY_SUSPENSION_DATA_GAP",
    }
)
REQUIRED_ARTIFACTS = frozenset(
    {
        "requests.json",
        "suspend_d_provider.parquet",
        "daily_provider.parquet",
        "comparison.parquet",
        "summary.json",
    }
)


class ProviderClient(Protocol):
    """Minimal query API shared by TushareClient and deterministic test doubles."""

    def query(self, endpoint: str, **params: object) -> DataFrame:
        """Return one provider response."""


@dataclass(frozen=True, slots=True)
class LifecycleSourceProbeResult:
    """Published source-completeness probe result."""

    probe_id: str
    output_dir: Path
    counts: JsonObject
    idempotent: bool


@dataclass(frozen=True, slots=True)
class ProbeWindow:
    """One merged provider query window for a canonical security."""

    canonical_ts_code: str
    source_ts_code: str
    start_date: str
    end_date: str


class SecurityLifecycleSourceProbe:
    """Query current Tushare history without mutating canonical local datasets."""

    def __init__(
        self,
        *,
        triage_manifest: Path,
        raw_root: Path,
        reports_root: Path,
        identity_resolver: SecurityIdentityResolver,
        provider_client: ProviderClient,
        buffer_sessions: int = 5,
    ) -> None:
        self.triage_manifest_path = triage_manifest
        self.raw_root = raw_root
        self.reports_root = reports_root
        self.identity = identity_resolver
        self.client = provider_client
        self.buffer_sessions = buffer_sessions

    def run(self) -> LifecycleSourceProbeResult:
        """Query, freeze and compare provider data for source-focused triage rows."""

        if self.buffer_sessions < 0:
            raise DataValidationError("SECURITY_LIFECYCLE_SOURCE_PROBE_BUFFER_INVALID")
        if self.triage_manifest_path.name != "manifest.json":
            raise DataValidationError("SECURITY_LIFECYCLE_SOURCE_PROBE_TRIAGE_INVALID")
        triage_root = self.triage_manifest_path.parent
        triage_manifest = validate_security_lifecycle_triage_artifact(triage_root)
        logical = cast(JsonObject, triage_manifest.get("logical_identity", {}))
        if logical.get("security_identity_mapping_hash") != self.identity.mapping_hash:
            raise DataValidationError("SECURITY_LIFECYCLE_SOURCE_PROBE_MAPPING_MISMATCH")
        triage = pd.read_parquet(triage_root / "unresolved_triage.parquet")
        targets = triage[triage["triage_category"].astype(str).isin(TARGET_CATEGORIES)].copy()
        if targets.empty:
            raise DataValidationError("SECURITY_LIFECYCLE_SOURCE_PROBE_NO_TARGETS")
        targets["parent_interval_id"] = targets.apply(_interval_id, axis=1)
        _validate_unique_intervals(targets)
        calendar = self._open_sessions()
        windows = _build_windows(targets, calendar, self.identity, self.buffer_sessions)
        requests, provider_frames = self._query_windows(windows)
        local = self._load_local_rows(windows)
        comparison = _compare_intervals(targets, requests, provider_frames, local)
        counts = _probe_counts(comparison, requests)
        response_identity = {
            "suspend_d_provider_hash": _frame_hash(provider_frames["suspend_d"]),
            "daily_provider_hash": _frame_hash(provider_frames["daily"]),
            "requests_content_hash": canonical_payload_hash(
                [_request_identity(item) for item in requests]
            ),
        }
        logical_identity: JsonObject = {
            "probe_schema_version": PROBE_SCHEMA_VERSION,
            "probe_contract_version": PROBE_CONTRACT_VERSION,
            "triage_id": triage_manifest["triage_id"],
            "triage_manifest_hash": file_sha256(self.triage_manifest_path),
            "security_identity_mapping_version": self.identity.mapping_version,
            "security_identity_mapping_hash": self.identity.mapping_hash,
            "buffer_sessions": self.buffer_sessions,
            "provider": "TUSHARE_PRO",
            **response_identity,
        }
        probe_id = (
            f"security_lifecycle_source_probe_{canonical_payload_hash(logical_identity)[:24]}"
        )
        output = self.reports_root / "security_lifecycle_source_probe" / probe_id
        if output.exists():
            manifest = validate_lifecycle_source_probe_artifact(output)
            return LifecycleSourceProbeResult(
                probe_id=probe_id,
                output_dir=output,
                counts=cast(JsonObject, manifest["counts"]),
                idempotent=True,
            )
        summary: JsonObject = {
            "probe_id": probe_id,
            "triage_id": triage_manifest["triage_id"],
            "target_categories": sorted(TARGET_CATEGORIES),
            "counts": counts,
            "notice": "Source-completeness status is not lifecycle classification.",
        }
        manifest = self._publish(
            output=output,
            logical_identity=logical_identity,
            requests=requests,
            provider_frames=provider_frames,
            comparison=comparison,
            summary=summary,
        )
        return LifecycleSourceProbeResult(
            probe_id=probe_id,
            output_dir=output,
            counts=cast(JsonObject, manifest["counts"]),
            idempotent=False,
        )

    def _open_sessions(self) -> tuple[str, ...]:
        path = self.raw_root / "trade_cal"
        if not any(path.glob("**/*.parquet")):
            raise DataValidationError("SECURITY_LIFECYCLE_SOURCE_PROBE_TRADE_CAL_MISSING")
        con = duckdb.connect()
        try:
            frame = con.execute(
                f"""SELECT DISTINCT cast(cal_date AS VARCHAR) trade_date
                FROM read_parquet('{_parquet_glob(path)}',
                  hive_partitioning=false,union_by_name=true)
                WHERE cast(is_open AS INTEGER)=1 ORDER BY 1"""
            ).fetchdf()
        finally:
            con.close()
        return tuple(frame["trade_date"].astype(str))

    def _query_windows(
        self, windows: tuple[ProbeWindow, ...]
    ) -> tuple[list[JsonObject], dict[str, DataFrame]]:
        requests: list[JsonObject] = []
        frames: dict[str, list[DataFrame]] = {"suspend_d": [], "daily": []}
        fields = {
            "suspend_d": "ts_code,trade_date,suspend_timing,suspend_type",
            "daily": "ts_code,trade_date,open,high,low,close,vol,amount",
        }
        for window in windows:
            for endpoint in ("suspend_d", "daily"):
                params: JsonObject = {
                    "ts_code": window.source_ts_code,
                    "start_date": window.start_date,
                    "end_date": window.end_date,
                    "fields": fields[endpoint],
                }
                request_identity = {
                    "provider": "TUSHARE_PRO",
                    "endpoint": endpoint,
                    "canonical_ts_code": window.canonical_ts_code,
                    "params": params,
                }
                request_id = f"request_{canonical_payload_hash(request_identity)[:24]}"
                requested_at = datetime.now(UTC).isoformat(timespec="seconds")
                try:
                    raw = self.client.query(endpoint, **params)
                    normalized = _normalize_provider_frame(
                        raw,
                        endpoint=endpoint,
                        request_id=request_id,
                        identity=self.identity,
                    )
                    raw_hash = _raw_frame_hash(raw)
                    normalized_hash = _frame_hash(normalized)
                    frames[endpoint].append(normalized)
                    requests.append(
                        {
                            **request_identity,
                            "request_id": request_id,
                            "requested_at": requested_at,
                            "client_version": _client_version(),
                            "status": "SUCCESS",
                            "error_class": None,
                            "raw_provider_response_hash": raw_hash,
                            "normalized_response_hash": normalized_hash,
                            "row_count": int(len(normalized)),
                            "minimum_date": _date_bound(normalized, "min"),
                            "maximum_date": _date_bound(normalized, "max"),
                        }
                    )
                except Exception as error:  # noqa: BLE001 - frozen as failure, not data.
                    requests.append(
                        {
                            **request_identity,
                            "request_id": request_id,
                            "requested_at": requested_at,
                            "client_version": _client_version(),
                            "status": "PROBE_FAILED",
                            "error_class": type(error).__name__,
                            "raw_provider_response_hash": None,
                            "normalized_response_hash": None,
                            "row_count": 0,
                            "minimum_date": None,
                            "maximum_date": None,
                        }
                    )
        combined = {endpoint: _concat_frames(items, endpoint) for endpoint, items in frames.items()}
        return requests, combined

    def _load_local_rows(self, windows: tuple[ProbeWindow, ...]) -> dict[str, DataFrame]:
        selected = pd.DataFrame(
            sorted({window.source_ts_code for window in windows}), columns=["source_ts_code"]
        )
        con = duckdb.connect()
        try:
            con.register("selected_codes", selected)
            suspend = _query_local(
                con,
                self.raw_root / "suspend_d",
                "ts_code,trade_date,suspend_timing,suspend_type",
            )
            daily = _query_local(
                con,
                self.raw_root / "daily",
                "ts_code,trade_date,open,high,low,close,vol,amount",
            )
        finally:
            con.close()
        return {
            "suspend_d": _normalize_provider_frame(
                suspend, endpoint="suspend_d", request_id="LOCAL", identity=self.identity
            ),
            "daily": _normalize_provider_frame(
                daily, endpoint="daily", request_id="LOCAL", identity=self.identity
            ),
        }

    def _publish(
        self,
        *,
        output: Path,
        logical_identity: JsonObject,
        requests: list[JsonObject],
        provider_frames: dict[str, DataFrame],
        comparison: DataFrame,
        summary: JsonObject,
    ) -> JsonObject:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
        try:
            atomic_write_json(
                staging / "requests.json",
                {"schema_version": 1, "requests": requests},
            )
            provider_frames["suspend_d"].to_parquet(
                staging / "suspend_d_provider.parquet", index=False
            )
            provider_frames["daily"].to_parquet(staging / "daily_provider.parquet", index=False)
            comparison.to_parquet(staging / "comparison.parquet", index=False)
            atomic_write_json(staging / "summary.json", summary)
            artifact_hashes = {
                name: file_sha256(staging / name) for name in sorted(REQUIRED_ARTIFACTS)
            }
            manifest: JsonObject = {
                "schema_version": PROBE_SCHEMA_VERSION,
                "artifact_name": ARTIFACT_NAME,
                "probe_id": summary["probe_id"],
                "triage_id": summary["triage_id"],
                "logical_identity": logical_identity,
                "counts": summary["counts"],
                "artifact_hashes": artifact_hashes,
                "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            atomic_write_json(staging / "manifest.json", manifest)
            if output.exists():
                raise DataValidationError(f"SECURITY_LIFECYCLE_SOURCE_PROBE_IMMUTABLE: {output}")
            os.replace(staging, output)
            return manifest
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def validate_lifecycle_source_probe_artifact(path: Path) -> JsonObject:
    """Validate a complete source-probe artifact through every child hash."""

    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError("SECURITY_LIFECYCLE_SOURCE_PROBE_MANIFEST_INVALID") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != PROBE_SCHEMA_VERSION
        or manifest.get("artifact_name") != ARTIFACT_NAME
        or manifest.get("probe_id") != path.name
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_SOURCE_PROBE_MANIFEST_INVALID")
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict) or set(hashes) != REQUIRED_ARTIFACTS:
        raise DataValidationError("SECURITY_LIFECYCLE_SOURCE_PROBE_ARTIFACT_SET_INVALID")
    for name, expected in hashes.items():
        if not isinstance(expected, str) or file_sha256(path / name) != expected:
            raise DataValidationError(f"SECURITY_LIFECYCLE_SOURCE_PROBE_HASH_MISMATCH: {name}")
    return cast(JsonObject, manifest)


def _compare_intervals(
    targets: DataFrame,
    requests: list[JsonObject],
    provider: dict[str, DataFrame],
    local: dict[str, DataFrame],
) -> DataFrame:
    rows: list[JsonObject] = []
    failures = {
        (str(item["canonical_ts_code"]), str(item["endpoint"]))
        for item in requests
        if item["status"] != "SUCCESS"
    }
    for item in targets.sort_values(["canonical_ts_code", "gap_start", "gap_end"]).itertuples(
        index=False
    ):
        code, start, end = str(item.canonical_ts_code), str(item.gap_start), str(item.gap_end)
        provider_suspend = _interval_rows(provider["suspend_d"], code, start, end)
        local_suspend = _interval_rows(local["suspend_d"], code, start, end)
        provider_daily = _valid_daily_rows(_interval_rows(provider["daily"], code, start, end))
        local_daily = _valid_daily_rows(_interval_rows(local["daily"], code, start, end))
        provider_s = _full_day_s_rows(provider_suspend)
        local_s = _full_day_s_rows(local_suspend)
        missing_suspend = _row_difference(provider_s, local_s, "suspend_d")
        missing_daily = _row_difference(provider_daily, local_daily, "daily")
        local_extra_suspend = _row_difference(local_s, provider_s, "suspend_d")
        local_extra_daily = _row_difference(local_daily, provider_daily, "daily")
        failed = (code, "suspend_d") in failures or (code, "daily") in failures
        status = _source_status(
            failed=failed,
            missing_suspend=missing_suspend,
            missing_daily=missing_daily,
            local_extra_suspend=local_extra_suspend,
            local_extra_daily=local_extra_daily,
            provider_s=provider_s,
            provider_daily=provider_daily,
        )
        rows.append(
            {
                "parent_interval_id": str(item.parent_interval_id),
                "canonical_ts_code": code,
                "gap_start": start,
                "gap_end": end,
                "session_count": int(str(item.session_count)),
                "triage_category": str(item.triage_category),
                "source_completeness_status": status,
                "repair_required": status.startswith("LOCAL_"),
                "probe_failed": failed,
                "provider_history_changed": bool(
                    missing_suspend or missing_daily or local_extra_suspend or local_extra_daily
                ),
                "provider_suspend_rows": int(len(provider_suspend)),
                "provider_full_day_s_rows": int(len(provider_s)),
                "local_suspend_rows": int(len(local_suspend)),
                "local_full_day_s_rows": int(len(local_s)),
                "provider_valid_daily_rows": int(len(provider_daily)),
                "local_valid_daily_rows": int(len(local_daily)),
                "provider_has_no_suspend_evidence": provider_s.empty,
                "provider_has_no_daily_evidence": provider_daily.empty,
                "missing_suspend_rows": json.dumps(missing_suspend, sort_keys=True),
                "missing_daily_rows": json.dumps(missing_daily, sort_keys=True),
                "provider_suspend_hash": _frame_hash(provider_suspend),
                "provider_daily_hash": _frame_hash(provider_daily),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != len(targets) or result["parent_interval_id"].duplicated().any():
        raise DataValidationError("SECURITY_LIFECYCLE_SOURCE_PROBE_RECONCILIATION_FAILED")
    return result


def _source_status(
    *,
    failed: bool,
    missing_suspend: list[JsonObject],
    missing_daily: list[JsonObject],
    local_extra_suspend: list[JsonObject],
    local_extra_daily: list[JsonObject],
    provider_s: DataFrame,
    provider_daily: DataFrame,
) -> SourceCompletenessStatus:
    if failed:
        return "PROBE_INCONCLUSIVE"
    if missing_suspend and missing_daily:
        return "LOCAL_BOTH_INCOMPLETE"
    if missing_suspend:
        return "LOCAL_SUSPEND_D_INCOMPLETE"
    if missing_daily:
        return "LOCAL_DAILY_INCOMPLETE"
    if local_extra_suspend or local_extra_daily:
        return "PROVIDER_CONTRADICTS_LOCAL"
    if provider_s.empty:
        return "PROVIDER_HAS_NO_SUSPEND_EVIDENCE"
    if provider_daily.empty and provider_s.empty:
        return "PROVIDER_HAS_NO_DAILY_EVIDENCE"
    return "PROVIDER_AND_LOCAL_AGREE"


def _build_windows(
    targets: DataFrame,
    calendar: tuple[str, ...],
    identity: SecurityIdentityResolver,
    buffer_sessions: int,
) -> tuple[ProbeWindow, ...]:
    positions = {date: index for index, date in enumerate(calendar)}
    raw: list[ProbeWindow] = []
    for row in targets.itertuples(index=False):
        start, end = str(row.gap_start), str(row.gap_end)
        if start not in positions or end not in positions:
            raise DataValidationError("SECURITY_LIFECYCLE_SOURCE_PROBE_CALENDAR_GAP")
        buffered_start = calendar[max(0, positions[start] - buffer_sessions)]
        buffered_end = calendar[min(len(calendar) - 1, positions[end] + buffer_sessions)]
        code = str(row.canonical_ts_code)
        for source_code in identity.source_codes_for({code}):
            raw.append(ProbeWindow(code, source_code, buffered_start, buffered_end))
    merged: list[ProbeWindow] = []
    rows = [
        {
            "canonical_ts_code": item.canonical_ts_code,
            "source_ts_code": item.source_ts_code,
            "start_date": item.start_date,
            "end_date": item.end_date,
        }
        for item in raw
    ]
    for key, group in pd.DataFrame(rows).groupby(
        ["canonical_ts_code", "source_ts_code"], sort=True
    ):
        current_start: str | None = None
        current_end: str | None = None
        for row in group.sort_values(["start_date", "end_date"]).itertuples(index=False):
            start, end = str(row.start_date), str(row.end_date)
            if current_start is None:
                current_start, current_end = start, end
            elif positions[start] <= positions[cast(str, current_end)] + 1:
                current_end = max(cast(str, current_end), end)
            else:
                merged.append(
                    ProbeWindow(str(key[0]), str(key[1]), current_start, cast(str, current_end))
                )
                current_start, current_end = start, end
        if current_start is not None and current_end is not None:
            merged.append(ProbeWindow(str(key[0]), str(key[1]), current_start, current_end))
    return tuple(merged)


def _normalize_provider_frame(
    frame: DataFrame,
    *,
    endpoint: str,
    request_id: str,
    identity: SecurityIdentityResolver,
) -> DataFrame:
    columns = (
        [
            "request_id",
            "source_ts_code",
            "canonical_ts_code",
            "trade_date",
            "suspend_timing",
            "suspend_type",
        ]
        if endpoint == "suspend_d"
        else [
            "request_id",
            "source_ts_code",
            "canonical_ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "amount",
        ]
    )
    if frame.empty:
        return pd.DataFrame(columns=columns)
    required = {"ts_code", "trade_date"}
    if not required.issubset(frame.columns):
        raise DataValidationError(f"SECURITY_LIFECYCLE_SOURCE_PROBE_SCHEMA_INVALID: {endpoint}")
    working = frame.copy()
    working["request_id"] = request_id
    working["source_ts_code"] = working["ts_code"].astype(str).str.strip().str.upper()
    working["trade_date"] = working["trade_date"].astype(str)
    working["canonical_ts_code"] = [
        identity.canonicalize(code, as_of_date=date)
        for code, date in zip(working["source_ts_code"], working["trade_date"], strict=True)
    ]
    for column in columns:
        if column not in working:
            working[column] = pd.NA
    return (
        working[columns]
        .sort_values(["canonical_ts_code", "trade_date", "source_ts_code"])
        .reset_index(drop=True)
    )


def _probe_counts(comparison: DataFrame, requests: list[JsonObject]) -> JsonObject:
    statuses = comparison["source_completeness_status"].astype(str)
    names = (
        "LOCAL_SUSPEND_D_INCOMPLETE",
        "LOCAL_DAILY_INCOMPLETE",
        "LOCAL_BOTH_INCOMPLETE",
        "PROVIDER_HAS_NO_SUSPEND_EVIDENCE",
        "PROVIDER_HAS_NO_DAILY_EVIDENCE",
        "PROVIDER_AND_LOCAL_AGREE",
        "PROVIDER_CONTRADICTS_LOCAL",
        "PROBE_INCONCLUSIVE",
    )
    by_category: JsonObject = {}
    for category, group in comparison.groupby("triage_category", sort=True):
        by_category[str(category)] = {
            name: int(group["source_completeness_status"].astype(str).eq(name).sum())
            for name in names
        }
        by_category[str(category)]["intervals"] = int(len(group))
    return {
        "intervals_probed": int(len(comparison)),
        "securities_probed": int(comparison["canonical_ts_code"].nunique()),
        "api_requests": len(requests),
        "request_failures": sum(item["status"] != "SUCCESS" for item in requests),
        **{name: int(statuses.eq(name).sum()) for name in names},
        "provider_has_no_daily_evidence": int(
            comparison["provider_has_no_daily_evidence"].fillna(False).astype(bool).sum()
        ),
        "by_triage_category": by_category,
    }


def _request_identity(item: JsonObject) -> JsonObject:
    return {
        key: item.get(key)
        for key in (
            "provider",
            "endpoint",
            "canonical_ts_code",
            "params",
            "request_id",
            "client_version",
            "status",
            "error_class",
            "raw_provider_response_hash",
            "normalized_response_hash",
            "row_count",
            "minimum_date",
            "maximum_date",
        )
    }


def _interval_id(row: pd.Series) -> str:
    identity = {
        "canonical_ts_code": str(row["canonical_ts_code"]),
        "gap_start": str(row["gap_start"]),
        "gap_end": str(row["gap_end"]),
        "session_count": int(str(row["session_count"])),
    }
    return f"interval_{canonical_payload_hash(identity)[:24]}"


def _validate_unique_intervals(frame: DataFrame) -> None:
    if frame["parent_interval_id"].duplicated().any():
        raise DataValidationError("SECURITY_LIFECYCLE_SOURCE_PROBE_DUPLICATE_INTERVAL")


def _raw_frame_hash(frame: DataFrame) -> str:
    payload = {
        "columns": list(frame.columns),
        "rows": frame.astype(object).where(pd.notna(frame), None).to_dict("records"),
    }
    return canonical_payload_hash(payload)


def _frame_hash(frame: DataFrame) -> str:
    if frame.empty:
        return canonical_payload_hash({"columns": list(frame.columns), "rows": []})
    columns = sorted(frame.columns)
    working = frame[columns].astype(object).where(pd.notna(frame[columns]), None)
    rows = working.sort_values(columns, key=lambda values: values.astype(str)).to_dict("records")
    return canonical_payload_hash({"columns": columns, "rows": rows})


def _date_bound(frame: DataFrame, operation: str) -> str | None:
    if frame.empty or "trade_date" not in frame:
        return None
    values = frame["trade_date"].astype(str)
    return str(values.min() if operation == "min" else values.max())


def _concat_frames(frames: list[DataFrame], endpoint: str) -> DataFrame:
    if frames:
        return pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    return _normalize_provider_frame(
        pd.DataFrame(),
        endpoint=endpoint,
        request_id="EMPTY",
        identity=SecurityIdentityResolver.empty(),
    )


def _query_local(con: duckdb.DuckDBPyConnection, root: Path, columns: str) -> DataFrame:
    if not any(root.glob("**/*.parquet")):
        raise DataValidationError(f"SECURITY_LIFECYCLE_SOURCE_PROBE_LOCAL_MISSING: {root.name}")
    frame: DataFrame = con.execute(
        f"""SELECT {columns} FROM read_parquet('{_parquet_glob(root)}',
          hive_partitioning=false,union_by_name=true) d
        JOIN selected_codes s ON cast(d.ts_code AS VARCHAR)=s.source_ts_code"""
    ).fetchdf()
    return frame


def _interval_rows(frame: DataFrame, code: str, start: str, end: str) -> DataFrame:
    if frame.empty:
        return frame.copy()
    dates = frame["trade_date"].astype(str)
    return frame[frame["canonical_ts_code"].astype(str).eq(code) & dates.between(start, end)].copy()


def _full_day_s_rows(frame: DataFrame) -> DataFrame:
    if frame.empty:
        return frame.copy()
    return frame[
        frame["suspend_type"].astype("string").fillna("").str.upper().eq("S")
        & frame["suspend_timing"].astype("string").fillna("").str.strip().eq("")
    ].copy()


def _valid_daily_rows(frame: DataFrame) -> DataFrame:
    if frame.empty:
        return frame.copy()
    prices = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    valid: DataFrame = frame[prices.notna().all(axis=1) & (prices > 0).all(axis=1)].copy()
    return valid


def _row_difference(left: DataFrame, right: DataFrame, endpoint: str) -> list[JsonObject]:
    columns = (
        ["source_ts_code", "canonical_ts_code", "trade_date", "suspend_timing", "suspend_type"]
        if endpoint == "suspend_d"
        else ["source_ts_code", "canonical_ts_code", "trade_date", "open", "high", "low", "close"]
    )
    left_rows = _record_map(left, columns)
    right_rows = _record_map(right, columns)
    return [left_rows[key] for key in sorted(set(left_rows) - set(right_rows))]


def _record_map(frame: DataFrame, columns: list[str]) -> dict[str, JsonObject]:
    if frame.empty:
        return {}
    records = frame[columns].astype(object).where(pd.notna(frame[columns]), None).to_dict("records")
    return {canonical_payload_hash(record): cast(JsonObject, record) for record in records}


def _client_version() -> str:
    try:
        return importlib.metadata.version("tushare")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _parquet_glob(root: Path) -> str:
    return str(root / "**" / "*.parquet").replace("'", "''")
