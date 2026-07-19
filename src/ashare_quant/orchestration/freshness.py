"""Session-aware, read-only production freshness and inference-readiness gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from ashare_quant.config.settings import AppSettings, ProductionFreshnessSettings
from ashare_quant.data.datasets import get_dataset_spec
from ashare_quant.data.ingestion import DataIngestionService
from ashare_quant.data.storage import ParquetDataStore
from ashare_quant.features.storage import FeatureStore
from ashare_quant.universe.storage import UniverseStore
from ashare_quant.utils.manifest import (
    artifact_manifest_status,
    current_git_info,
    parquet_artifact_statistics,
    read_manifest,
)

SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")
MARKET_CLOSE_TIME = time(15, 0)

type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class GateResult:
    """Structured outcome from one read-only production gate."""

    gate: str
    expected_as_of: str
    hard_failures: tuple[str, ...]
    warnings: tuple[str, ...]
    details: dict[str, Any]

    @property
    def ready(self) -> bool:
        """Return whether this gate permits downstream execution."""

        return not self.hard_failures

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable stage result."""

        return {
            "gate": self.gate,
            "ready": self.ready,
            "expected_as_of": self.expected_as_of,
            "hard_failures": list(self.hard_failures),
            "warnings": list(self.warnings),
            **self.details,
        }


class FreshnessService:
    """Evaluate raw and processed artifacts without modifying them."""

    def __init__(
        self,
        settings: AppSettings,
        raw_store: ParquetDataStore,
        universe_store: UniverseStore,
        feature_store: FeatureStore,
        *,
        config_path: Path,
        now: datetime | None = None,
    ) -> None:
        self.settings = settings
        self.raw_store = raw_store
        self.universe_store = universe_store
        self.feature_store = feature_store
        self.config_path = config_path
        self.now = now

    def check_session(self, as_of: str) -> GateResult:
        """Require an explicit open session that has completed in Shanghai time."""

        failures: list[str] = []
        warnings: list[str] = []
        calendar = self.raw_store.read_dataset(get_dataset_spec("trade_cal"))
        local_now = (self.now or datetime.now(UTC)).astimezone(SHANGHAI_TIMEZONE)
        completed_cutoff = _completed_cutoff(local_now)
        exists = False
        is_open = False
        if calendar.empty or not {"cal_date", "is_open"}.issubset(calendar.columns):
            failures.append("trade_cal with cal_date and is_open is required")
        else:
            dates = calendar["cal_date"].astype(str)
            matching = calendar.loc[dates.eq(as_of)]
            exists = not matching.empty
            is_open = exists and bool(matching["is_open"].astype(int).eq(1).any())
            if not exists:
                failures.append(f"as-of date is absent from trade_cal: {as_of}")
            elif not is_open:
                failures.append(f"as-of date is not an open A-share session: {as_of}")
        if not _is_yyyymmdd(as_of):
            failures.append(f"invalid YYYYMMDD as-of date: {as_of}")
        elif as_of > completed_cutoff:
            failures.append(f"as-of session is future or incomplete: {as_of}")
        self._apply_git_policy(failures, warnings)
        return GateResult(
            "session_gate",
            as_of,
            tuple(failures),
            tuple(warnings),
            {
                "session": {
                    "exists": exists,
                    "is_open": is_open,
                    "completed_cutoff": completed_cutoff,
                    "timezone": "Asia/Shanghai",
                },
                "thresholds": {"market_close_time": "15:00:00"},
            },
        )

    def check_raw(self, as_of: str) -> GateResult:
        """Require hard raw datasets and explicitly classify soft/empty datasets."""

        policy = self.settings.production.freshness
        failures: list[str] = []
        warnings: list[str] = []
        session = self.check_session(as_of)
        failures.extend(session.hard_failures)
        warnings.extend(session.warnings)
        actual_max_dates: dict[str, str | None] = {}
        row_counts: dict[str, int] = {}
        gap_details: dict[str, Any] = {}
        ingestion = DataIngestionService(self.settings, store=self.raw_store)

        for name in policy.hard_datasets:
            spec = get_dataset_spec(name)
            statistics = parquet_artifact_statistics(
                self.raw_store.dataset_dir(spec), date_column=spec.date_column or "trade_date"
            )
            actual_max_dates[name] = statistics.max_date
            frame = self.raw_store.read_dataset(spec, as_of, as_of)
            row_counts[name] = len(frame)
            if statistics.partition_count == 0:
                failures.append(f"required raw dataset is missing: {name}")
            if frame.empty:
                failures.append(f"required raw dataset lacks as-of rows: {name} {as_of}")
            report = ingestion.scan_gaps((name,), as_of, as_of)[0]
            gap_details[name] = {
                "missing_dates": list(report.missing_dates),
                "missing_by_entity": {
                    entity: list(dates) for entity, dates in report.missing_by_entity.items()
                },
            }
            if report.has_gaps:
                failures.append(f"required raw dataset has unresolved as-of gap: {name} {as_of}")

        required_indices = self._required_index_codes(policy)
        index_frame = self.raw_store.read_dataset(get_dataset_spec("index_daily"), as_of, as_of)
        available_indices = set(index_frame.get("ts_code", pd.Series(dtype=str)).astype(str))
        missing_indices = sorted(set(required_indices) - available_indices)
        if missing_indices:
            failures.append(f"benchmark index entities missing on {as_of}: {missing_indices}")

        legitimate_empty: dict[str, str] = {}
        for name in policy.legitimate_empty_datasets:
            spec = get_dataset_spec(name)
            frame = self.raw_store.read_dataset(spec, as_of, as_of)
            if not frame.empty:
                legitimate_empty[name] = "rows_present"
            elif self.raw_store.has_empty_result(spec, as_of):
                legitimate_empty[name] = "explicit_empty_marker"
            else:
                legitimate_empty[name] = "unresolved_empty"
                failures.append(
                    f"conditional dataset is empty without an explicit completion marker: "
                    f"{name} {as_of}"
                )

        soft_lags: dict[str, Any] = {}
        for name, max_lag in policy.soft_dataset_max_lag_calendar_days.items():
            spec = get_dataset_spec(name)
            statistics = parquet_artifact_statistics(
                self.raw_store.dataset_dir(spec), date_column=spec.date_column or "trade_date"
            )
            lag = _calendar_lag(as_of, statistics.max_date)
            soft_lags[name] = {"max_date": statistics.max_date, "lag_days": lag, "limit": max_lag}
            if statistics.partition_count == 0:
                warnings.append(f"optional low-frequency dataset is missing: {name}")
            elif lag is not None and lag > max_lag:
                warnings.append(f"optional low-frequency dataset is stale: {name} lag_days={lag}")

        for name in policy.event_datasets:
            spec = get_dataset_spec(name)
            exists = bool(list(self.raw_store.dataset_dir(spec).glob("**/*.parquet")))
            if not exists:
                warnings.append(f"optional event dataset is missing: {name}")

        snapshot_ages: dict[str, Any] = {}
        for name, max_age in policy.snapshot_max_age_days.items():
            status = self.raw_store.status(get_dataset_spec(name))
            snapshot_ages[name] = {"age_days": status.snapshot_age_days, "limit": max_age}
            if not status.exists:
                warnings.append(f"optional snapshot dataset is missing: {name}")
            elif status.snapshot_age_days is None or status.snapshot_age_days > max_age:
                warnings.append(f"optional snapshot dataset is stale: {name}")

        baseline_dates = self._baseline_dates(as_of)
        daily_counts = self._raw_counts("daily", baseline_dates)
        daily_drift = _evaluate_count(
            "daily_rows",
            row_counts.get("daily", 0),
            daily_counts,
            policy.minimum_daily_rows,
            policy,
            failures,
            warnings,
        )
        return GateResult(
            "raw_freshness_gate",
            as_of,
            tuple(_deduplicate(failures)),
            tuple(_deduplicate(warnings)),
            {
                "actual_max_dates": actual_max_dates,
                "required_entities": {"index_daily": list(required_indices)},
                "row_counts": row_counts,
                "gap_details": gap_details,
                "legitimate_empty_sessions": legitimate_empty,
                "soft_dataset_lags": soft_lags,
                "snapshot_ages": snapshot_ages,
                "recent_baseline_statistics": {"daily_rows": daily_drift},
                "thresholds": _threshold_dict(policy),
            },
        )

    def check_universe(self, as_of: str) -> GateResult:
        """Validate one universe session, its drift, and canonical manifest identity."""

        policy = self.settings.production.freshness
        failures: list[str] = []
        warnings: list[str] = []
        frame = self.universe_store.read(as_of, as_of)
        duplicate_rows = _duplicate_count(frame, ("trade_date", "ts_code"))
        if frame.empty:
            failures.append(f"universe_daily lacks as-of partition rows: {as_of}")
        if duplicate_rows:
            failures.append(f"universe_daily has duplicate keys on {as_of}: {duplicate_rows}")
        counts = {
            "rows": len(frame),
            "in_base_universe": _boolean_count(frame, "in_base_universe"),
            "in_model_universe": _boolean_count(frame, "in_model_universe"),
        }
        manifest_details = self._check_artifact_manifest(
            "universe_daily", self.universe_store.dataset_dir, as_of, failures
        )
        baseline_dates = self._baseline_dates(as_of)
        baseline = self.universe_store.read(
            baseline_dates[0] if baseline_dates else as_of,
            baseline_dates[-1] if baseline_dates else as_of,
        )
        baseline_counts = _grouped_universe_counts(baseline)
        drift = {
            "rows": _evaluate_count(
                "universe_rows",
                counts["rows"],
                baseline_counts["rows"],
                policy.minimum_universe_rows,
                policy,
                failures,
                warnings,
            ),
            "in_model_universe": _evaluate_count(
                "in_model_universe",
                counts["in_model_universe"],
                baseline_counts["in_model_universe"],
                policy.minimum_model_universe_rows,
                policy,
                failures,
                warnings,
            ),
        }
        if counts["in_base_universe"] < policy.minimum_base_universe_rows:
            failures.append(
                "in_base_universe count below absolute minimum: "
                f"{counts['in_base_universe']} < {policy.minimum_base_universe_rows}"
            )
        return GateResult(
            "universe_readiness_gate",
            as_of,
            tuple(_deduplicate(failures)),
            tuple(_deduplicate(warnings)),
            {
                "row_counts": counts,
                "duplicate_rows": duplicate_rows,
                "artifact_manifest": manifest_details,
                "recent_baseline_statistics": drift,
                "thresholds": _threshold_dict(policy),
            },
        )

    def check_features(self, as_of: str) -> GateResult:
        """Validate feature schema, universe alignment, missingness, and coverage drift."""

        policy = self.settings.production.freshness
        failures: list[str] = []
        warnings: list[str] = []
        features = self.feature_store.read(as_of, as_of)
        universe = self.universe_store.read(as_of, as_of)
        duplicate_rows = _duplicate_count(features, ("trade_date", "ts_code"))
        if features.empty:
            failures.append(f"features_daily lacks as-of partition rows: {as_of}")
        if duplicate_rows:
            failures.append(f"features_daily has duplicate keys on {as_of}: {duplicate_rows}")
        if len(features) != len(universe):
            failures.append(
                f"features/universe row mismatch on {as_of}: {len(features)} != {len(universe)}"
            )
        try:
            required_features = self._required_features(policy)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            failures.append(f"invalid required production feature list: {error}")
            required_features = policy.hard_required_features
        missing_columns = sorted(set(required_features) - set(features.columns))
        if missing_columns:
            failures.append(f"hard-required feature columns are missing: {missing_columns}")
        manifest_details = self._check_artifact_manifest(
            "features_daily", self.feature_store.dataset_dir, as_of, failures
        )
        physical_features = sorted(set(features.columns) - {"trade_date", "ts_code"})
        manifest_feature_count = manifest_details.get("feature_count")
        if manifest_feature_count is not None and manifest_feature_count != len(physical_features):
            failures.append(
                "features manifest/schema feature count mismatch: "
                f"{manifest_feature_count} != {len(physical_features)}"
            )

        eligible = _eligible_feature_rows(features, universe)
        missingness: dict[str, float] = {}
        for name in sorted(
            set(required_features)
            | set(policy.warning_features)
            | set(policy.structurally_sparse_features)
        ):
            if name not in eligible.columns:
                if name in policy.warning_features:
                    warnings.append(f"warning feature column is absent: {name}")
                continue
            ratio = float(eligible[name].isna().mean()) if len(eligible) else 1.0
            missingness[name] = ratio
            if name in required_features and ratio > policy.hard_feature_missing_ratio:
                failures.append(f"hard-required feature missingness breach: {name}={ratio:.4f}")
            elif name in policy.warning_features and ratio > policy.warning_feature_missing_ratio:
                warnings.append(f"warning feature missingness breach: {name}={ratio:.4f}")

        eligible_after_hard = eligible
        available_required = [name for name in required_features if name in eligible.columns]
        if available_required:
            eligible_after_hard = eligible.dropna(subset=available_required)
        baseline_dates = self._baseline_dates(as_of)
        baseline_features = self.feature_store.read(
            baseline_dates[0] if baseline_dates else as_of,
            baseline_dates[-1] if baseline_dates else as_of,
        )
        baseline_universe = self.universe_store.read(
            baseline_dates[0] if baseline_dates else as_of,
            baseline_dates[-1] if baseline_dates else as_of,
        )
        baseline_eligible = _eligible_feature_rows(baseline_features, baseline_universe)
        if available_required and not baseline_eligible.empty:
            baseline_eligible = baseline_eligible.dropna(subset=available_required)
        history_counts = (
            baseline_eligible.groupby("trade_date").size().astype(int).tolist()
            if not baseline_eligible.empty
            else []
        )
        eligible_drift = _evaluate_count(
            "feature_eligible_rows",
            len(eligible_after_hard),
            history_counts,
            policy.minimum_model_universe_rows,
            policy,
            failures,
            warnings,
        )
        return GateResult(
            "features_readiness_gate",
            as_of,
            tuple(_deduplicate(failures)),
            tuple(_deduplicate(warnings)),
            {
                "row_counts": {
                    "features": len(features),
                    "universe": len(universe),
                    "eligible_universe": len(eligible),
                    "eligible_after_hard_features": len(eligible_after_hard),
                },
                "duplicate_rows": duplicate_rows,
                "required_features": list(required_features),
                "physical_feature_count": len(physical_features),
                "missingness_summary": missingness,
                "artifact_manifest": manifest_details,
                "recent_baseline_statistics": {"feature_eligible_rows": eligible_drift},
                "thresholds": _threshold_dict(policy),
            },
        )

    def check_all(self, as_of: str) -> tuple[GateResult, ...]:
        """Run all read-only gates in production order."""

        return (
            self.check_raw(as_of),
            self.check_universe(as_of),
            self.check_features(as_of),
        )

    def _required_index_codes(self, policy: ProductionFreshnessSettings) -> tuple[str, ...]:
        if policy.required_index_codes:
            return tuple(dict.fromkeys(policy.required_index_codes))
        return tuple(
            dict.fromkeys(
                (
                    self.settings.labels.benchmark_index_code,
                    self.settings.features.benchmark_index_code,
                    self.settings.backtest.benchmark_index_code,
                )
            )
        )

    def _required_features(self, policy: ProductionFreshnessSettings) -> tuple[str, ...]:
        features = list(policy.hard_required_features)
        if policy.required_feature_list_path is not None:
            payload = json.loads(policy.required_feature_list_path.read_text(encoding="utf-8"))
            configured = payload.get("features") if isinstance(payload, dict) else None
            if not isinstance(configured, list) or not all(
                isinstance(item, str) for item in configured
            ):
                raise ValueError(
                    "production freshness required feature list must contain `features` strings: "
                    f"{policy.required_feature_list_path}"
                )
            features.extend(configured)
        return tuple(dict.fromkeys(features))

    def _baseline_dates(self, as_of: str) -> tuple[str, ...]:
        calendar = self.raw_store.read_dataset(get_dataset_spec("trade_cal"))
        if calendar.empty:
            return ()
        open_dates = sorted(
            calendar.loc[calendar["is_open"].astype(int).eq(1), "cal_date"]
            .dropna()
            .astype(str)
            .unique()
        )
        prior = [date for date in open_dates if date < as_of]
        return tuple(prior[-self.settings.production.freshness.baseline_sessions :])

    def _raw_counts(self, dataset: str, dates: tuple[str, ...]) -> list[int]:
        if not dates:
            return []
        spec = get_dataset_spec(dataset)
        frame = self.raw_store.read_dataset(spec, dates[0], dates[-1])
        if frame.empty or spec.date_column is None:
            return []
        return frame.groupby(spec.date_column).size().astype(int).tolist()

    def _check_artifact_manifest(
        self,
        artifact_name: str,
        artifact_dir: Path,
        as_of: str,
        failures: list[str],
    ) -> dict[str, Any]:
        manifest = read_manifest(artifact_dir)
        status = artifact_manifest_status(artifact_dir, config_path=self.config_path)
        if manifest is None:
            failures.append(f"{artifact_name} manifest is missing")
            return {"exists": False, "current": False, "max_date": None}
        canonical = manifest.get("canonical_artifact")
        canonical_mapping = canonical if isinstance(canonical, dict) else {}
        max_date = canonical_mapping.get("max_date")
        if not isinstance(max_date, str) or max_date < as_of:
            failures.append(f"{artifact_name} canonical manifest is stale for {as_of}")
        if status.config_hash_match is not True:
            failures.append(f"{artifact_name} manifest config hash does not match current config")
        if status.artifact_git_revision != status.current_git_revision:
            failures.append(f"{artifact_name} manifest Git revision is not current")
        return {
            "exists": True,
            "current": not status.stale,
            "max_date": max_date,
            "config_hash_match": status.config_hash_match,
            "artifact_git_revision": status.artifact_git_revision,
            "current_git_revision": status.current_git_revision,
            "feature_count": canonical_mapping.get("feature_count", manifest.get("feature_count")),
        }

    def _apply_git_policy(self, failures: list[str], warnings: list[str]) -> None:
        if not current_git_info()["dirty"]:
            return
        policy = self.settings.production.freshness.git_dirty_policy
        if policy == "fail":
            failures.append("repository working tree is dirty")
        elif policy == "warning":
            warnings.append("repository working tree is dirty")


def _completed_cutoff(local_now: datetime) -> str:
    today = local_now.strftime("%Y%m%d")
    if local_now.time() >= MARKET_CLOSE_TIME:
        return today
    return (local_now - timedelta(days=1)).strftime("%Y%m%d")


def _is_yyyymmdd(value: str) -> bool:
    try:
        return datetime.strptime(value, "%Y%m%d").strftime("%Y%m%d") == value
    except ValueError:
        return False


def _calendar_lag(as_of: str, actual: str | None) -> int | None:
    if actual is None or not _is_yyyymmdd(actual):
        return None
    return max(
        0,
        (datetime.strptime(as_of, "%Y%m%d") - datetime.strptime(actual, "%Y%m%d")).days,
    )


def _duplicate_count(frame: DataFrame, keys: tuple[str, ...]) -> int:
    if frame.empty or not set(keys).issubset(frame.columns):
        return 0
    return int(frame.duplicated(subset=list(keys)).sum())


def _boolean_count(frame: DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def _grouped_universe_counts(frame: DataFrame) -> dict[str, list[int]]:
    if frame.empty:
        return {"rows": [], "in_model_universe": []}
    grouped = frame.groupby("trade_date", sort=True)
    return {
        "rows": grouped.size().astype(int).tolist(),
        "in_model_universe": grouped["in_model_universe"].sum().astype(int).tolist(),
    }


def _eligible_feature_rows(features: DataFrame, universe: DataFrame) -> DataFrame:
    if features.empty or universe.empty or "in_model_universe" not in universe.columns:
        return features.iloc[0:0].copy()
    eligible_keys = universe.loc[
        universe["in_model_universe"].fillna(False).astype(bool), ["trade_date", "ts_code"]
    ]
    return features.merge(eligible_keys, on=["trade_date", "ts_code"], how="inner")


def _evaluate_count(
    name: str,
    current: int,
    history: list[int],
    absolute_minimum: int,
    policy: ProductionFreshnessSettings,
    failures: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    if current < absolute_minimum:
        failures.append(f"{name} below absolute minimum: {current} < {absolute_minimum}")
    if len(history) < policy.minimum_baseline_sessions:
        warnings.append(
            f"{name} has insufficient baseline history: "
            f"{len(history)} < {policy.minimum_baseline_sessions}"
        )
        return {"sessions": len(history), "median": None, "ratio": None}
    baseline = float(median(history))
    ratio = current / baseline if baseline > 0 else None
    if ratio is None:
        warnings.append(f"{name} baseline median is zero")
    elif ratio < policy.severe_count_ratio_low or ratio > policy.severe_count_ratio_high:
        failures.append(f"{name} severe baseline deviation: ratio={ratio:.4f}")
    elif ratio < policy.moderate_count_ratio_low or ratio > policy.moderate_count_ratio_high:
        warnings.append(f"{name} moderate baseline deviation: ratio={ratio:.4f}")
    return {"sessions": len(history), "median": baseline, "ratio": ratio}


def _threshold_dict(policy: ProductionFreshnessSettings) -> dict[str, Any]:
    return {
        "baseline_sessions": policy.baseline_sessions,
        "minimum_baseline_sessions": policy.minimum_baseline_sessions,
        "moderate_count_ratio": [
            policy.moderate_count_ratio_low,
            policy.moderate_count_ratio_high,
        ],
        "severe_count_ratio": [
            policy.severe_count_ratio_low,
            policy.severe_count_ratio_high,
        ],
        "minimum_daily_rows": policy.minimum_daily_rows,
        "minimum_universe_rows": policy.minimum_universe_rows,
        "minimum_base_universe_rows": policy.minimum_base_universe_rows,
        "minimum_model_universe_rows": policy.minimum_model_universe_rows,
        "hard_feature_missing_ratio": policy.hard_feature_missing_ratio,
        "warning_feature_missing_ratio": policy.warning_feature_missing_ratio,
    }


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
