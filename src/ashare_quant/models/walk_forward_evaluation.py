"""Immutable multi-fold Ranker evaluation and read-only recovery inspection."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast

import duckdb
import numpy as np
import pandas as pd

from ashare_quant.backtest.costs import ExecutionCostPolicy
from ashare_quant.backtest.data import load_benchmark, load_calendar, load_execution_prices
from ashare_quant.backtest.engine import BacktestInputs, simulate_portfolio
from ashare_quant.backtest.executable_validation import REQUIRED_TOP_N, _signals
from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_identity import SecurityIdentityResolver
from ashare_quant.data.security_identity_transition import (
    SecurityIdentityTransitionResolver,
    load_identity_transition_contract,
)
from ashare_quant.data.security_lifecycle import SecurityLifecycleResolver
from ashare_quant.data.security_lifecycle_audit import (
    LifecycleAuditPolicy,
    validate_pass_lifecycle_scan,
)
from ashare_quant.data.security_listing_metadata import SecurityListingMetadataResolver
from ashare_quant.models.compute import lightgbm_build_identity, resolve_training_backend
from ashare_quant.models.feature_provenance import (
    FeatureSetProvenance,
    feature_provenance_hash,
    feature_provenance_locator,
    validate_governed_feature_set,
)
from ashare_quant.models.horizon_experiments import dataset_fingerprint
from ashare_quant.models.ranker import feature_importance, fit_ranker, ranker_semantic_parameters
from ashare_quant.models.ranker_data import (
    RankerDataLoader,
    RankerDataset,
    RankerPredictionDataset,
)
from ashare_quant.models.ranker_metrics import evaluate_ranker, portfolio_metric_name
from ashare_quant.models.research_policy import enforce_research_window, load_research_policy
from ashare_quant.models.temporal_isolation import required_temporal_gap_sessions
from ashare_quant.utils.manifest import atomic_write_json

SCHEMA_VERSION = 3
LEGACY_SCHEMA_VERSION = 2
EVALUATION_CONTRACT_VERSION = 7
PREVIOUS_EVALUATION_CONTRACT_VERSION = 6
PREVIOUS_POST_H5_EVALUATION_CONTRACT_VERSION = 5
LEGACY_EVALUATION_CONTRACT_VERSION = 4
OLDER_EVALUATION_CONTRACT_VERSION = 3
ACCOUNTING_SCHEMA_VERSION = 2
EXECUTION_TAIL_POLICY_VERSION = 2
EXECUTION_TAIL_LOAD_CHUNK_SESSIONS = 252
EXECUTION_TAIL_POLICY = "carry_to_source_or_lockbox_cutoff"
REQUIRED_FOLD_ARTIFACTS = frozenset(
    {
        "model.txt",
        "predictions.parquet",
        "validation_metrics.json",
        "ranking_metrics.json",
        "executable_metrics.json",
        "feature_importance.json",
    }
)
type JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class FoldExecutionResult:
    """Material produced by one fold executor before immutable publication."""

    predictions: pd.DataFrame
    validation_metrics: JsonObject
    ranking_metrics: JsonObject
    executable_metrics: JsonObject
    feature_importance: list[JsonObject]
    training_compute: JsonObject
    model_saver: Callable[[Path], None]


class FoldExecutor(Protocol):
    """Existing-training adapter used by the orchestration service."""

    def validate_sources(self, plan: JsonObject) -> JsonObject: ...

    def execution_contract(
        self,
        *,
        horizon: int,
        require_executable: bool,
        prospective_lockbox_start: str,
    ) -> JsonObject: ...

    def mature_information_end(self, signal_end: str, horizon: int) -> str: ...

    def execute(
        self,
        *,
        fold: JsonObject,
        horizon: int,
        features: tuple[str, ...],
        require_executable: bool,
        execution_contract: JsonObject,
    ) -> FoldExecutionResult: ...


@dataclass(frozen=True, slots=True)
class WalkForwardEvaluationResult:
    experiment_id: str
    status: str
    fold_count: int
    output_dir: Path


class RankerFoldExecutor:
    """Execute folds through the common Ranker and portfolio simulation primitives."""

    def __init__(
        self,
        *,
        raw_root: Path,
        processed_root: Path,
        settings: AppSettings,
    ) -> None:
        self.raw_root = raw_root
        self.processed_root = processed_root
        self.settings = settings
        self._runtime: Any | None = None
        self._identity_transitions: SecurityIdentityTransitionResolver | None = (
            load_identity_transition_contract(
                mode=settings.security_identity.identity_transition_mode,
                artifact_path=settings.security_identity.identity_transition_path,
            )
        )

    def bind_identity_transitions(self, resolver: SecurityIdentityTransitionResolver) -> None:
        """Bind one root-preflight-validated transition contract for every fold."""

        self._identity_transitions = resolver

    def validate_sources(self, plan: JsonObject) -> JsonObject:
        """Verify current processed manifests against the frozen experiment plan."""

        features_manifest = self.processed_root / "features_daily" / "_manifest.json"
        universe_manifest = self.processed_root / "universe_daily" / "_manifest.json"
        if _file_hash(features_manifest) != plan.get("features_manifest_hash"):
            raise DataValidationError("walk-forward features source identity changed")
        if _file_hash(universe_manifest) != plan.get("universe_hash"):
            raise DataValidationError("walk-forward universe source identity changed")
        labels_root = self.processed_root / "labels_forward"
        label_files = sorted(labels_root.glob("**/*.parquet"))
        if not label_files:
            raise DataValidationError("walk-forward labels source is missing")
        labels_fingerprint = dataset_fingerprint(label_files, labels_root)
        if labels_fingerprint != plan.get("labels_fingerprint"):
            raise DataValidationError("walk-forward labels source identity changed")
        return {
            "features_manifest_hash": _file_hash(features_manifest),
            "universe_manifest_hash": _file_hash(universe_manifest),
            "labels_fingerprint": labels_fingerprint,
        }

    def execution_contract(
        self,
        *,
        horizon: int,
        require_executable: bool,
        prospective_lockbox_start: str,
    ) -> JsonObject:
        """Resolve and freeze execution-only inputs before run identity is computed."""

        runtime = resolve_training_backend(self.settings.ranker.training_backend)
        self._runtime = runtime
        execution = self.settings.backtest.model_copy(
            update={
                "execution": "next_open",
                "holding_period_days": horizon,
                "top_n": REQUIRED_TOP_N,
            }
        )
        cost_policy = ExecutionCostPolicy.from_backtest_settings(execution)
        processed_data_end = self._processed_execution_data_end()
        lifecycle = SecurityLifecycleResolver.from_path(
            self.settings.security_identity.lifecycle_path
        )
        transitions = self._identity_transitions
        if require_executable and transitions is None:
            raise DataValidationError("SECURITY_IDENTITY_TRANSITION_CONTRACT_REQUIRED")
        return {
            "training_compute": runtime.identity_payload(),
            "lightgbm_version": runtime.lightgbm_version,
            "lightgbm_build_identity": lightgbm_build_identity(runtime.lightgbm_version),
            "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
            "accounting_schema_version": ACCOUNTING_SCHEMA_VERSION,
            "require_executable": require_executable,
            "execution_mode": execution.execution,
            "holding_period_days": execution.holding_period_days,
            "top_n": list(REQUIRED_TOP_N),
            "sell_delay_max_days": execution.sell_delay_max_days,
            "execution_tail": {
                "policy_version": EXECUTION_TAIL_POLICY_VERSION,
                "policy": EXECUTION_TAIL_POLICY,
                "sell_delay_alert_sessions": execution.sell_delay_max_days,
                "processed_data_end": processed_data_end,
                "prospective_lockbox_start_exclusive": prospective_lockbox_start,
                "unresolved_at_cutoff": "fail_closed",
                **lifecycle.provenance(),
            },
            "cost_policy_hash": cost_policy.policy_hash,
            "security_identity_transitions": (
                transitions.provenance() if transitions is not None else None
            ),
            "data_logic": {
                "prediction_universe": "signal_date_universe_left_join_features_v1",
                "metric_universe": "frozen_predictions_left_join_mature_labels_v1",
                "execution_universe": "complete_frozen_predictions_v1",
            },
        }

    def _processed_execution_data_end(self) -> str:
        manifest = _load_json(
            self.processed_root / "universe_daily" / "_manifest.json",
            "walk-forward universe manifest",
        )
        canonical = manifest.get("canonical_artifact")
        maximum = canonical.get("max_date") if isinstance(canonical, dict) else None
        if not isinstance(maximum, str) or len(maximum) != 8 or not maximum.isdigit():
            raise DataValidationError("walk-forward execution data cutoff is unavailable")
        return maximum

    def mature_information_end(self, signal_end: str, horizon: int) -> str:
        """Resolve the last forward-label session consumed by feature selection."""

        required = required_temporal_gap_sessions(horizon)
        calendar_glob = self.raw_root / "trade_cal" / "**" / "*.parquet"
        query = f"""
            SELECT CAST(cal_date AS VARCHAR) AS trade_date
            FROM read_parquet('{calendar_glob.as_posix()}', hive_partitioning=false)
            WHERE CAST(is_open AS INTEGER) = 1
              AND CAST(cal_date AS VARCHAR) > ?
            ORDER BY cal_date
            LIMIT ?
        """  # noqa: S608 -- local configured Parquet path
        with duckdb.connect() as connection:
            dates = connection.execute(query, [signal_end, required]).fetch_df()
        if len(dates) != required:
            raise DataValidationError(
                "FEATURE_SELECTION_INFORMATION_END_UNRESOLVED: "
                f"signal_end={signal_end} horizon={horizon}"
            )
        return str(dates.iloc[-1]["trade_date"])

    def execute(
        self,
        *,
        fold: JsonObject,
        horizon: int,
        features: tuple[str, ...],
        require_executable: bool,
        execution_contract: JsonObject,
    ) -> FoldExecutionResult:
        loader = RankerDataLoader(
            self.processed_root,
            horizon=horizon,
            minimum_group_size=self.settings.ranker.minimum_group_size,
        )
        train = loader.load(
            str(fold["train_start"]),
            str(fold["train_end"]),
            features,
            self.settings.ranker.relevance_grades,
        )
        validation = loader.load(
            str(fold["validation_start"]),
            str(fold["validation_end"]),
            features,
            self.settings.ranker.relevance_grades,
        )
        evaluation = loader.load_prediction_universe(
            str(fold["evaluation_start"]),
            str(fold["evaluation_end"]),
            features,
        )
        runtime = self._runtime or resolve_training_backend(self.settings.ranker.training_backend)
        model = fit_ranker(train, validation, self.settings.ranker, runtime)
        validation_predictions = np.asarray(model.predict(validation.features), dtype=float)
        evaluation_predictions = np.asarray(model.predict(evaluation.features), dtype=float)
        predictions = _build_prediction_frame(evaluation.frame, evaluation_predictions)
        label_cutoff = _governed_execution_cutoff(execution_contract["execution_tail"])
        label_calendar = load_calendar(
            self.raw_root,
            str(predictions["trade_date"].astype(str).min()),
            label_cutoff,
            None,
            maximum_date=label_cutoff,
        )
        labeled_evaluation = loader.attach_evaluation_labels(
            predictions,
            self.settings.ranker.relevance_grades,
            trade_calendar=label_calendar,
            maturity_cutoff=label_cutoff,
        )
        ranking = _ranking_metrics(evaluation, labeled_evaluation, self.settings)
        ranking["sample_selection_audit"] = {
            "contract": "labeled_subset_audit_v1",
            "horizon": horizon,
            "train": train.sample_selection_by_date.to_dict("records"),
            "validation": validation.sample_selection_by_date.to_dict("records"),
        }
        executable = (
            self._executable_metrics(predictions, horizon, execution_contract)
            if require_executable
            else {"status": "NOT_REQUIRED", "accounting_schema_version": ACCOUNTING_SCHEMA_VERSION}
        )

        def save_model(path: Path) -> None:
            model.booster_.save_model(str(path))

        return FoldExecutionResult(
            predictions=predictions,
            validation_metrics=cast(
                JsonObject,
                evaluate_ranker(
                    validation,
                    validation_predictions,
                    self.settings.ranker.ndcg_at,
                    self.settings.ranker.portfolio_fractions,
                ),
            ),
            ranking_metrics=ranking,
            executable_metrics=executable,
            feature_importance=cast(list[JsonObject], feature_importance(model, features)),
            training_compute=runtime.model_dump(mode="json"),
            model_saver=save_model,
        )

    def _executable_metrics(
        self,
        predictions: pd.DataFrame,
        horizon: int,
        execution_contract: JsonObject,
    ) -> JsonObject:
        dates = tuple(sorted(predictions["trade_date"].astype(str).unique()))
        execution = self.settings.backtest.model_copy(
            update={
                "execution": "next_open",
                "holding_period_days": horizon,
                "top_n": REQUIRED_TOP_N,
            }
        )
        tail = _validated_execution_tail_contract(execution_contract)
        lockbox_start = str(tail["prospective_lockbox_start_exclusive"])
        governed_cutoff = _governed_execution_cutoff(tail)
        full_calendar = load_calendar(
            self.raw_root,
            dates[0],
            dates[-1],
            None,
            maximum_date=governed_cutoff,
        )
        if not full_calendar or dates[-1] not in full_calendar:
            raise DataValidationError(
                "BACKTEST_EXECUTION_DATA_CUTOFF_INVALID: evaluation end is unavailable"
            )
        if full_calendar[-1] >= lockbox_start:
            raise DataValidationError(
                "RESEARCH_LOCKBOX_VIOLATION: walk-forward execution tail is not bounded"
            )
        signals = _signals(predictions)
        execution_codes = _execution_signal_codes(signals, max(REQUIRED_TOP_N))
        identity = SecurityIdentityResolver.from_path(self.settings.security_identity.mapping_path)
        lifecycle = SecurityLifecycleResolver.from_path(
            self.settings.security_identity.lifecycle_path
        )
        transitions = self._identity_transitions
        if transitions is None:
            raise DataValidationError("SECURITY_IDENTITY_TRANSITION_CONTRACT_REQUIRED")
        evaluation_end_index = full_calendar.index(dates[-1])
        tail_sessions = EXECUTION_TAIL_LOAD_CHUNK_SESSIONS
        while True:
            calendar_end_index = min(
                len(full_calendar) - 1, evaluation_end_index + horizon + tail_sessions + 1
            )
            calendar = full_calendar[: calendar_end_index + 1]
            prices = load_execution_prices(
                self.raw_root,
                self.processed_root,
                calendar[0],
                calendar[-1],
                self.settings.universe.price_tolerance,
                identity_resolver=identity,
                lifecycle_resolver=lifecycle,
                identity_transitions=transitions,
                ts_codes=execution_codes,
            )
            benchmark = load_benchmark(
                self.raw_root, execution.benchmark_index_code, calendar[0], calendar[-1]
            )
            inputs = BacktestInputs(
                signals=signals,
                prices=prices,
                calendar=tuple(calendar),
                benchmark=benchmark,
                identity_transitions=transitions.transition_records(),
                identity_transition_version=transitions.artifact_version,
                identity_transition_hash=transitions.artifact_hash,
            )
            try:
                results = tuple(
                    simulate_portfolio(
                        inputs,
                        top_n=top_n,
                        settings=execution,
                        purpose="executable_validation",
                        delayed_exit_policy="carry_to_calendar_end",
                    )
                    for top_n in REQUIRED_TOP_N
                )
                break
            except DataValidationError as error:
                if (
                    not _is_intermediate_execution_cutoff(error)
                    or calendar_end_index == len(full_calendar) - 1
                ):
                    raise
                tail_sessions += EXECUTION_TAIL_LOAD_CHUNK_SESSIONS
        if any(
            not result.holdings.empty
            and result.holdings["trade_date"].astype(str).eq(calendar[-1]).any()
            for result in results
        ):
            raise DataValidationError("walk-forward fold has unresolved executable positions")
        return {
            "status": "COMPLETE",
            "accounting_schema_version": ACCOUNTING_SCHEMA_VERSION,
            "top_n": {str(result.top_n): result.metrics for result in results},
            "accounting_summaries": {
                str(result.top_n): result.accounting_summary for result in results
            },
            "delayed_exit_evidence": {
                str(result.top_n): _delayed_exit_evidence(result.trades) for result in results
            },
            "execution_tail": {
                **tail,
                "governed_data_cutoff": governed_cutoff,
                "actual_calendar_cutoff": calendar[-1],
                "actual_execution_end": max(
                    str(result.daily_returns["trade_date"].astype(str).max()) for result in results
                ),
                "actual_execution_end_by_top_n": {
                    str(result.top_n): str(result.daily_returns["trade_date"].astype(str).max())
                    for result in results
                },
            },
            "cost_policy_hash": str(results[0].cost_policy["cost_policy_hash"]),
        }


def _build_prediction_frame(
    evaluation_frame: pd.DataFrame,
    evaluation_predictions: np.ndarray,
) -> pd.DataFrame:
    """Build the canonical upstream prediction artifact consumed by executable validation."""

    predictions = evaluation_frame.loc[:, ["trade_date", "ts_code"]].copy()
    predictions["prediction_score"] = evaluation_predictions
    return predictions


def _validated_execution_tail_contract(execution_contract: JsonObject) -> JsonObject:
    tail = execution_contract.get("execution_tail")
    if not isinstance(tail, dict):
        raise DataValidationError("WALK_FORWARD_EXECUTION_TAIL_CONTRACT_INVALID")
    required = {
        "policy_version": EXECUTION_TAIL_POLICY_VERSION,
        "policy": EXECUTION_TAIL_POLICY,
        "unresolved_at_cutoff": "fail_closed",
    }
    if any(tail.get(key) != value for key, value in required.items()):
        raise DataValidationError("WALK_FORWARD_EXECUTION_TAIL_CONTRACT_INVALID")
    for key in ("processed_data_end", "prospective_lockbox_start_exclusive"):
        value = tail.get(key)
        if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
            raise DataValidationError("WALK_FORWARD_EXECUTION_TAIL_CONTRACT_INVALID")
    alert = tail.get("sell_delay_alert_sessions")
    if not isinstance(alert, int) or isinstance(alert, bool) or alert < 0:
        raise DataValidationError("WALK_FORWARD_EXECUTION_TAIL_CONTRACT_INVALID")
    lifecycle_version = tail.get("security_lifecycle_policy_version")
    lifecycle_hash = tail.get("security_lifecycle_policy_hash")
    lifecycle_count = tail.get("security_lifecycle_event_count")
    if (
        not isinstance(lifecycle_version, str)
        or not lifecycle_version
        or not isinstance(lifecycle_hash, str)
        or len(lifecycle_hash) != 64
        or not isinstance(lifecycle_count, int)
        or isinstance(lifecycle_count, bool)
        or lifecycle_count < 0
    ):
        raise DataValidationError("WALK_FORWARD_EXECUTION_TAIL_CONTRACT_INVALID")
    return cast(JsonObject, tail)


def _execution_signal_codes(signals: pd.DataFrame, top_n: int) -> tuple[str, ...]:
    """Return the exact union of securities that can enter any requested Top-N."""

    selected: set[str] = set()
    for _, daily in signals.groupby("trade_date", sort=True):
        selected.update(
            daily.sort_values(["score", "ts_code"], ascending=[False, True], kind="stable")
            .head(top_n)["ts_code"]
            .astype(str)
        )
    if not selected:
        raise DataValidationError("BACKTEST_MARKET_DATA_INCOMPLETE: no executable signal codes")
    return tuple(sorted(selected))


def _is_intermediate_execution_cutoff(error: DataValidationError) -> bool:
    return str(error).startswith(
        "BACKTEST_UNRESOLVED_POSITION: open positions remain at governed cutoff"
    )


def _governed_execution_cutoff(tail: JsonObject) -> str:
    lockbox_start = str(tail["prospective_lockbox_start_exclusive"])
    try:
        lockbox_previous_date = (
            datetime.strptime(lockbox_start, "%Y%m%d") - timedelta(days=1)
        ).strftime("%Y%m%d")
    except ValueError as error:
        raise DataValidationError("WALK_FORWARD_EXECUTION_TAIL_CONTRACT_INVALID") from error
    cutoff = min(str(tail["processed_data_end"]), lockbox_previous_date)
    if cutoff >= lockbox_start:
        raise DataValidationError(
            "RESEARCH_LOCKBOX_VIOLATION: walk-forward execution cutoff reaches lockbox"
        )
    return cutoff


def _delayed_exit_evidence(trades: pd.DataFrame) -> JsonObject:
    if trades.empty or "sell_delay_breached" not in trades:
        return {
            "breached_positions": 0,
            "maximum_delayed_exit_sessions": 0,
            "resolutions": [],
        }
    breached = trades[trades["sell_delay_breached"].fillna(False).astype(bool)]
    resolved = breached[
        (breached["side"] == "sell") & breached["status"].isin(["filled", "terminal_writeoff"])
    ]
    resolutions = [
        {
            "position_id": str(row.position_id),
            "ts_code": str(row.ts_code),
            "resolution_date": str(row.trade_date),
            "resolution_status": str(row.status),
            "delayed_exit_sessions": int(cast(Any, row).delayed_exit_days),
        }
        for row in resolved.itertuples(index=False)
    ]
    return {
        "breached_positions": int(breached["position_id"].dropna().astype(str).nunique()),
        "maximum_delayed_exit_sessions": (
            int(breached["delayed_exit_days"].astype(int).max()) if not breached.empty else 0
        ),
        "resolutions": resolutions,
    }


class MultiFoldEvaluationRunner:
    """Run every fold in one exact horizon experiment and aggregate validated evidence."""

    def __init__(
        self,
        *,
        reports_root: Path,
        settings: AppSettings,
        executor: FoldExecutor,
        research_policy_path: Path = Path("config/research_policy.yaml"),
        lifecycle_audit_required: bool = True,
    ) -> None:
        self.reports_root = reports_root
        self.settings = settings
        self.executor = executor
        self.research_policy_path = research_policy_path
        self.lifecycle_audit_required = lifecycle_audit_required

    def run(
        self,
        *,
        experiment_manifest: Path,
        experiment_id: str,
        feature_provenance_path: Path,
        require_executable: bool = True,
        lifecycle_scan_manifest: Path | None = None,
        identity_transition_artifact: Path | None = None,
        explicit_no_identity_transitions: bool = False,
    ) -> WalkForwardEvaluationResult:
        plan = _load_json(experiment_manifest, "horizon experiment manifest")
        experiment = _select_experiment(plan, experiment_id)
        horizon = _required_int(experiment, "horizon")
        policy = load_research_policy(self.research_policy_path)
        provenance = validate_governed_feature_set(
            feature_provenance_path,
            reports_root=self.reports_root,
        )
        provenance_sha256 = feature_provenance_hash(feature_provenance_path)
        _validate_feature_lineage(plan, provenance, provenance_sha256)
        current_source_identity = self.executor.validate_sources(plan)
        raw_folds = _load_eligible_folds(plan, experiment, horizon)
        diagnostics_horizon = _diagnostics_horizon(provenance, self.reports_root)
        if provenance.selection_end is None:
            raise DataValidationError("feature selection end is required for governed evaluation")
        selection_information_end = self.executor.mature_information_end(
            provenance.selection_end,
            diagnostics_horizon,
        )
        folds = tuple(
            _with_research_classification(fold, selection_information_end) for fold in raw_folds
        )
        for fold in folds:
            evaluation_information_end = self.executor.mature_information_end(
                str(fold["evaluation_end"]), horizon
            )
            enforce_research_window(
                policy,
                consumer="walk_forward_evaluation",
                start_date=str(fold["evaluation_start"]),
                end_date=evaluation_information_end,
            )
        if require_executable and self.lifecycle_audit_required and lifecycle_scan_manifest is None:
            raise DataValidationError("SECURITY_LIFECYCLE_AUDIT_REQUIRED")
        transitions: SecurityIdentityTransitionResolver | None = None
        if require_executable and self.lifecycle_audit_required:
            if identity_transition_artifact is not None and explicit_no_identity_transitions:
                raise DataValidationError("SECURITY_IDENTITY_TRANSITION_CONTRACT_INVALID")
            if identity_transition_artifact is None and not explicit_no_identity_transitions:
                raise DataValidationError("SECURITY_IDENTITY_TRANSITION_CONTRACT_REQUIRED")
            transitions = load_identity_transition_contract(
                mode="artifact" if identity_transition_artifact is not None else "none",
                artifact_path=identity_transition_artifact,
            )
            if not isinstance(self.executor, RankerFoldExecutor):
                raise DataValidationError("SECURITY_IDENTITY_TRANSITION_CONTRACT_REQUIRED")
            identity_resolver = SecurityIdentityResolver.from_path(
                self.settings.security_identity.mapping_path
            )
            transitions.validate_alias_coexistence(identity_resolver)
            self.executor.bind_identity_transitions(transitions)
        execution_contract = self.executor.execution_contract(
            horizon=horizon,
            require_executable=require_executable,
            prospective_lockbox_start=policy.prospective_lockbox.start_date,
        )
        if require_executable and self.lifecycle_audit_required:
            if lifecycle_scan_manifest is None:
                raise DataValidationError("SECURITY_LIFECYCLE_AUDIT_REQUIRED")
            if not isinstance(self.executor, RankerFoldExecutor):
                raise DataValidationError("SECURITY_LIFECYCLE_AUDIT_REQUIRED: unsupported executor")
            identity_resolver = SecurityIdentityResolver.from_path(
                self.settings.security_identity.mapping_path
            )
            lifecycle_evidence = SecurityLifecycleResolver.from_path(
                self.settings.security_identity.lifecycle_path
            )
            lifecycle_policy = LifecycleAuditPolicy.from_path(
                self.settings.security_identity.lifecycle_policy_path
            )
            listing_metadata = SecurityListingMetadataResolver.from_path(
                self.settings.security_identity.listing_metadata_path
            )
            audit = validate_pass_lifecycle_scan(
                lifecycle_scan_manifest,
                required_start=min(str(fold["train_start"]) for fold in folds),
                required_end=_governed_execution_cutoff(execution_contract["execution_tail"]),
                raw_root=self.executor.raw_root,
                processed_root=self.executor.processed_root,
                identity_resolver=identity_resolver,
                lifecycle_evidence=lifecycle_evidence,
                lifecycle_policy=lifecycle_policy,
                identity_transitions=transitions,
                listing_metadata=listing_metadata,
            )
            execution_contract = {
                **execution_contract,
                "security_lifecycle_audit": {
                    "lifecycle_scan_id": audit["scan_id"],
                    "lifecycle_scan_manifest_hash": _file_hash(lifecycle_scan_manifest),
                    "lifecycle_policy_hash": audit["policy_hash"],
                    "lifecycle_evidence_hash": audit["lifecycle_evidence_hash"],
                    "source_inventory_hash": audit["source_inventory_hash"],
                    "security_identity_transition_version": audit[
                        "security_identity_transition_version"
                    ],
                    "security_identity_transition_hash": audit["security_identity_transition_hash"],
                    "listing_metadata_version": audit["listing_metadata_version"],
                    "listing_metadata_hash": audit["listing_metadata_hash"],
                },
            }
        identities = _experiment_identities(
            plan=plan,
            experiment=experiment,
            fold_contracts=tuple(
                {
                    "fold_id": str(fold["fold_id"]),
                    "research_validity": fold["research_validity"],
                }
                for fold in folds
            ),
            feature_set_id=provenance.feature_set_id,
            feature_provenance_hash=provenance_sha256,
            research_policy_hash=policy.policy_hash,
            semantic_parameters=ranker_semantic_parameters(self.settings.ranker),
            source_identity=current_source_identity,
            execution_contract=execution_contract,
        )
        identity = str(identities["run_identity"])
        run_id = f"walk_forward_{identity[:16]}"
        output_dir = self.reports_root / "research" / "walk_forward" / run_id
        existing = _existing_complete(output_dir, identity)
        if existing is not None:
            return existing
        fold_root = output_dir / "folds"
        fold_root.mkdir(parents=True, exist_ok=True)
        fold_manifests: list[JsonObject] = []
        for fold in folds:
            fold_dir = fold_root / str(fold["fold_id"])
            validated = _existing_fold(fold_dir, fold, identity)
            if validated is None:
                result = self.executor.execute(
                    fold=fold,
                    horizon=horizon,
                    features=provenance.features,
                    require_executable=require_executable,
                    execution_contract=execution_contract,
                )
                _publish_fold(
                    fold_dir,
                    fold=fold,
                    experiment_identity=identity,
                    modeling_identity=str(identities["modeling_identity"]),
                    execution_identity=str(identities["execution_identity"]),
                    execution_contract=execution_contract,
                    horizon=horizon,
                    feature_set_id=provenance.feature_set_id,
                    feature_set_hash=provenance.feature_list_hash,
                    feature_provenance_hash=provenance_sha256,
                    walk_forward_plan_hash=str(plan["folds_manifest_hash"]),
                    horizon_plan_hash=_file_hash(experiment_manifest),
                    feature_hash=provenance.feature_list_hash,
                    research_policy_hash=policy.policy_hash,
                    semantic_parameters=ranker_semantic_parameters(self.settings.ranker),
                    source_identity={
                        "plan_identity_hash": plan.get("plan_identity_hash"),
                        "folds_manifest_hash": plan.get("folds_manifest_hash"),
                        "folds_hash": plan.get("folds_hash"),
                        "universe_hash": plan.get("universe_hash"),
                        "labels_fingerprint": plan.get("labels_fingerprint"),
                        "validated_current_sources": current_source_identity,
                    },
                    result=result,
                )
                validated = _existing_fold(fold_dir, fold, identity)
                assert validated is not None
            fold_manifests.append(validated)
        aggregate = _aggregate(fold_manifests)
        _publish_aggregate(
            output_dir,
            identity=identity,
            modeling_identity=str(identities["modeling_identity"]),
            execution_identity=str(identities["execution_identity"]),
            execution_contract=execution_contract,
            run_id=run_id,
            plan_path=str(experiment_manifest),
            experiment=experiment,
            feature_provenance_path=feature_provenance_locator(
                feature_provenance_path, self.reports_root
            ),
            feature_set_id=provenance.feature_set_id,
            feature_set_hash=provenance.feature_list_hash,
            feature_provenance_hash=provenance_sha256,
            walk_forward_plan_hash=str(plan["folds_manifest_hash"]),
            horizon_plan_hash=_file_hash(experiment_manifest),
            research_policy_path=str(self.research_policy_path),
            research_policy_hash=policy.policy_hash,
            folds=fold_manifests,
            aggregate=aggregate,
        )
        return WalkForwardEvaluationResult(run_id, "COMPLETE", len(folds), output_dir)


class WalkForwardRecoveryInspector:
    """Inspect multi-fold publication state without changing it."""

    def __init__(self, reports_root: Path) -> None:
        self.reports_root = reports_root

    def inspect(self, experiment_id: str) -> JsonObject:
        root = self.reports_root / "research" / "walk_forward" / experiment_id
        issues: list[str] = []
        if not root.is_dir():
            return {"status": "ACTION_REQUIRED", "issues": ["experiment directory missing"]}
        for staging in root.parent.glob(f".{experiment_id}*.staging"):
            issues.append(f"stale staging directory: {staging}")
        for staging in root.rglob(".*.staging-*"):
            issues.append(f"stale fold staging directory: {staging}")
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            issues.append("top-level manifest missing")
        else:
            try:
                validate_completed_walk_forward_artifact(root)
            except DataValidationError as error:
                issues.append(str(error))
        return {"status": "CLEAN" if not issues else "ACTION_REQUIRED", "issues": issues}


def walk_forward_status(reports_root: Path, run_id: str) -> JsonObject:
    """Return a validated immutable experiment status snapshot."""

    root = reports_root / "research" / "walk_forward" / run_id
    manifest = validate_completed_walk_forward_artifact(root)
    return {
        "run_id": run_id,
        "status": manifest.get("status"),
        "fold_count": len(cast(dict[str, str], manifest["fold_manifest_hashes"])),
        "research_policy_hash": manifest.get("research_policy_hash"),
        "feature_set_id": manifest.get("feature_set_id"),
        "evaluation_contract_version": manifest.get("evaluation_contract_version", 2),
        "evidence_classification": (
            "CURRENT"
            if manifest.get("schema_version") == SCHEMA_VERSION
            and manifest.get("evaluation_contract_version") == EVALUATION_CONTRACT_VERSION
            else "LEGACY_READ_ONLY"
        ),
    }


def _ranking_metrics(
    prediction_dataset: RankerPredictionDataset,
    labeled_predictions: pd.DataFrame,
    settings: AppSettings,
) -> JsonObject:
    """Evaluate frozen predictions without allowing labels to alter the signal universe."""

    available = labeled_predictions["is_label_available"].astype(bool)
    metric_frame = labeled_predictions.loc[available].copy()
    group_sizes = metric_frame.groupby("trade_date")["ts_code"].transform("size")
    metric_frame = metric_frame.loc[group_sizes >= settings.ranker.minimum_group_size].reset_index(
        drop=True
    )
    if metric_frame.empty:
        raise DataValidationError(
            "walk-forward evaluation has no mature label groups meeting minimum_group_size"
        )
    metric_frame["relevance"] = metric_frame["relevance"].astype("int32")
    metric_dataset = RankerDataset(frame=metric_frame, feature_names=())
    base = cast(
        JsonObject,
        evaluate_ranker(
            metric_dataset,
            metric_frame["prediction_score"].to_numpy(dtype=float),
            settings.ranker.ndcg_at,
            settings.ranker.portfolio_fractions,
        ),
    )
    base.pop("yearly", None)
    daily_values = [
        group["prediction_score"].corr(group["future_excess_ret_5d"], method="spearman")
        for _, group in metric_frame.groupby("trade_date", sort=True)
    ]
    values = pd.to_numeric(pd.Series(daily_values, dtype="float64"), errors="coerce").dropna()
    coverage = prediction_dataset.coverage_by_date.copy()
    scored = (
        labeled_predictions.groupby("trade_date", sort=True)
        .agg(
            scored_rows=("ts_code", "size"),
            finite_score_rows=("prediction_score", lambda value: int(np.isfinite(value).sum())),
            mature_label_rows=("is_label_mature", "sum"),
            available_label_rows=("is_label_available", "sum"),
        )
        .reset_index()
    )
    coverage = coverage.merge(scored, on="trade_date", how="left", validate="one_to_one")
    count_columns = (
        "expected_universe_rows",
        "feature_rows_present",
        "scored_rows",
        "finite_score_rows",
        "mature_label_rows",
        "available_label_rows",
    )
    for column in count_columns:
        coverage[column] = coverage[column].fillna(0).astype(int)
    coverage["unavailable_label_rows"] = coverage["scored_rows"] - coverage["available_label_rows"]
    coverage["prediction_coverage"] = pd.Series(
        [
            _optional_ratio(int(scored), int(expected))
            for scored, expected in zip(
                coverage["scored_rows"], coverage["expected_universe_rows"], strict=True
            )
        ],
        dtype="object",
    )
    coverage["label_coverage"] = pd.Series(
        [
            _optional_ratio(int(available_count), int(scored))
            for available_count, scored in zip(
                coverage["available_label_rows"], coverage["scored_rows"], strict=True
            )
        ],
        dtype="object",
    )
    expected_total = int(coverage["expected_universe_rows"].sum())
    scored_total = int(coverage["scored_rows"].sum())
    available_total = int(coverage["available_label_rows"].sum())
    reason_counts = {
        str(reason): int(count)
        for reason, count in labeled_predictions.loc[~available, "label_unavailable_reason"]
        .value_counts(dropna=False)
        .items()
    }
    top_label_proxies = _top_label_proxies(
        labeled_predictions,
        settings.ranker.portfolio_fractions,
    )
    for name, payload in top_label_proxies.items():
        base[name] = payload["conditional_mean_future_excess_ret"]
    base.update(
        {
            "rank_ic_median": float(values.median()),
            "rank_ic_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "positive_rank_ic_ratio": float((values > 0).mean()),
            "coverage": _optional_ratio(scored_total, expected_total),
            "prediction_coverage": _optional_ratio(scored_total, expected_total),
            "label_coverage": _optional_ratio(available_total, scored_total),
            "expected_universe_rows": expected_total,
            "feature_rows_present": int(coverage["feature_rows_present"].sum()),
            "scored_rows": scored_total,
            "finite_score_rows": int(coverage["finite_score_rows"].sum()),
            "mature_label_rows": int(coverage["mature_label_rows"].sum()),
            "available_label_rows": available_total,
            "unavailable_label_rows": int(coverage["unavailable_label_rows"].sum()),
            "unavailable_label_reasons": reason_counts,
            "coverage_by_date": coverage.to_dict("records"),
            "top_n_label_proxies": top_label_proxies,
            "signal_dates": int(labeled_predictions["trade_date"].nunique()),
            "securities_scored": int(labeled_predictions["ts_code"].nunique()),
            "metric_rows": len(metric_frame),
            "metric_groups": int(metric_frame["trade_date"].nunique()),
            "metric_scope": "available labels joined after frozen prediction membership",
        }
    )
    return base


def _top_label_proxies(
    labeled_predictions: pd.DataFrame,
    fractions: tuple[float, ...],
) -> JsonObject:
    output: JsonObject = {}
    for fraction in fractions:
        requested = 0
        available = 0
        values: list[float] = []
        for _, daily in labeled_predictions.groupby("trade_date", sort=True):
            count = max(1, int(math.ceil(len(daily) * fraction)))
            top = daily.sort_values(
                ["prediction_score", "ts_code"], ascending=[False, True], kind="stable"
            ).head(count)
            requested += count
            valid = top["is_label_available"].astype(bool)
            available += int(valid.sum())
            values.extend(
                pd.to_numeric(top.loc[valid, "future_excess_ret_5d"], errors="coerce")
                .dropna()
                .astype(float)
                .tolist()
            )
        name = portfolio_metric_name(fraction)
        output[name] = {
            "requested_rows": requested,
            "available_label_rows": available,
            "unavailable_label_rows": requested - available,
            "conditional_mean_future_excess_ret": (float(np.mean(values)) if values else None),
        }
    return output


def _optional_ratio(numerator: int | float, denominator: int | float) -> float | None:
    divisor = int(denominator)
    return None if divisor == 0 else float(int(numerator) / divisor)


def _load_eligible_folds(
    plan: JsonObject, experiment: JsonObject, horizon: int
) -> tuple[JsonObject, ...]:
    fold_path = Path(str(plan.get("folds_manifest", "")))
    if _file_hash(fold_path) != plan.get("folds_manifest_hash"):
        raise DataValidationError("walk-forward fold manifest hash changed")
    fold_manifest = _load_json(fold_path, "walk-forward fold manifest")
    outputs = cast(JsonObject, fold_manifest.get("outputs", {}))
    folds_path = Path(str(outputs.get("folds", "folds.json")))
    if not folds_path.is_absolute():
        folds_path = fold_path.parent / folds_path
    if _file_hash(folds_path) != plan.get("folds_hash"):
        raise DataValidationError("walk-forward folds hash changed")
    payload = _load_json(folds_path, "walk-forward folds")
    raw = payload.get("folds")
    if not isinstance(raw, list) or not raw:
        raise DataValidationError("walk-forward plan has no folds")
    by_id = {str(item.get("fold_id")): item for item in raw if isinstance(item, dict)}
    if len(by_id) != len(raw) or "" in by_id:
        raise DataValidationError("walk-forward fold IDs must be non-empty and unique")
    references: list[JsonObject] = []
    for period_name in ("selection_period", "final_test_period"):
        period = experiment.get(period_name)
        if not isinstance(period, dict) or not isinstance(period.get("folds"), list):
            raise DataValidationError(f"horizon experiment lacks {period_name} folds")
        references.extend(cast(list[JsonObject], period["folds"]))
    selected: list[JsonObject] = []
    seen: set[str] = set()
    required = required_temporal_gap_sessions(horizon)
    for reference in references:
        fold_id = str(reference.get("fold_id", ""))
        if fold_id in seen:
            continue
        fold = by_id.get(fold_id)
        if fold is None:
            raise DataValidationError(f"required fold is missing: {fold_id}")
        resolved = dict(fold)
        resolved["evaluation_start"] = str(reference["evaluation_start"])
        resolved["evaluation_end"] = str(reference["evaluation_end"])
        _validate_fold(resolved, required)
        selected.append(resolved)
        seen.add(fold_id)
    return tuple(selected)


def _diagnostics_horizon(
    provenance: FeatureSetProvenance,
    reports_root: Path,
) -> int:
    locator = provenance.source_diagnostics_manifest_locator
    if not locator:
        raise DataValidationError("feature provenance lacks diagnostics manifest locator")
    root = reports_root.resolve()
    manifest_path = (root / locator).resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError as error:
        raise DataValidationError("feature diagnostics locator escapes reports root") from error
    manifest = _load_json(manifest_path, "feature diagnostics manifest")
    horizon = manifest.get("horizon")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise DataValidationError("feature diagnostics manifest lacks a valid label horizon")
    return horizon


def _with_research_classification(
    fold: JsonObject,
    selection_information_end: str,
) -> JsonObject:
    resolved = dict(fold)
    evaluation_start = str(fold["evaluation_start"])
    parameter_fit_oos = str(fold["validation_end"]) < evaluation_start
    feature_selection_oos = selection_information_end < evaluation_start
    if parameter_fit_oos and feature_selection_oos:
        classification = "STRICT_OOS"
    elif parameter_fit_oos:
        classification = "RETROSPECTIVE_FIXED_FEATURE_REPLAY"
    else:
        classification = "INVALID_PARAMETER_FIT_OVERLAP"
    resolved["research_validity"] = {
        "parameter_fit_oos": parameter_fit_oos,
        "feature_selection_oos": feature_selection_oos,
        "selection_information_end": selection_information_end,
        "evaluation_start": evaluation_start,
        "research_classification": classification,
    }
    if not parameter_fit_oos:
        raise DataValidationError(f"fold parameter fit is not OOS: {fold.get('fold_id')}")
    return resolved


def _validate_feature_lineage(
    plan: JsonObject,
    provenance: FeatureSetProvenance,
    provenance_sha256: str,
) -> None:
    if plan.get("schema_version") != 3 or plan.get("feature_authority") != ("governed_feature_set"):
        raise DataValidationError("WALK_FORWARD_FEATURE_AUTHORITY_INVALID")
    expected = (
        provenance.feature_set_id,
        provenance.feature_list_hash,
        provenance_sha256,
    )
    actual = (
        plan.get("feature_set_id"),
        plan.get("feature_set_hash"),
        plan.get("feature_provenance_hash"),
    )
    if actual != expected or plan.get("feature_hash") != provenance.feature_list_hash:
        raise DataValidationError("WALK_FORWARD_FEATURE_PROVENANCE_MISMATCH")
    fold_path = Path(str(plan.get("folds_manifest", "")))
    fold_manifest = _load_json(fold_path, "walk-forward fold manifest")
    fold_actual = (
        fold_manifest.get("feature_set_id"),
        fold_manifest.get("feature_set_hash"),
        fold_manifest.get("feature_provenance_hash"),
    )
    if (
        fold_manifest.get("schema_version") != 4
        or fold_manifest.get("feature_authority") != "governed_feature_set"
        or fold_actual != expected
        or fold_manifest.get("feature_hash") != provenance.feature_list_hash
    ):
        raise DataValidationError("WALK_FORWARD_FEATURE_PROVENANCE_MISMATCH")


def _validate_fold(fold: JsonObject, required_gap: int) -> None:
    if not (
        str(fold.get("train_end", "")) < str(fold.get("validation_start", ""))
        and str(fold.get("validation_end", "")) < str(fold.get("evaluation_start", ""))
    ):
        raise DataValidationError(f"fold chronology is invalid: {fold.get('fold_id')}")
    if _required_int(fold, "purge_sessions") < required_gap:
        raise DataValidationError(f"fold purge is unsafe: {fold.get('fold_id')}")
    if _required_int(fold, "embargo_sessions") < required_gap:
        raise DataValidationError(f"fold embargo is unsafe: {fold.get('fold_id')}")


def _select_experiment(plan: JsonObject, requested: str) -> JsonObject:
    experiments = plan.get("experiments")
    if not isinstance(experiments, list):
        raise DataValidationError("horizon experiment plan has no experiments")
    matches = [
        item
        for item in experiments
        if isinstance(item, dict) and item.get("experiment_id") == requested
    ]
    if len(matches) != 1:
        raise DataValidationError(f"horizon experiment is not uniquely available: {requested}")
    return matches[0]


def _experiment_identities(
    *,
    plan: JsonObject,
    experiment: JsonObject,
    fold_contracts: tuple[JsonObject, ...],
    feature_set_id: str,
    feature_provenance_hash: str,
    research_policy_hash: str,
    semantic_parameters: JsonObject,
    source_identity: JsonObject,
    execution_contract: JsonObject,
) -> JsonObject:
    modeling = {
        "schema_version": SCHEMA_VERSION,
        "plan_identity_hash": plan.get("plan_identity_hash"),
        "experiment_id": experiment.get("experiment_id"),
        "fold_contracts": fold_contracts,
        "feature_set_id": feature_set_id,
        "feature_provenance_hash": feature_provenance_hash,
        "research_policy_hash": research_policy_hash,
        "semantic_parameters": semantic_parameters,
        "source_identity": source_identity,
    }
    modeling_identity = _payload_hash(modeling)
    execution_identity = _payload_hash(execution_contract)
    return {
        "modeling_identity": modeling_identity,
        "execution_identity": execution_identity,
        "run_identity": _payload_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "modeling_identity": modeling_identity,
                "execution_identity": execution_identity,
            }
        ),
    }


def _publish_fold(
    path: Path,
    *,
    fold: JsonObject,
    experiment_identity: str,
    modeling_identity: str,
    execution_identity: str,
    execution_contract: JsonObject,
    horizon: int,
    feature_set_id: str,
    feature_set_hash: str,
    feature_provenance_hash: str,
    walk_forward_plan_hash: str,
    horizon_plan_hash: str,
    feature_hash: str,
    research_policy_hash: str,
    semantic_parameters: JsonObject,
    source_identity: JsonObject,
    result: FoldExecutionResult,
) -> None:
    expected_compute = execution_contract.get("training_compute")
    if not isinstance(expected_compute, dict) or any(
        result.training_compute.get(key) != value for key, value in expected_compute.items()
    ):
        raise DataValidationError("WALK_FORWARD_TRAINING_COMPUTE_IDENTITY_MISMATCH")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=f".{path.name}.staging-") as temporary:
        staging = Path(temporary)
        result.model_saver(staging / "model.txt")
        result.predictions.to_parquet(staging / "predictions.parquet", index=False)
        for name, payload in (
            ("validation_metrics.json", result.validation_metrics),
            ("ranking_metrics.json", result.ranking_metrics),
            ("executable_metrics.json", result.executable_metrics),
            ("feature_importance.json", {"features": result.feature_importance}),
        ):
            atomic_write_json(staging / name, payload)
        files = {
            name: _file_hash(staging / name)
            for name in (
                "model.txt",
                "predictions.parquet",
                "validation_metrics.json",
                "ranking_metrics.json",
                "executable_metrics.json",
                "feature_importance.json",
            )
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_name": "walk_forward_fold_evidence",
            "technical_status": "VALID",
            "fold": fold,
            "experiment_identity": experiment_identity,
            "modeling_identity": modeling_identity,
            "execution_identity": execution_identity,
            "execution_contract": execution_contract,
            "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
            "horizon": horizon,
            "feature_set_id": feature_set_id,
            "feature_set_hash": feature_set_hash,
            "feature_provenance_hash": feature_provenance_hash,
            "walk_forward_plan_hash": walk_forward_plan_hash,
            "horizon_plan_hash": horizon_plan_hash,
            "feature_hash": feature_hash,
            "research_policy_hash": research_policy_hash,
            "semantic_parameters": semantic_parameters,
            "source_identity": source_identity,
            "training_compute": result.training_compute,
            "research_validity": fold.get("research_validity"),
            "artifact_hashes": files,
        }
        atomic_write_json(staging / "manifest.json", manifest)
        if path.exists():
            raise DataValidationError(f"immutable fold artifact already exists: {path}")
        staging.rename(path)


def _existing_fold(path: Path, fold: JsonObject, identity: str) -> JsonObject | None:
    if not path.exists():
        return None
    return _validate_fold_artifact(
        path,
        expected_identity=identity,
        expected_fold_id=str(fold.get("fold_id", "")),
    )


def _aggregate(folds: list[JsonObject]) -> JsonObject:
    if not folds or any(item.get("technical_status") != "VALID" for item in folds):
        raise DataValidationError("aggregate requires every planned fold to be technically valid")
    return {
        "technical": {
            "total_folds": len(folds),
            "valid_folds": len(folds),
            "failed_folds": 0,
            "incomplete_folds": 0,
            "all_required_folds_valid": True,
        },
        "performance": {},
    }


def _publish_aggregate(
    output_dir: Path,
    *,
    identity: str,
    modeling_identity: str,
    execution_identity: str,
    execution_contract: JsonObject,
    run_id: str,
    plan_path: str,
    experiment: JsonObject,
    feature_provenance_path: str,
    feature_set_id: str,
    feature_set_hash: str,
    feature_provenance_hash: str,
    walk_forward_plan_hash: str,
    horizon_plan_hash: str,
    research_policy_path: str,
    research_policy_hash: str,
    folds: list[JsonObject],
    aggregate: JsonObject,
) -> None:
    fold_hashes = {
        str(item["fold"]["fold_id"]): _file_hash(
            output_dir / "folds" / str(item["fold"]["fold_id"]) / "manifest.json"
        )
        for item in folds
    }
    ranking_rows = [
        _load_json(output_dir / "folds" / fold_id / "ranking_metrics.json", "fold ranking metrics")
        for fold_id in fold_hashes
    ]
    executable_rows = [
        _load_json(
            output_dir / "folds" / fold_id / "executable_metrics.json",
            "fold executable metrics",
        )
        for fold_id in fold_hashes
    ]
    importance_rows = [
        _load_json(
            output_dir / "folds" / fold_id / "feature_importance.json",
            "fold feature importance",
        )
        for fold_id in fold_hashes
    ]
    strict_indices = [
        index
        for index, fold in enumerate(folds)
        if cast(JsonObject, fold.get("research_validity", {})).get("research_classification")
        == "STRICT_OOS"
    ]
    aggregate["performance"] = _classified_metric_distributions(ranking_rows, strict_indices)
    aggregate["executable_performance"] = _classified_executable_distributions(
        executable_rows, strict_indices
    )
    aggregate["feature_importance_stability"] = _importance_stability(importance_rows)
    atomic_write_json(output_dir / "aggregate_metrics.json", aggregate)
    pd.DataFrame(
        [
            {
                "fold_id": fold_id,
                "research_classification": cast(
                    JsonObject, folds[index].get("research_validity", {})
                ).get("research_classification"),
                **{key: value for key, value in row.items() if isinstance(value, (int, float))},
            }
            for index, (fold_id, row) in enumerate(zip(fold_hashes, ranking_rows, strict=True))
        ]
    ).to_parquet(output_dir / "fold_summary.parquet", index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_name": "multi_fold_walk_forward_evidence",
        "status": "COMPLETE",
        "identity": identity,
        "modeling_identity": modeling_identity,
        "execution_identity": execution_identity,
        "execution_contract": execution_contract,
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "run_id": run_id,
        "plan_path": plan_path,
        "experiment": experiment,
        "feature_provenance_path": feature_provenance_path,
        "feature_set_id": feature_set_id,
        "feature_set_hash": feature_set_hash,
        "feature_provenance_hash": feature_provenance_hash,
        "walk_forward_plan_hash": walk_forward_plan_hash,
        "horizon_plan_hash": horizon_plan_hash,
        "research_policy_path": research_policy_path,
        "research_policy_hash": research_policy_hash,
        "fold_manifest_hashes": fold_hashes,
        "aggregate_metrics_sha256": _file_hash(output_dir / "aggregate_metrics.json"),
        "fold_summary_sha256": _file_hash(output_dir / "fold_summary.parquet"),
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)


def _metric_distributions(rows: list[JsonObject]) -> JsonObject:
    keys = sorted(
        set.intersection(
            *[
                {
                    key
                    for key, value in row.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                for row in rows
            ]
        )
    )
    output: JsonObject = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) != len(values):
            raise DataValidationError(f"aggregate metric contains non-finite values: {key}")
        output[key] = {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
            "minimum": float(np.min(finite)),
            "maximum": float(np.max(finite)),
            "positive_fold_ratio": float(np.mean(finite > 0)),
            "worst_fold_index": int(np.argmin(finite)),
            "best_fold_index": int(np.argmax(finite)),
        }
    return output


def _classified_metric_distributions(
    rows: list[JsonObject], strict_indices: list[int]
) -> JsonObject:
    descriptive = _metric_distributions(rows)
    if not strict_indices:
        return {
            "status": "NO_STRICT_OOS_FOLDS",
            "strict_oos_fold_count": 0,
            "descriptive_all_folds": descriptive,
        }
    strict = _metric_distributions([rows[index] for index in strict_indices])
    strict["status"] = "COMPLETE"
    strict["strict_oos_fold_count"] = len(strict_indices)
    strict["descriptive_all_folds"] = descriptive
    return strict


def _executable_distributions(rows: list[JsonObject]) -> JsonObject:
    if all(row.get("status") == "NOT_REQUIRED" for row in rows):
        return {"status": "NOT_REQUIRED"}
    if any(
        row.get("status") != "COMPLETE" or row.get("accounting_schema_version") != 2 for row in rows
    ):
        raise DataValidationError("executable fold evidence is missing or not accounting schema v2")
    flattened: list[JsonObject] = []
    for row in rows:
        top_n = row.get("top_n")
        if not isinstance(top_n, dict):
            raise DataValidationError("executable fold evidence has no Top-N metrics")
        record: JsonObject = {}
        for bucket, metrics in top_n.items():
            if not isinstance(metrics, dict):
                raise DataValidationError("executable Top-N metrics must be objects")
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    record[f"top_{bucket}_{name}"] = value
        flattened.append(record)
    return {"status": "COMPLETE", "metrics": _metric_distributions(flattened)}


def _classified_executable_distributions(
    rows: list[JsonObject], strict_indices: list[int]
) -> JsonObject:
    descriptive = _executable_distributions(rows)
    if not strict_indices:
        return {
            "status": "NO_STRICT_OOS_FOLDS",
            "strict_oos_fold_count": 0,
            "descriptive_all_folds": descriptive,
        }
    strict = _executable_distributions([rows[index] for index in strict_indices])
    strict["strict_oos_fold_count"] = len(strict_indices)
    strict["descriptive_all_folds"] = descriptive
    return strict


def _importance_stability(rows: list[JsonObject]) -> list[JsonObject]:
    by_feature: dict[str, list[float]] = {}
    for row in rows:
        features = row.get("features")
        if not isinstance(features, list):
            raise DataValidationError("fold feature importance is missing")
        ranked = sorted(
            (item for item in features if isinstance(item, dict)),
            key=lambda item: float(item.get("gain", 0.0)),
            reverse=True,
        )
        for rank, item in enumerate(ranked, start=1):
            by_feature.setdefault(str(item.get("feature", "")), []).append(float(rank))
    return [
        {
            "feature": feature,
            "fold_presence": len(ranks),
            "median_rank": float(np.median(ranks)),
            "rank_std": float(np.std(ranks, ddof=1)) if len(ranks) > 1 else 0.0,
        }
        for feature, ranks in sorted(by_feature.items())
        if feature
    ]


def _existing_complete(path: Path, identity: str) -> WalkForwardEvaluationResult | None:
    if not (path / "manifest.json").is_file():
        return None
    manifest = validate_completed_walk_forward_artifact(path, expected_identity=identity)
    return WalkForwardEvaluationResult(
        str(manifest["run_id"]),
        str(manifest["status"]),
        len(manifest["fold_manifest_hashes"]),
        path,
    )


def validate_completed_walk_forward_artifact(
    path: Path,
    *,
    expected_identity: str | None = None,
) -> JsonObject:
    """Validate a COMPLETE multi-fold artifact from its root through every leaf."""

    manifest = _load_json(path / "manifest.json", "walk-forward manifest")
    schema_version = manifest.get("schema_version")
    if (
        schema_version not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}
        or manifest.get("artifact_name") != "multi_fold_walk_forward_evidence"
        or manifest.get("status") != "COMPLETE"
        or manifest.get("run_id") != path.name
    ):
        raise DataValidationError("WALK_FORWARD_MANIFEST_INVALID")
    identity = manifest.get("identity")
    if not isinstance(identity, str) or not identity:
        raise DataValidationError("WALK_FORWARD_MANIFEST_INVALID: identity missing")
    if expected_identity is not None and identity != expected_identity:
        raise DataValidationError(f"walk-forward identity conflict: {path}")
    root_evaluation_contract = manifest.get("evaluation_contract_version")
    if schema_version == SCHEMA_VERSION and (
        not isinstance(root_evaluation_contract, int)
        or isinstance(root_evaluation_contract, bool)
        or root_evaluation_contract
        not in {
            OLDER_EVALUATION_CONTRACT_VERSION,
            LEGACY_EVALUATION_CONTRACT_VERSION,
            PREVIOUS_EVALUATION_CONTRACT_VERSION,
            PREVIOUS_POST_H5_EVALUATION_CONTRACT_VERSION,
            EVALUATION_CONTRACT_VERSION,
        }
        or not isinstance(manifest.get("modeling_identity"), str)
        or not isinstance(manifest.get("execution_identity"), str)
        or not isinstance(manifest.get("execution_contract"), dict)
    ):
        raise DataValidationError("WALK_FORWARD_EXECUTION_CONTRACT_INVALID")
    validated_evaluation_contract = (
        root_evaluation_contract
        if isinstance(root_evaluation_contract, int)
        and not isinstance(root_evaluation_contract, bool)
        else EVALUATION_CONTRACT_VERSION
    )
    expected_root_entries = {
        "aggregate_metrics.json",
        "fold_summary.parquet",
        "folds",
        "manifest.json",
    }
    if {child.name for child in path.iterdir()} != expected_root_entries:
        raise DataValidationError("WALK_FORWARD_ROOT_ARTIFACT_SET_MISMATCH")
    _validate_root_hash(
        path / "aggregate_metrics.json",
        manifest.get("aggregate_metrics_sha256"),
        "WALK_FORWARD_AGGREGATE_HASH_MISMATCH",
    )
    _validate_root_hash(
        path / "fold_summary.parquet",
        manifest.get("fold_summary_sha256"),
        "WALK_FORWARD_FOLD_SUMMARY_HASH_MISMATCH",
    )
    expected = manifest.get("fold_manifest_hashes")
    if (
        not isinstance(expected, dict)
        or not expected
        or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in expected.items()
        )
    ):
        raise DataValidationError("WALK_FORWARD_MANIFEST_INVALID: fold hashes missing")
    fold_root = path / "folds"
    actual_ids = (
        {child.name for child in fold_root.iterdir() if child.is_dir()}
        if fold_root.is_dir()
        else set()
    )
    if set(expected) != actual_ids:
        raise DataValidationError(
            "WALK_FORWARD_CHILD_SET_MISMATCH: "
            f"expected={sorted(expected)} actual={sorted(actual_ids)}"
        )
    for fold_id, digest in cast(dict[str, str], expected).items():
        fold_dir = fold_root / fold_id
        _validate_root_hash(
            fold_dir / "manifest.json",
            digest,
            f"WALK_FORWARD_CHILD_MANIFEST_HASH_MISMATCH: {fold_id}",
        )
        _validate_fold_artifact(
            fold_dir,
            expected_identity=identity,
            expected_fold_id=fold_id,
            expected_schema=int(schema_version),
            expected_evaluation_contract_version=(
                validated_evaluation_contract
                if schema_version == SCHEMA_VERSION
                else EVALUATION_CONTRACT_VERSION
            ),
            expected_lineage={
                "feature_set_id": manifest.get("feature_set_id"),
                "feature_set_hash": manifest.get("feature_set_hash"),
                "feature_provenance_hash": manifest.get("feature_provenance_hash"),
                "walk_forward_plan_hash": manifest.get("walk_forward_plan_hash"),
                "horizon_plan_hash": manifest.get("horizon_plan_hash"),
            },
        )
    return manifest


def _validate_fold_artifact(
    path: Path,
    *,
    expected_identity: str,
    expected_fold_id: str,
    expected_schema: int = SCHEMA_VERSION,
    expected_evaluation_contract_version: int = EVALUATION_CONTRACT_VERSION,
    expected_lineage: JsonObject | None = None,
) -> JsonObject:
    manifest = _load_json(path / "manifest.json", "fold manifest")
    raw_fold = manifest.get("fold")
    if (
        manifest.get("schema_version") != expected_schema
        or manifest.get("artifact_name") != "walk_forward_fold_evidence"
        or manifest.get("technical_status") != "VALID"
        or manifest.get("experiment_identity") != expected_identity
        or not isinstance(raw_fold, dict)
        or raw_fold.get("fold_id") != expected_fold_id
    ):
        raise DataValidationError(f"WALK_FORWARD_FOLD_MANIFEST_INVALID: {expected_fold_id}")
    if expected_schema == SCHEMA_VERSION and (
        manifest.get("evaluation_contract_version") != expected_evaluation_contract_version
        or not isinstance(manifest.get("modeling_identity"), str)
        or not isinstance(manifest.get("execution_identity"), str)
        or not isinstance(manifest.get("execution_contract"), dict)
        or not isinstance(manifest.get("research_validity"), dict)
    ):
        raise DataValidationError(f"WALK_FORWARD_EXECUTION_CONTRACT_INVALID: {expected_fold_id}")
    hashes = manifest.get("artifact_hashes")
    if expected_lineage is not None and any(
        not isinstance(value, str) or not value or manifest.get(key) != value
        for key, value in expected_lineage.items()
    ):
        raise DataValidationError(f"WALK_FORWARD_FEATURE_PROVENANCE_MISMATCH: {expected_fold_id}")
    if not isinstance(hashes, dict) or set(hashes) != REQUIRED_FOLD_ARTIFACTS:
        raise DataValidationError(f"WALK_FORWARD_FOLD_ARTIFACT_SET_MISMATCH: {expected_fold_id}")
    for name, digest in cast(dict[str, str], hashes).items():
        _validate_root_hash(
            path / name,
            digest,
            f"WALK_FORWARD_CHILD_ARTIFACT_HASH_MISMATCH: {expected_fold_id}/{name}",
        )
    if {child.name for child in path.iterdir()} != REQUIRED_FOLD_ARTIFACTS | {"manifest.json"}:
        raise DataValidationError(f"WALK_FORWARD_FOLD_ARTIFACT_SET_MISMATCH: {expected_fold_id}")
    return manifest


def _validate_root_hash(path: Path, expected: object, reason: str) -> None:
    if not isinstance(expected, str) or not path.is_file() or _file_hash(path) != expected:
        raise DataValidationError(reason)


def _load_json(path: Path, description: str) -> JsonObject:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"cannot read {description}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"{description} must be an object: {path}")
    return payload


def _required_int(payload: JsonObject, name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"{name} must be an integer")
    return value


def _file_hash(path: Path) -> str:
    if not path.is_file():
        raise DataValidationError(f"referenced artifact is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_hash(payload: JsonObject) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()
