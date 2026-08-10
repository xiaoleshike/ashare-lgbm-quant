"""Immutable multi-fold Ranker evaluation and read-only recovery inspection."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from ashare_quant.backtest.data import load_benchmark, load_calendar, load_execution_prices
from ashare_quant.backtest.engine import BacktestInputs, simulate_portfolio
from ashare_quant.backtest.executable_validation import REQUIRED_TOP_N, _signals
from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.compute import resolve_training_backend
from ashare_quant.models.feature_provenance import validate_governed_feature_set
from ashare_quant.models.horizon_experiments import dataset_fingerprint
from ashare_quant.models.ranker import feature_importance, fit_ranker, ranker_semantic_parameters
from ashare_quant.models.ranker_data import RankerDataLoader, RankerDataset
from ashare_quant.models.ranker_metrics import evaluate_ranker
from ashare_quant.models.research_policy import enforce_research_window, load_research_policy
from ashare_quant.models.temporal_isolation import required_temporal_gap_sessions
from ashare_quant.utils.manifest import atomic_write_json

SCHEMA_VERSION = 1
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

    def execute(
        self,
        *,
        fold: JsonObject,
        horizon: int,
        features: tuple[str, ...],
        require_executable: bool,
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

    def execute(
        self,
        *,
        fold: JsonObject,
        horizon: int,
        features: tuple[str, ...],
        require_executable: bool,
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
        evaluation = loader.load(
            str(fold["evaluation_start"]),
            str(fold["evaluation_end"]),
            features,
            self.settings.ranker.relevance_grades,
        )
        runtime = resolve_training_backend(self.settings.ranker.training_backend)
        model = fit_ranker(train, validation, self.settings.ranker, runtime)
        validation_predictions = np.asarray(model.predict(validation.features), dtype=float)
        evaluation_predictions = np.asarray(model.predict(evaluation.features), dtype=float)
        predictions = evaluation.frame.loc[:, ["trade_date", "ts_code"]].copy()
        predictions["score"] = evaluation_predictions
        ranking = _ranking_metrics(evaluation, evaluation_predictions, self.settings)
        executable = (
            self._executable_metrics(predictions, horizon)
            if require_executable
            else {"status": "NOT_REQUIRED", "accounting_schema_version": 2}
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

    def _executable_metrics(self, predictions: pd.DataFrame, horizon: int) -> JsonObject:
        dates = tuple(sorted(predictions["trade_date"].astype(str).unique()))
        execution = self.settings.backtest.model_copy(
            update={
                "execution": "next_open",
                "holding_period_days": horizon,
                "top_n": REQUIRED_TOP_N,
            }
        )
        calendar = load_calendar(
            self.raw_root, dates[0], dates[-1], horizon + execution.sell_delay_max_days
        )
        prices = load_execution_prices(
            self.raw_root,
            self.processed_root,
            calendar[0],
            calendar[-1],
            self.settings.universe.price_tolerance,
        )
        benchmark = load_benchmark(
            self.raw_root, execution.benchmark_index_code, calendar[0], calendar[-1]
        )
        inputs = BacktestInputs(
            signals=_signals(predictions),
            prices=prices,
            calendar=tuple(calendar),
            benchmark=benchmark,
        )
        results = tuple(
            simulate_portfolio(
                inputs, top_n=top_n, settings=execution, purpose="executable_validation"
            )
            for top_n in REQUIRED_TOP_N
        )
        if any(
            not result.holdings.empty
            and result.holdings["trade_date"].astype(str).eq(calendar[-1]).any()
            for result in results
        ):
            raise DataValidationError("walk-forward fold has unresolved executable positions")
        return {
            "status": "COMPLETE",
            "accounting_schema_version": 2,
            "top_n": {str(result.top_n): result.metrics for result in results},
            "accounting_summaries": {
                str(result.top_n): result.accounting_summary for result in results
            },
            "cost_policy_hash": str(results[0].cost_policy["cost_policy_hash"]),
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
    ) -> None:
        self.reports_root = reports_root
        self.settings = settings
        self.executor = executor
        self.research_policy_path = research_policy_path

    def run(
        self,
        *,
        experiment_manifest: Path,
        experiment_id: str,
        feature_provenance_path: Path,
        require_executable: bool = True,
    ) -> WalkForwardEvaluationResult:
        plan = _load_json(experiment_manifest, "horizon experiment manifest")
        experiment = _select_experiment(plan, experiment_id)
        horizon = _required_int(experiment, "horizon")
        policy = load_research_policy(self.research_policy_path)
        if (
            plan.get("feature_hash")
            != validate_governed_feature_set(feature_provenance_path).feature_list_hash
        ):
            raise DataValidationError("walk-forward feature-set hash differs from experiment plan")
        provenance = validate_governed_feature_set(feature_provenance_path)
        current_source_identity = self.executor.validate_sources(plan)
        folds = _load_eligible_folds(plan, experiment, horizon)
        for fold in folds:
            enforce_research_window(
                policy,
                consumer="walk_forward_evaluation",
                start_date=str(fold["evaluation_start"]),
                end_date=str(fold["evaluation_end"]),
            )
        identity = _experiment_identity(
            plan=plan,
            experiment=experiment,
            fold_ids=tuple(str(fold["fold_id"]) for fold in folds),
            feature_set_id=provenance.feature_set_id,
            research_policy_hash=policy.policy_hash,
            semantic_parameters=ranker_semantic_parameters(self.settings.ranker),
            source_identity=current_source_identity,
            require_executable=require_executable,
        )
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
                )
                _publish_fold(
                    fold_dir,
                    fold=fold,
                    experiment_identity=identity,
                    horizon=horizon,
                    feature_set_id=provenance.feature_set_id,
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
            run_id=run_id,
            plan_path=str(experiment_manifest),
            experiment=experiment,
            feature_provenance_path=str(feature_provenance_path),
            feature_set_id=provenance.feature_set_id,
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
                manifest = _load_json(manifest_path, "walk-forward manifest")
                expected = manifest.get("fold_manifest_hashes")
                actual = {
                    path.parent.name: _file_hash(path)
                    for path in sorted((root / "folds").glob("*/manifest.json"))
                }
                if expected != actual:
                    issues.append("aggregate child manifest set does not match fold artifacts")
            except DataValidationError as error:
                issues.append(str(error))
        return {"status": "CLEAN" if not issues else "ACTION_REQUIRED", "issues": issues}


def walk_forward_status(reports_root: Path, run_id: str) -> JsonObject:
    """Return a validated immutable experiment status snapshot."""

    root = reports_root / "research" / "walk_forward" / run_id
    manifest = _load_json(root / "manifest.json", "walk-forward manifest")
    if manifest.get("artifact_name") != "multi_fold_walk_forward_evidence":
        raise DataValidationError("invalid multi-fold walk-forward artifact")
    expected = manifest.get("fold_manifest_hashes")
    actual = {
        path.parent.name: _file_hash(path)
        for path in sorted((root / "folds").glob("*/manifest.json"))
    }
    if expected != actual:
        raise DataValidationError("walk-forward child manifest hashes changed")
    return {
        "run_id": run_id,
        "status": manifest.get("status"),
        "fold_count": len(actual),
        "research_policy_hash": manifest.get("research_policy_hash"),
        "feature_set_id": manifest.get("feature_set_id"),
    }


def _ranking_metrics(
    dataset: RankerDataset, predictions: np.ndarray, settings: AppSettings
) -> JsonObject:
    base = cast(
        JsonObject,
        evaluate_ranker(
            dataset, predictions, settings.ranker.ndcg_at, settings.ranker.portfolio_fractions
        ),
    )
    frame = dataset.frame.loc[:, ["trade_date", "future_excess_ret_5d"]].copy()
    frame["score"] = predictions
    daily_values = [
        group["score"].corr(group["future_excess_ret_5d"], method="spearman")
        for _, group in frame.groupby("trade_date", sort=True)
    ]
    values = pd.to_numeric(pd.Series(daily_values, dtype="float64"), errors="coerce").dropna()
    base.update(
        {
            "rank_ic_median": float(values.median()),
            "rank_ic_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "positive_rank_ic_ratio": float((values > 0).mean()),
            "coverage": float(len(dataset.frame) / max(1, len(dataset.frame))),
            "signal_dates": int(dataset.frame["trade_date"].nunique()),
            "securities_scored": int(dataset.frame["ts_code"].nunique()),
        }
    )
    return base


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


def _experiment_identity(
    *,
    plan: JsonObject,
    experiment: JsonObject,
    fold_ids: tuple[str, ...],
    feature_set_id: str,
    research_policy_hash: str,
    semantic_parameters: JsonObject,
    source_identity: JsonObject,
    require_executable: bool,
) -> str:
    stable = {
        "schema_version": SCHEMA_VERSION,
        "plan_identity_hash": plan.get("plan_identity_hash"),
        "experiment_id": experiment.get("experiment_id"),
        "fold_ids": fold_ids,
        "feature_set_id": feature_set_id,
        "research_policy_hash": research_policy_hash,
        "semantic_parameters": semantic_parameters,
        "source_identity": source_identity,
        "require_executable": require_executable,
    }
    return _payload_hash(stable)


def _publish_fold(
    path: Path,
    *,
    fold: JsonObject,
    experiment_identity: str,
    horizon: int,
    feature_set_id: str,
    feature_hash: str,
    research_policy_hash: str,
    semantic_parameters: JsonObject,
    source_identity: JsonObject,
    result: FoldExecutionResult,
) -> None:
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
            "horizon": horizon,
            "feature_set_id": feature_set_id,
            "feature_hash": feature_hash,
            "research_policy_hash": research_policy_hash,
            "semantic_parameters": semantic_parameters,
            "source_identity": source_identity,
            "training_compute": result.training_compute,
            "artifact_hashes": files,
        }
        atomic_write_json(staging / "manifest.json", manifest)
        if path.exists():
            raise DataValidationError(f"immutable fold artifact already exists: {path}")
        staging.rename(path)


def _existing_fold(path: Path, fold: JsonObject, identity: str) -> JsonObject | None:
    if not path.exists():
        return None
    manifest = _load_json(path / "manifest.json", "fold manifest")
    raw_fold = manifest.get("fold")
    if (
        manifest.get("experiment_identity") != identity
        or not isinstance(raw_fold, dict)
        or raw_fold.get("fold_id") != fold.get("fold_id")
    ):
        raise DataValidationError(f"fold identity conflict: {path}")
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict) or any(
        _file_hash(path / name) != digest for name, digest in hashes.items()
    ):
        raise DataValidationError(f"fold artifact hash mismatch: {path}")
    return manifest


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
    run_id: str,
    plan_path: str,
    experiment: JsonObject,
    feature_provenance_path: str,
    feature_set_id: str,
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
    aggregate["performance"] = _metric_distributions(ranking_rows)
    aggregate["executable_performance"] = _executable_distributions(executable_rows)
    aggregate["feature_importance_stability"] = _importance_stability(importance_rows)
    atomic_write_json(output_dir / "aggregate_metrics.json", aggregate)
    pd.DataFrame(
        [
            {
                "fold_id": fold_id,
                **{key: value for key, value in row.items() if isinstance(value, (int, float))},
            }
            for fold_id, row in zip(fold_hashes, ranking_rows, strict=True)
        ]
    ).to_parquet(output_dir / "fold_summary.parquet", index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_name": "multi_fold_walk_forward_evidence",
        "status": "COMPLETE",
        "identity": identity,
        "run_id": run_id,
        "plan_path": plan_path,
        "experiment": experiment,
        "feature_provenance_path": feature_provenance_path,
        "feature_set_id": feature_set_id,
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
    manifest = _load_json(path / "manifest.json", "walk-forward manifest")
    if manifest.get("identity") != identity:
        raise DataValidationError(f"walk-forward identity conflict: {path}")
    expected = manifest.get("fold_manifest_hashes")
    actual = {
        child.parent.name: _file_hash(child)
        for child in sorted((path / "folds").glob("*/manifest.json"))
    }
    if expected != actual:
        raise DataValidationError("completed walk-forward child manifests changed")
    return WalkForwardEvaluationResult(
        str(manifest["run_id"]),
        str(manifest["status"]),
        len(manifest["fold_manifest_hashes"]),
        path,
    )


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
