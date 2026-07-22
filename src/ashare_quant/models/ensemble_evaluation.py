"""Immutable equal-weight multi-horizon rank ensemble evaluation."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ashare_quant.backtest.diagnostic_metrics import daily_prediction_ic, summarize_ic
from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.challenger_evaluation import (
    _load_mature_labels,
    _load_predictions,
    _validate_prediction_contract,
)
from ashare_quant.models.inference import (
    PredictionModel,
    load_registered_feature_list,
    score_registered_model_range,
)
from ashare_quant.models.registry import ModelRegistry, RegisteredModel
from ashare_quant.utils.manifest import atomic_write_json, config_hash, current_git_info

ENSEMBLE_MANIFEST_SCHEMA_VERSION = 1
REQUIRED_HORIZONS = (5, 10, 20, 60)
TOP_COUNTS = (10, 20, 50)

type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class EnsembleEvaluationResult:
    """One immutable multi-horizon ensemble evaluation."""

    run_id: str
    model_ids: tuple[str, ...]
    prediction_rows: int
    prediction_dates: int
    output_dir: Path


@dataclass(frozen=True, slots=True)
class _Component:
    horizon: int
    model: RegisteredModel
    model_manifest: dict[str, Any]
    prediction_manifest: dict[str, Any]
    predictions: DataFrame


class MultiHorizonEnsembleEngine:
    """Combine frozen horizon scores and evaluate them only after scoring is complete."""

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        processed_root: Path,
        reports_root: Path,
        settings: AppSettings,
        config_path: Path,
        model_loader: Callable[[Path], PredictionModel] | None = None,
    ) -> None:
        self.registry = registry
        self.processed_root = processed_root
        self.reports_root = reports_root
        self.settings = settings
        self.config_path = config_path
        self._model_loader = model_loader

    def evaluate(self, model_ids: Sequence[str]) -> EnsembleEvaluationResult:
        """Evaluate a fixed equal-weight rank ensemble without changing model state."""

        registry_before = _file_hash(self.registry.registry_path)
        champion = self.registry.get_champion("lightgbm_ranker")
        if champion is None:
            raise DataValidationError("no lightgbm_ranker champion is registered")
        components = self._load_components(model_ids, champion)
        ensemble, score_frame, common_dates = self._build_scores(components, champion)
        # Labels are deliberately loaded only after every score is frozen.
        metric_rows, overall = self._evaluate_scores(score_frame, components, common_dates)
        if _file_hash(self.registry.registry_path) != registry_before:
            raise DataValidationError("model registry changed during ensemble evaluation")
        identity = self._identity(champion, components, common_dates)
        run_id = f"ensemble_eval_{identity[:16]}"
        output_dir = self.reports_root / "ensemble_evaluation" / run_id
        existing = _existing_result(output_dir, identity)
        if existing is not None:
            return existing
        metrics = {
            "schema_version": 1,
            "artifact_name": "multi_horizon_ensemble_metrics",
            "target_horizons": list(REQUIRED_HORIZONS),
            "overall": overall,
            "rows": metric_rows,
        }
        manifest = self._manifest(
            identity=identity,
            run_id=run_id,
            champion=champion,
            components=components,
            common_dates=common_dates,
            ensemble=ensemble,
        )
        _publish(output_dir, ensemble, metrics, manifest)
        return EnsembleEvaluationResult(
            run_id=run_id,
            model_ids=tuple(component.model.model_id for component in components),
            prediction_rows=len(ensemble),
            prediction_dates=len(common_dates),
            output_dir=output_dir,
        )

    def _load_components(
        self, model_ids: Sequence[str], champion: RegisteredModel
    ) -> tuple[_Component, ...]:
        if len(model_ids) != len(REQUIRED_HORIZONS) or len(set(model_ids)) != len(model_ids):
            raise DataValidationError("ensemble requires four unique challenger model IDs")
        champion_features, champion_hash = load_registered_feature_list(
            Path(champion.artifact_path), champion
        )
        components: list[_Component] = []
        for model_id in model_ids:
            model = _find_model(self.registry, model_id)
            if model.status != "candidate":
                raise DataValidationError(f"ensemble model must be a candidate: {model_id}")
            features, feature_hash = load_registered_feature_list(Path(model.artifact_path), model)
            if feature_hash != champion_hash or features != champion_features:
                raise DataValidationError("ensemble model feature hash differs from champion")
            model_manifest = _load_json(
                Path(model.artifact_path) / "manifest.json", "challenger model manifest"
            )
            prediction_dir = self.reports_root / "challenger_predictions" / model_id
            prediction_manifest = _load_json(
                prediction_dir / "manifest.json", "challenger prediction manifest"
            )
            horizon, _ = _validate_prediction_contract(
                prediction_manifest, model, processed_root=self.processed_root
            )
            if prediction_manifest.get("input_manifests", {}).get("model") != _file_hash(
                Path(model.artifact_path) / "manifest.json"
            ):
                raise DataValidationError("challenger prediction model version has changed")
            predictions = _load_predictions(
                prediction_dir / "predictions.parquet", model_id, prediction_manifest
            )
            components.append(
                _Component(horizon, model, model_manifest, prediction_manifest, predictions)
            )
        components.sort(key=lambda item: item.horizon)
        if tuple(component.horizon for component in components) != REQUIRED_HORIZONS:
            raise DataValidationError(
                f"ensemble horizons must be exactly {list(REQUIRED_HORIZONS)}"
            )
        _require_common_model_contract(components)
        return tuple(components)

    def _build_scores(
        self,
        components: tuple[_Component, ...],
        champion: RegisteredModel,
    ) -> tuple[DataFrame, DataFrame, tuple[str, ...]]:
        date_sets = [
            set(component.predictions["trade_date"].astype(str)) for component in components
        ]
        common_dates = tuple(sorted(set.intersection(*date_sets)))
        if not common_dates:
            raise DataValidationError("ensemble predictions have no common trading dates")
        filtered = {
            component.horizon: component.predictions.loc[
                component.predictions["trade_date"].astype(str).isin(common_dates),
                ["trade_date", "ts_code", "prediction_score"],
            ].reset_index(drop=True)
            for component in components
        }
        _require_common_prediction_keys(filtered)
        ensemble = build_rank_percentile_ensemble(filtered)
        champion_batch = score_registered_model_range(
            champion,
            processed_root=self.processed_root,
            start_date=common_dates[0],
            end_date=common_dates[-1],
            allowed_ranges=((common_dates[0], common_dates[-1]),),
            model_loader=self._model_loader,
        )
        champion_predictions = champion_batch.predictions.loc[
            champion_batch.predictions["trade_date"].astype(str).isin(common_dates),
            ["trade_date", "ts_code", "prediction_score"],
        ].reset_index(drop=True)
        _require_same_keys(ensemble, champion_predictions, "champion")
        scores = ensemble[["trade_date", "ts_code", "ensemble_score"]].copy()
        scores["champion_score"] = _aligned_scores(ensemble, champion_predictions)
        for component in components:
            scores[f"h{component.horizon}_score"] = _aligned_scores(
                ensemble, filtered[component.horizon]
            )
        return ensemble, scores, common_dates

    def _evaluate_scores(
        self,
        scores: DataFrame,
        components: tuple[_Component, ...],
        common_dates: tuple[str, ...],
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
        rows: list[dict[str, Any]] = []
        overall: dict[str, dict[str, dict[str, Any]]] = {}
        score_columns = {
            "champion": "champion_score",
            **{f"h{component.horizon}": f"h{component.horizon}_score" for component in components},
            "ensemble": "ensemble_score",
        }
        maximum_mature = {
            component.horizon: str(component.prediction_manifest["maximum_mature_evaluation_date"])
            for component in components
        }
        for horizon in REQUIRED_HORIZONS:
            labels = _load_mature_labels(
                self.processed_root,
                horizon,
                dates=common_dates,
                maximum_mature_date=maximum_mature[horizon],
            )
            comparison = scores.merge(
                labels, on=["trade_date", "ts_code"], how="left", validate="one_to_one"
            )
            target_rows, target_overall = _evaluate_target(
                comparison,
                target_horizon=horizon,
                score_columns=score_columns,
                minimum_cross_section=self.settings.models.challenger_evaluation.minimum_cross_section,
                regime_threshold=self.settings.models.challenger_evaluation.regime_return_threshold,
            )
            rows.extend(target_rows)
            overall[str(horizon)] = target_overall
        return rows, overall

    def _identity(
        self,
        champion: RegisteredModel,
        components: tuple[_Component, ...],
        common_dates: tuple[str, ...],
    ) -> str:
        git = current_git_info()
        return _payload_hash(
            {
                "schema_version": ENSEMBLE_MANIFEST_SCHEMA_VERSION,
                "champion_model_id": champion.model_id,
                "champion_manifest_hash": _file_hash(
                    Path(champion.artifact_path) / "manifest.json"
                ),
                "components": {
                    str(component.horizon): {
                        "model_id": component.model.model_id,
                        "prediction_identity": component.prediction_manifest.get(
                            "prediction_identity"
                        ),
                    }
                    for component in components
                },
                "common_dates": [common_dates[0], common_dates[-1], len(common_dates)],
                "labels_manifest_hash": _file_hash(
                    self.processed_root / "labels_forward" / "_manifest.json"
                ),
                "config_hash": config_hash(self.config_path),
                "git_commit": git["commit"],
                "method": "daily_equal_weight_rank_percentile",
            }
        )

    def _manifest(
        self,
        *,
        identity: str,
        run_id: str,
        champion: RegisteredModel,
        components: tuple[_Component, ...],
        common_dates: tuple[str, ...],
        ensemble: DataFrame,
    ) -> dict[str, Any]:
        git = current_git_info()
        return {
            "schema_version": ENSEMBLE_MANIFEST_SCHEMA_VERSION,
            "artifact_name": "multi_horizon_ensemble_evaluation_manifest",
            "ensemble_identity": identity,
            "run_id": run_id,
            "champion_model_id": champion.model_id,
            "component_models": {
                str(component.horizon): component.model.model_id for component in components
            },
            "feature_hash": components[0].prediction_manifest["feature_hash"],
            "universe_hash": components[0].prediction_manifest["universe_hash"],
            "method": "daily_equal_weight_rank_percentile",
            "weights": {str(horizon): 0.25 for horizon in REQUIRED_HORIZONS},
            "target_horizons": list(REQUIRED_HORIZONS),
            "top_counts": list(TOP_COUNTS),
            "minimum_date": common_dates[0],
            "maximum_date": common_dates[-1],
            "prediction_dates": len(common_dates),
            "prediction_rows": len(ensemble),
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "config_path": str(self.config_path),
            "config_hash": config_hash(self.config_path),
            "input_manifests": {
                "champion_model": _file_hash(Path(champion.artifact_path) / "manifest.json"),
                "challenger_predictions": {
                    str(component.horizon): {
                        "path": str(
                            (
                                self.reports_root
                                / "challenger_predictions"
                                / component.model.model_id
                                / "manifest.json"
                            ).resolve()
                        ),
                        "sha256": _file_hash(
                            self.reports_root
                            / "challenger_predictions"
                            / component.model.model_id
                            / "manifest.json"
                        ),
                    }
                    for component in components
                },
                "labels_forward": _file_hash(
                    self.processed_root / "labels_forward" / "_manifest.json"
                ),
                "features_daily": _file_hash(
                    self.processed_root / "features_daily" / "_manifest.json"
                ),
                "universe_daily": _file_hash(
                    self.processed_root / "universe_daily" / "_manifest.json"
                ),
            },
            "isolation_contract": {
                "raw_scores_averaged": False,
                "labels_loaded_after_scoring": True,
                "labels_used_only_post_hoc": True,
                "same_dates": True,
                "same_universe_rows": True,
                "automatic_weights": False,
                "automatic_promotion": False,
                "registry_modified": False,
                "models_modified": False,
            },
        }


def build_rank_percentile_ensemble(predictions: Mapping[int, DataFrame]) -> DataFrame:
    """Build deterministic daily percentile scores without labels or raw-score averaging."""

    if tuple(sorted(predictions)) != REQUIRED_HORIZONS:
        raise DataValidationError(f"ensemble horizons must be exactly {list(REQUIRED_HORIZONS)}")
    _require_common_prediction_keys(predictions)
    keys = ["trade_date", "ts_code"]
    first = predictions[REQUIRED_HORIZONS[0]].sort_values(keys, kind="mergesort")
    result = first[keys].reset_index(drop=True)
    percentile_columns: list[str] = []
    for horizon in REQUIRED_HORIZONS:
        aligned = predictions[horizon].sort_values(keys, kind="mergesort").reset_index(drop=True)
        scores = pd.to_numeric(aligned["prediction_score"], errors="coerce")
        if not np.isfinite(scores).all():
            raise DataValidationError(f"h{horizon} predictions contain non-finite scores")
        column = f"h{horizon}_rank_percentile"
        result[column] = scores.groupby(aligned["trade_date"], sort=False).rank(
            method="average", pct=True, ascending=True
        )
        percentile_columns.append(column)
    result["ensemble_score"] = result[percentile_columns].mean(axis=1)
    result = result.sort_values(
        ["trade_date", "ensemble_score", "ts_code"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result["rank"] = result.groupby("trade_date", sort=False).cumcount() + 1
    return result[[*keys, "ensemble_score", "rank", *percentile_columns]]


def _evaluate_target(
    comparison: DataFrame,
    *,
    target_horizon: int,
    score_columns: Mapping[str, str],
    minimum_cross_section: int,
    regime_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    benchmark = (
        comparison.groupby("trade_date", sort=True)["benchmark_forward_ret"]
        .median()
        .rename("benchmark_forward_ret")
        .reset_index()
    )
    benchmark["regime"] = np.select(
        [
            benchmark["benchmark_forward_ret"] > regime_threshold,
            benchmark["benchmark_forward_ret"] < -regime_threshold,
        ],
        ["bull", "bear"],
        default="neutral",
    )
    dates = sorted(comparison["trade_date"].astype(str).unique())
    periods: list[tuple[str, str, set[str]]] = [("overall", "all", set(dates))]
    periods.extend(
        ("year", year, {date for date in dates if date.startswith(year)})
        for year in sorted({date[:4] for date in dates})
    )
    periods.extend(
        (
            "regime",
            regime,
            set(benchmark.loc[benchmark["regime"] == regime, "trade_date"].astype(str)),
        )
        for regime in ("bull", "bear", "neutral")
    )
    rows: list[dict[str, Any]] = []
    overall: dict[str, dict[str, Any]] = {}
    for role, score_column in score_columns.items():
        evaluation = comparison[
            ["trade_date", "ts_code", "future_excess_ret", "benchmark_forward_ret"]
        ].copy()
        evaluation["prediction_score"] = comparison[score_column]
        evaluation = evaluation.sort_values(
            ["trade_date", "prediction_score", "ts_code"],
            ascending=[True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        evaluation["rank"] = evaluation.groupby("trade_date", sort=False).cumcount() + 1
        daily_ic = daily_prediction_ic(evaluation, minimum_cross_section)
        for period_type, period, period_dates in periods:
            if not period_dates:
                continue
            subset = evaluation.loc[evaluation["trade_date"].astype(str).isin(period_dates)]
            ic_subset = daily_ic.loc[daily_ic["date"].astype(str).isin(period_dates)]
            row = _metric_row(target_horizon, role, period_type, period, subset, ic_subset)
            rows.append(row)
            if period_type == "overall":
                overall[role] = row
    return rows, overall


def _metric_row(
    target_horizon: int,
    role: str,
    period_type: str,
    period: str,
    evaluation: DataFrame,
    daily_ic: DataFrame,
) -> dict[str, Any]:
    ic = summarize_ic(daily_ic)
    row: dict[str, Any] = {
        "target_horizon": target_horizon,
        "model_role": role,
        "period_type": period_type,
        "period": period,
        "days": ic["days"],
        "rank_ic": ic["mean_ic"],
        "icir": ic["icir"],
        "positive_ic_ratio": ic["positive_ic_ratio"],
    }
    for count in TOP_COUNTS:
        selected = evaluation.loc[
            (evaluation["rank"] <= count) & evaluation["future_excess_ret"].notna()
        ]
        daily = selected.groupby("trade_date", sort=True)["future_excess_ret"].mean()
        row[f"top_{count}_mean_excess_return"] = None if daily.empty else float(daily.mean())
        row[f"top_{count}_positive_ratio"] = None if daily.empty else float((daily > 0).mean())
    return row


def _require_common_model_contract(components: Sequence[_Component]) -> None:
    feature_hashes = {str(item.prediction_manifest.get("feature_hash")) for item in components}
    universe_hashes = {str(item.prediction_manifest.get("universe_hash")) for item in components}
    feature_inputs = {
        str(item.prediction_manifest.get("input_manifests", {}).get("features_daily"))
        for item in components
    }
    universe_inputs = {
        str(item.prediction_manifest.get("input_manifests", {}).get("universe_daily"))
        for item in components
    }
    source_plans = {
        str(
            item.model_manifest.get("source_manifests", {})
            .get("horizon_experiment", {})
            .get("sha256")
        )
        for item in components
    }
    execution_rules = {str(item.prediction_manifest.get("execution_rule")) for item in components}
    if len(feature_hashes) != 1 or len(feature_inputs) != 1:
        raise DataValidationError("ensemble predictions use different feature versions")
    if len(universe_hashes) != 1 or len(universe_inputs) != 1:
        raise DataValidationError("ensemble predictions use different universe versions")
    if len(source_plans) != 1 or "None" in source_plans:
        raise DataValidationError("ensemble challengers do not share one horizon experiment plan")
    if execution_rules != {"next_open"}:
        raise DataValidationError("ensemble challengers must use the same next_open execution rule")


def _require_common_prediction_keys(predictions: Mapping[int, DataFrame]) -> None:
    first_horizon = min(predictions)
    first = predictions[first_horizon]
    for horizon, frame in predictions.items():
        if frame.duplicated(["trade_date", "ts_code"]).any():
            raise DataValidationError(f"h{horizon} predictions contain duplicate keys")
        _require_same_keys(first, frame, f"h{horizon}")


def _require_same_keys(reference: DataFrame, other: DataFrame, description: str) -> None:
    keys = ["trade_date", "ts_code"]
    reference_keys = pd.MultiIndex.from_frame(reference[keys].astype(str))
    other_keys = pd.MultiIndex.from_frame(other[keys].astype(str))
    if len(reference) != len(other) or set(reference_keys) != set(other_keys):
        raise DataValidationError(f"{description} predictions use a different universe")


def _aligned_scores(reference: DataFrame, predictions: DataFrame) -> np.ndarray:
    aligned = reference[["trade_date", "ts_code"]].merge(
        predictions[["trade_date", "ts_code", "prediction_score"]],
        on=["trade_date", "ts_code"],
        validate="one_to_one",
    )
    return pd.to_numeric(aligned["prediction_score"], errors="coerce").to_numpy(dtype=float)


def _existing_result(output_dir: Path, identity: str) -> EnsembleEvaluationResult | None:
    if not output_dir.exists():
        return None
    manifest = _load_json(output_dir / "manifest.json", "ensemble evaluation manifest")
    if manifest.get("ensemble_identity") != identity:
        raise DataValidationError(f"immutable ensemble evaluation identity differs: {output_dir}")
    return EnsembleEvaluationResult(
        run_id=str(manifest["run_id"]),
        model_ids=tuple(str(value) for value in manifest["component_models"].values()),
        prediction_rows=int(manifest["prediction_rows"]),
        prediction_dates=int(manifest["prediction_dates"]),
        output_dir=output_dir,
    )


def _publish(
    output_dir: Path,
    predictions: DataFrame,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix=".ensemble-") as temporary:
        staging = Path(temporary)
        predictions.to_parquet(staging / "ensemble_predictions.parquet", index=False)
        atomic_write_json(staging / "metrics.json", metrics)
        (staging / "report.md").write_text(_render_report(metrics, manifest), encoding="utf-8")
        atomic_write_json(staging / "manifest.json", manifest)
        if output_dir.exists():
            raise DataValidationError(f"immutable ensemble evaluation already exists: {output_dir}")
        staging.rename(output_dir)


def _render_report(metrics: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [
        "# Multi-Horizon Ensemble Evaluation",
        "",
        f"- Date range: {manifest['minimum_date']} to {manifest['maximum_date']}",
        f"- Common dates: {manifest['prediction_dates']}",
        f"- Common rows: {manifest['prediction_rows']}",
        "- Method: daily equal-weight percentile rank",
        "",
    ]
    for horizon in REQUIRED_HORIZONS:
        lines.extend(
            [
                f"## {horizon}-Day Target",
                "",
                "| Score | Rank IC | ICIR | Top 10 | Top 20 | Top 50 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for role, row in metrics["overall"][str(horizon)].items():
            lines.append(
                f"| {role} | {_number(row['rank_ic'])} | {_number(row['icir'])} | "
                f"{_number(row['top_10_mean_excess_return'])} | "
                f"{_number(row['top_20_mean_excess_return'])} | "
                f"{_number(row['top_50_mean_excess_return'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "This is a post-hoc research evaluation. It does not select weights, promote models, "
            "modify the registry, or generate trading signals.",
            "",
        ]
    )
    return "\n".join(lines)


def _number(value: object) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{float(value):.6f}"


def _find_model(registry: ModelRegistry, model_id: str) -> RegisteredModel:
    try:
        return next(model for model in registry.list_models() if model.model_id == model_id)
    except StopIteration as error:
        raise DataValidationError(f"model_id is not registered: {model_id}") from error


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"{description} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"cannot read {description}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"{description} must be a JSON object: {path}")
    return payload


def _file_hash(path: Path) -> str:
    if not path.is_file():
        raise DataValidationError(f"manifest source does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()
