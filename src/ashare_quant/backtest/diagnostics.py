"""Read-only diagnostics for one immutable historical champion backtest."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import numpy as np
import pandas as pd

from ashare_quant.backtest.data import load_model_and_features
from ashare_quant.backtest.diagnostic_attribution import (
    load_attribution_sample,
    model_feature_importance,
    shap_importance,
    single_factor_group_returns,
)
from ashare_quant.backtest.diagnostic_metrics import (
    assign_score_layers,
    daily_layer_returns,
    daily_prediction_ic,
    monthly_stability,
    summarize_ic,
    summarize_score_layers,
)
from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.utils.manifest import atomic_write_json, config_hash, current_git_info

type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class BacktestDiagnosticResult:
    """Published diagnostics for one historical backtest run."""

    run_id: str
    output_dir: Path
    prediction_rows: int
    labelled_rows: int
    ic_days: int


class BacktestDiagnosticEngine:
    """Measure frozen ranking alpha using realized labels strictly after scoring."""

    def __init__(
        self,
        *,
        processed_root: Path,
        backtest_root: Path,
        output_root: Path,
        settings: AppSettings,
        config_path: Path,
    ) -> None:
        self.processed_root = processed_root
        self.backtest_root = backtest_root
        self.output_root = output_root
        self.settings = settings
        self.config_path = config_path

    def run(self, run_id: str) -> BacktestDiagnosticResult:
        """Diagnose one run without fitting models or modifying source artifacts."""

        source_dir = _safe_run_directory(self.backtest_root, run_id)
        source_manifest = _load_source_manifest(source_dir / "manifest.json", run_id)
        prediction_path = source_dir / "predictions.parquet"
        if not prediction_path.exists():
            raise DataValidationError(
                "historical run lacks predictions.parquet; rerun `backtest historical` with "
                "the current code before diagnostics"
            )
        horizon = self.settings.backtest.diagnostics.horizon
        if horizon != int(source_manifest["backtest_config"]["historical"]["holding_period_days"]):
            raise DataValidationError(
                "diagnostic horizon differs from the historical run holding period: "
                f"diagnostic={horizon} run="
                f"{source_manifest['backtest_config']['historical']['holding_period_days']}"
            )
        evaluation, prediction_rows = _load_evaluation(
            prediction_path,
            self.processed_root / "labels_forward" / "**" / "*.parquet",
            horizon,
        )
        settings = self.settings.backtest.diagnostics
        layered = assign_score_layers(evaluation, settings.score_layers, settings.bottom_fraction)
        layer_daily = daily_layer_returns(layered)
        score_summary = summarize_score_layers(
            layer_daily,
            horizon=horizon,
            annualization_days=self.settings.backtest.annualization_days,
        )
        daily_ic = daily_prediction_ic(evaluation, settings.minimum_cross_section)
        ic_summary = summarize_ic(daily_ic)
        monthly = monthly_stability(layer_daily, daily_ic)

        model_path = Path(str(source_manifest["model_artifact"]))
        model, feature_names, artifact_hash = load_model_and_features(model_path)
        expected_hash = str(source_manifest["feature_hash"])
        if artifact_hash != expected_hash or feature_list_hash(feature_names) != expected_hash:
            raise DataValidationError("historical run and model artifact feature hashes differ")
        sample = load_attribution_sample(
            self.processed_root,
            prediction_path,
            feature_names,
            settings.shap_sample_rows,
        )
        shap_rows, shap_method, score_error = shap_importance(
            model,
            sample,
            feature_names,
            prediction_tolerance=settings.prediction_tolerance,
        )
        model_importance = model_feature_importance(model, feature_names)
        labels_path = self.processed_root / "labels_forward" / "**" / "*.parquet"
        factor_groups = single_factor_group_returns(
            self.processed_root,
            prediction_path,
            labels_path,
            feature_names,
            horizon=horizon,
            quantiles=settings.factor_quantiles,
        )
        labelled_rows = int(evaluation["future_excess_ret"].notna().sum())
        coverage = {
            "prediction_rows": prediction_rows,
            "labelled_rows": labelled_rows,
            "label_coverage": labelled_rows / prediction_rows,
            "dates": int(evaluation["trade_date"].nunique()),
        }
        summary = {
            "schema_version": 1,
            "artifact_name": "historical_backtest_diagnostics",
            "run_id": run_id,
            "model_id": source_manifest["model_id"],
            "feature_hash": expected_hash,
            "feature_count": len(feature_names),
            "horizon": horizon,
            "coverage": coverage,
            "score_layers": score_summary,
            "daily_rank_ic": ic_summary,
            "monthly_stability_scope": "all_score_layers",
            "factor_attribution": {
                "model_importance": model_importance,
                "shap_importance": shap_rows,
                "shap_method": shap_method,
                "shap_sample_rows": len(sample),
                "prediction_reproduction_max_abs_error": score_error,
                "single_factor_quantiles": settings.factor_quantiles,
            },
            "scientific_scope": {
                "labels_used_for_training": False,
                "labels_used_for_ranking": False,
                "future_returns_used_only_for_post_hoc_evaluation": True,
                "score_layer_return_type": "overlapping_forward_excess_return_cohorts",
                "cumulative_return_method": "median_of_non_overlapping_horizon_vintages",
            },
        }
        manifest = self._manifest(
            run_id,
            source_dir,
            source_manifest,
            expected_hash,
            feature_names,
            coverage,
            shap_method,
        )
        output_dir = self.output_root / run_id
        _publish(output_dir, summary, manifest, daily_ic, layer_daily, monthly, factor_groups)
        ic_days = ic_summary["days"]
        assert isinstance(ic_days, int)
        return BacktestDiagnosticResult(
            run_id=run_id,
            output_dir=output_dir,
            prediction_rows=prediction_rows,
            labelled_rows=labelled_rows,
            ic_days=ic_days,
        )

    def _manifest(
        self,
        run_id: str,
        source_dir: Path,
        source_manifest: dict[str, Any],
        feature_hash: str,
        feature_names: tuple[str, ...],
        coverage: dict[str, Any],
        shap_method: str,
    ) -> dict[str, Any]:
        git = current_git_info()
        return {
            "schema_version": 1,
            "artifact_name": "historical_backtest_diagnostics_manifest",
            "run_id": run_id,
            "source_backtest_path": str(source_dir),
            "source_backtest_manifest": source_manifest,
            "model_id": source_manifest["model_id"],
            "feature_hash": feature_hash,
            "feature_count": len(feature_names),
            "horizon": self.settings.backtest.diagnostics.horizon,
            "diagnostic_config": self.settings.backtest.diagnostics.model_dump(mode="json"),
            "coverage": coverage,
            "shap_method": shap_method,
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "config_path": str(self.config_path),
            "config_hash": config_hash(self.config_path),
            "completed_at": datetime.now(UTC).isoformat(),
        }


def _safe_run_directory(root: Path, run_id: str) -> Path:
    if not run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in run_id
    ):
        raise DataValidationError(f"invalid historical backtest run_id: {run_id}")
    resolved_root = root.resolve()
    resolved = (root / run_id).resolve()
    if resolved.parent != resolved_root or not resolved.is_dir():
        raise DataValidationError(f"historical backtest run does not exist: {run_id}")
    return resolved


def _load_source_manifest(path: Path, run_id: str) -> dict[str, Any]:
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DataValidationError(f"invalid historical backtest manifest: {error}") from error
    required = {"model_id", "model_artifact", "feature_hash", "backtest_config"}
    missing = sorted(required - payload.keys())
    if missing:
        raise DataValidationError(f"historical backtest manifest missing fields: {missing}")
    if payload.get("artifact_name") != "historical_champion_backtest":
        raise DataValidationError(f"run_id={run_id} is not a historical champion backtest")
    if payload.get("out_of_sample") is not True:
        raise DataValidationError(f"run_id={run_id} is not marked out-of-sample")
    return cast(dict[str, Any], payload)


def _load_evaluation(
    prediction_path: Path,
    labels_path: Path,
    horizon: int,
) -> tuple[DataFrame, int]:
    query = f"""
        WITH predictions AS (
            SELECT *, COUNT(*) OVER (PARTITION BY trade_date) AS cross_section_size
            FROM read_parquet('{prediction_path.as_posix()}', hive_partitioning=false)
        )
        SELECT CAST(p.trade_date AS VARCHAR) AS trade_date,
               CAST(p.ts_code AS VARCHAR) AS ts_code,
               CAST(p.prediction_score AS DOUBLE) AS prediction_score,
               CAST(p.rank AS BIGINT) AS rank,
               CAST(p.cross_section_size AS BIGINT) AS cross_section_size,
               CAST(l.future_excess_ret AS DOUBLE) AS future_excess_ret
        FROM predictions AS p
        LEFT JOIN read_parquet('{labels_path.as_posix()}', hive_partitioning=false) AS l
          ON CAST(p.trade_date AS VARCHAR) = CAST(l.trade_date AS VARCHAR)
         AND CAST(p.ts_code AS VARCHAR) = CAST(l.ts_code AS VARCHAR)
         AND CAST(l.horizon AS INTEGER) = ?
         AND CAST(l.is_label_available AS BOOLEAN)
        ORDER BY p.trade_date, p.rank, p.ts_code
    """  # noqa: S608 -- fixed local artifacts and parameterized horizon
    try:
        with duckdb.connect() as connection:
            prediction_row = connection.execute(
                f"SELECT COUNT(*) FROM read_parquet('{prediction_path.as_posix()}')"  # noqa: S608
            ).fetchone()
            frame = connection.execute(query, [horizon]).fetch_df()
    except duckdb.Error as error:
        raise DataValidationError(f"cannot join predictions to realized labels: {error}") from error
    prediction_rows = 0 if prediction_row is None else int(prediction_row[0])
    if prediction_rows == 0:
        raise DataValidationError("historical predictions are empty")
    if frame.empty or not frame["future_excess_ret"].notna().any():
        raise DataValidationError(f"no available future_excess_ret labels for horizon={horizon}")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise DataValidationError("diagnostic prediction/label join contains duplicate keys")
    if not np.isfinite(frame["prediction_score"]).all():
        raise DataValidationError("historical predictions contain non-finite scores")
    return frame, prediction_rows


def _publish(
    output_dir: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    daily_ic: DataFrame,
    layer_daily: DataFrame,
    monthly: DataFrame,
    factor_groups: list[dict[str, Any]],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        daily_ic.to_csv(staging / "daily_ic.csv", index=False)
        layer_daily.to_csv(staging / "score_layer_returns.csv", index=False)
        monthly.to_csv(staging / "monthly_stability.csv", index=False)
        pd.DataFrame(factor_groups).to_csv(staging / "single_factor_groups.csv", index=False)
        atomic_write_json(staging / "summary.json", summary)
        (staging / "diagnostics_report.md").write_text(_render_report(summary), encoding="utf-8")
        atomic_write_json(staging / "manifest.json", manifest)
        output_dir.mkdir(parents=True, exist_ok=True)
        filenames = (
            "daily_ic.csv",
            "score_layer_returns.csv",
            "monthly_stability.csv",
            "single_factor_groups.csv",
            "summary.json",
            "diagnostics_report.md",
            "manifest.json",
        )
        for filename in filenames:
            os.replace(staging / filename, output_dir / filename)


def _render_report(summary: dict[str, Any]) -> str:
    coverage = summary["coverage"]
    ic = summary["daily_rank_ic"]
    lines = [
        "# Historical Model Diagnostics",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Model ID: `{summary['model_id']}`",
        f"- Horizon: {summary['horizon']} trading days",
        f"- Predictions: {coverage['prediction_rows']}",
        f"- Realized labels: {coverage['labelled_rows']}",
        f"- Label coverage: {coverage['label_coverage']:.2%}",
        "",
        "## Daily Rank IC",
        "",
        f"- Mean IC: {_number(ic['mean_ic'])}",
        f"- ICIR: {_number(ic['icir'])}",
        f"- Positive IC ratio: {_percent(ic['positive_ic_ratio'])}",
        "",
        "## Score Layers",
        "",
        "| Layer | Mean 5d excess | Annual | Sharpe | Max drawdown | Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["score_layers"]:
        lines.append(
            f"| {row['layer']} | {_percent(row['mean_forward_excess_return'])} | "
            f"{_percent(row['annual_return'])} | {_number(row['sharpe'])} | "
            f"{_percent(row['max_drawdown'])} | {_percent(row['mean_label_coverage'])} |"
        )
    lines.extend(
        [
            "",
            "> Layer returns are post-hoc forward excess-return cohorts, not an executable "
            "portfolio backtest. Cumulative statistics use non-overlapping horizon vintages.",
            "",
            "## Factor Attribution",
            "",
            f"- Features: {summary['feature_count']}",
            f"- SHAP method: `{summary['factor_attribution']['shap_method']}`",
            f"- SHAP sample rows: {summary['factor_attribution']['shap_sample_rows']}",
            "- Labels are never used for model fitting, scoring, ranking, or feature attribution.",
        ]
    )
    return "\n".join(lines) + "\n"


def _number(value: object) -> str:
    return "NA" if value is None else f"{float(value):.6f}"  # type: ignore[arg-type]


def _percent(value: object) -> str:
    return "NA" if value is None else f"{float(value):.2%}"  # type: ignore[arg-type]
