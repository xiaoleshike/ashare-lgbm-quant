"""Champion-model historical selection backtests with strict OOS provenance."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from ashare_quant.backtest.data import load_backtest_inputs, load_model_and_features
from ashare_quant.backtest.engine import BacktestResult, simulate_portfolio
from ashare_quant.backtest.provenance import (
    ModelEvaluationBoundary,
    require_oos_evaluation,
    resolve_model_evaluation_boundary,
)
from ashare_quant.backtest.runner import build_predictions_frame
from ashare_quant.config.settings import AppSettings, BacktestSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.registry import ModelRegistry, RegisteredModel
from ashare_quant.utils.manifest import (
    atomic_write_json,
    config_hash,
    current_git_info,
    read_manifest,
)

type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class HistoricalBacktestResult:
    """Published historical backtest artifact."""

    run_id: str
    output_dir: Path
    model_id: str
    start_date: str
    end_date: str
    metrics: dict[str, dict[str, Any]]


class HistoricalBacktestEngine:
    """Evaluate a frozen champion with same-date signals and future execution only."""

    def __init__(
        self,
        *,
        raw_root: Path,
        processed_root: Path,
        output_root: Path,
        models_root: Path,
        settings: AppSettings,
        config_path: Path,
    ) -> None:
        self.raw_root = raw_root
        self.processed_root = processed_root
        self.output_root = output_root
        self.models_root = models_root
        self.settings = settings
        self.config_path = config_path

    def run(
        self,
        *,
        period: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        top_n: tuple[int, ...] | None = None,
    ) -> HistoricalBacktestResult:
        """Run one configured or explicit chronological OOS evaluation range."""

        requested_start, requested_end = self._resolve_dates(period, start_date, end_date)
        champion = ModelRegistry(self.models_root).get_champion("lightgbm_ranker")
        if champion is None:
            raise DataValidationError("no champion is registered for model_type=lightgbm_ranker")
        boundary = resolve_model_evaluation_boundary(Path(champion.artifact_path))
        require_oos_evaluation(
            boundary,
            model_dir=Path(champion.artifact_path),
            evaluation_start=requested_start,
            evaluation_end=requested_end,
        )
        effective_end = _effective_end_date(
            self.processed_root,
            requested_start,
            requested_end,
            self.settings.backtest.historical.holding_period_days,
        )
        model, feature_names, feature_hash = load_model_and_features(Path(champion.artifact_path))
        if feature_hash != champion.feature_hash:
            raise DataValidationError("champion feature hash differs from model artifact")
        selected_top_n = top_n or self.settings.backtest.historical.top_n
        execution = self._execution_settings(selected_top_n)
        effective_settings = self.settings.model_copy(update={"backtest": execution})
        inputs = load_backtest_inputs(
            raw_root=self.raw_root,
            processed_root=self.processed_root,
            model=model,
            feature_names=feature_names,
            start_date=requested_start,
            end_date=effective_end,
            settings=effective_settings,
        )
        _validate_signal_chronology(inputs.signals, requested_start, effective_end)
        predictions = build_predictions_frame(inputs.signals, selected_top_n)
        label_audit = _audit_labels(
            self.processed_root,
            predictions,
            requested_start,
            effective_end,
            execution.holding_period_days,
        )
        results = [
            simulate_portfolio(
                inputs,
                top_n=value,
                settings=execution,
                purpose="oos_evidence",
            )
            for value in selected_top_n
        ]
        daily = pd.concat([result.daily_returns for result in results], ignore_index=True)
        holdings = pd.concat([result.holdings for result in results], ignore_index=True)
        metrics = {
            str(result.top_n): _metrics_with_years(
                result,
                execution,
                self.settings.backtest.historical.bull_annual_return_threshold,
                self.settings.backtest.historical.bear_annual_return_threshold,
            )
            for result in results
        }
        run_id = _run_id(
            champion.model_id,
            requested_start,
            effective_end,
            feature_hash,
            selected_top_n,
        )
        output_dir = self.output_root / run_id
        summary = {
            "schema_version": 2,
            "artifact_name": "historical_champion_backtest_summary",
            "run_id": run_id,
            "model_id": champion.model_id,
            "feature_hash": feature_hash,
            "requested_start_date": requested_start,
            "requested_end_date": requested_end,
            "effective_start_date": requested_start,
            "effective_end_date": effective_end,
            "top_n": list(selected_top_n),
            "holding_period_days": execution.holding_period_days,
            "prediction_rows": len(predictions),
            "label_audit": label_audit,
            "metrics": metrics,
            "accounting_schema_version": 2,
            "accounting_summaries": {
                str(result.top_n): result.accounting_summary for result in results
            },
            "interpretation": (
                "Executable historical simulation using same-date champion scores and next-open "
                "execution. Labels are audited after selection and never drive scores or returns."
            ),
        }
        manifest = self._manifest(
            champion,
            feature_hash,
            feature_names,
            requested_start,
            requested_end,
            effective_end,
            selected_top_n,
            label_audit,
            boundary,
            results,
        )
        _publish(output_dir, summary, manifest, predictions, daily, holdings)
        return HistoricalBacktestResult(
            run_id, output_dir, champion.model_id, requested_start, effective_end, metrics
        )

    def _resolve_dates(
        self,
        period: str | None,
        start_date: str | None,
        end_date: str | None,
    ) -> tuple[str, str]:
        if period is not None:
            if start_date is not None or end_date is not None:
                raise DataValidationError("--period cannot be combined with explicit dates")
            configured = self.settings.backtest.historical.periods.get(period)
            if configured is None:
                known = sorted(self.settings.backtest.historical.periods)
                raise DataValidationError(f"unknown historical period {period}; configured={known}")
            return configured.start_date, configured.end_date
        if start_date is None or end_date is None:
            raise DataValidationError("provide --period or both --start-date and --end-date")
        if start_date > end_date:
            raise DataValidationError("historical backtest start_date is after end_date")
        return start_date, end_date

    def _execution_settings(self, top_n: tuple[int, ...]) -> BacktestSettings:
        return self.settings.backtest.model_copy(
            update={
                "top_n": top_n,
                "holding_period_days": self.settings.backtest.historical.holding_period_days,
            }
        )

    def _manifest(
        self,
        champion: RegisteredModel,
        feature_hash: str,
        feature_names: tuple[str, ...],
        requested_start: str,
        requested_end: str,
        effective_end: str,
        top_n: tuple[int, ...],
        label_audit: dict[str, Any],
        boundary: ModelEvaluationBoundary,
        results: list[BacktestResult],
    ) -> dict[str, Any]:
        git = current_git_info()
        candidate_config = {
            "universe_filter": "same_date_in_model_universe",
            "ranking": "prediction_score_desc_then_ts_code",
            "top_n": list(top_n),
            "does_not_mutate_candidate_selection": True,
        }
        backtest_config = {
            **self.settings.backtest.model_dump(mode="json", exclude={"historical"}),
            "historical": self.settings.backtest.historical.model_dump(mode="json"),
        }
        return {
            "schema_version": 2,
            "artifact_name": "historical_champion_backtest",
            "model_id": champion.model_id,
            "model_type": champion.model_type,
            "model_artifact": champion.artifact_path,
            "model_training_date_range": champion.training_date_range,
            "model_boundary": boundary.to_dict(),
            "model_manifest_hash": boundary.manifest_hash,
            "feature_hash": feature_hash,
            "feature_count": len(feature_names),
            "candidate_config": candidate_config,
            "backtest_config": backtest_config,
            "requested_start_date": requested_start,
            "requested_end_date": requested_end,
            "effective_end_date": effective_end,
            "out_of_sample": True,
            "purpose": "OOS_EVIDENCE",
            "accounting_schema_version": 2,
            "execution_cost_policy": results[0].cost_policy,
            "cost_policy_hash": results[0].cost_policy["cost_policy_hash"],
            "label_audit": label_audit,
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "config_path": str(self.config_path),
            "config_hash": config_hash(self.config_path),
            "source_manifests": {
                name: read_manifest(self.processed_root / name)
                for name in ("features_daily", "universe_daily", "labels_forward")
            },
            "prediction_file": "predictions.parquet",
        }


def _effective_end_date(
    processed_root: Path,
    start_date: str,
    requested_end: str,
    horizon: int,
) -> str:
    maxima: list[str] = []
    with duckdb.connect() as connection:
        for dataset in ("features_daily", "universe_daily"):
            glob = processed_root / dataset / "**" / "*.parquet"
            try:
                row = connection.execute(
                    f"SELECT MAX(CAST(trade_date AS VARCHAR)) "  # noqa: S608
                    f"FROM read_parquet('{glob.as_posix()}', hive_partitioning=false)"
                ).fetchone()
            except duckdb.Error as error:
                raise DataValidationError(f"cannot inspect {dataset}: {error}") from error
            if row is None or row[0] is None:
                raise DataValidationError(f"{dataset} is empty")
            maxima.append(str(row[0]))
        label_glob = processed_root / "labels_forward" / "**" / "*.parquet"
        try:
            row = connection.execute(
                f"SELECT MAX(CAST(trade_date AS VARCHAR)) "  # noqa: S608
                f"FROM read_parquet('{label_glob.as_posix()}', hive_partitioning=false) "
                "WHERE CAST(horizon AS INTEGER) = ? AND CAST(is_label_available AS BOOLEAN)",
                [horizon],
            ).fetchone()
        except duckdb.Error as error:
            raise DataValidationError(f"cannot inspect labels_forward: {error}") from error
    if row is None or row[0] is None:
        raise DataValidationError(f"labels_forward has no available horizon={horizon} rows")
    maxima.append(str(row[0]))
    effective = min(requested_end, *maxima)
    if effective < start_date:
        raise DataValidationError(
            f"no complete historical backtest data for {start_date}..{requested_end}"
        )
    return effective


def _validate_signal_chronology(signals: DataFrame, start_date: str, end_date: str) -> None:
    dates = signals["trade_date"].astype(str)
    if signals.empty or (dates < start_date).any() or (dates > end_date).any():
        raise DataValidationError("historical signals are empty or outside the requested range")
    ordered = signals.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(
        drop=True
    )
    if not ordered[["trade_date", "ts_code"]].equals(
        signals.reset_index(drop=True)[["trade_date", "ts_code"]]
    ):
        raise DataValidationError("historical prediction inputs are not chronologically ordered")
    if signals.duplicated(["trade_date", "ts_code"]).any():
        raise DataValidationError("historical predictions contain duplicate keys")


def _audit_labels(
    processed_root: Path,
    predictions: DataFrame,
    start_date: str,
    end_date: str,
    horizon: int,
) -> dict[str, Any]:
    selected = predictions.loc[predictions["selected_flag"], ["trade_date", "ts_code"]].copy()
    glob = processed_root / "labels_forward" / "**" / "*.parquet"
    query = f"""
        SELECT CAST(trade_date AS VARCHAR) AS trade_date,
               CAST(ts_code AS VARCHAR) AS ts_code,
               CAST(entry_date AS VARCHAR) AS entry_date,
               CAST(exit_date AS VARCHAR) AS exit_date,
               CAST(is_label_available AS BOOLEAN) AS is_label_available,
               CAST(label_unavailable_reason AS VARCHAR) AS label_unavailable_reason
        FROM read_parquet('{glob.as_posix()}', hive_partitioning=false)
        WHERE CAST(trade_date AS VARCHAR) BETWEEN ? AND ?
          AND CAST(horizon AS INTEGER) = ?
    """  # noqa: S608 -- fixed local artifact and parameterized values
    try:
        with duckdb.connect() as connection:
            labels = connection.execute(query, [start_date, end_date, horizon]).fetch_df()
    except duckdb.Error as error:
        raise DataValidationError(f"cannot audit labels_forward: {error}") from error
    if labels.duplicated(["trade_date", "ts_code"]).any():
        raise DataValidationError("labels_forward contains duplicate audit keys")
    audited = selected.merge(
        labels,
        on=["trade_date", "ts_code"],
        how="left",
        validate="many_to_one",
    )
    available = audited["is_label_available"].fillna(False).astype(bool)
    dated = audited.loc[available]
    invalid = (dated["entry_date"] <= dated["trade_date"]) | (
        dated["exit_date"] <= dated["entry_date"]
    )
    if invalid.any():
        raise DataValidationError("label audit found non-future entry or exit dates")
    return {
        "selected_rows": len(selected),
        "matched_label_rows": int(audited["entry_date"].notna().sum()),
        "available_label_rows": int(available.sum()),
        "available_ratio": float(available.mean()) if len(available) else 0.0,
        "labels_used_for_selection": False,
        "labels_used_for_returns": False,
    }


def _metrics_with_years(
    result: BacktestResult,
    settings: BacktestSettings,
    bull_threshold: float,
    bear_threshold: float,
) -> dict[str, Any]:
    daily = result.daily_returns.sort_values("trade_date").copy()
    returns = daily["net_return"].to_numpy(dtype=float)
    benchmark = daily["benchmark_return"].to_numpy(dtype=float)
    cumulative = float(np.prod(1.0 + returns) - 1.0)
    benchmark_cumulative = float(np.prod(1.0 + benchmark) - 1.0)
    overall = {
        "cumulative_return": cumulative,
        "annual_return": result.metrics.get("annual_return"),
        "benchmark_return": benchmark_cumulative,
        "excess_return": (1.0 + cumulative) / (1.0 + benchmark_cumulative) - 1.0,
        "sharpe": result.metrics.get("sharpe"),
        "max_drawdown": result.metrics.get("maximum_drawdown"),
        "volatility": result.metrics.get("annual_volatility"),
        "turnover": result.metrics.get("average_turnover"),
        "holding_days": result.metrics.get("average_holding_period_sessions"),
        "win_rate": result.metrics.get("daily_win_rate"),
        "days": result.metrics.get("days"),
    }
    daily["year"] = daily["trade_date"].astype(str).str[:4]
    yearly: list[dict[str, Any]] = []
    for year, frame in daily.groupby("year", sort=True):
        year_returns = frame["net_return"].to_numpy(dtype=float)
        year_benchmark = frame["benchmark_return"].to_numpy(dtype=float)
        strategy_return = float(np.prod(1.0 + year_returns) - 1.0)
        benchmark_return = float(np.prod(1.0 + year_benchmark) - 1.0)
        session_std = float(np.std(year_returns, ddof=1)) if len(year_returns) > 1 else 0.0
        volatility = session_std * np.sqrt(settings.annualization_days)
        daily_risk_free = (1.0 + settings.risk_free_annual_rate) ** (
            1.0 / settings.annualization_days
        ) - 1.0
        equity = np.cumprod(1.0 + year_returns)
        drawdown = equity / np.maximum.accumulate(equity) - 1.0
        regime = (
            "bull"
            if benchmark_return >= bull_threshold
            else "bear"
            if benchmark_return <= bear_threshold
            else "neutral"
        )
        yearly.append(
            {
                "year": str(year),
                "regime": regime,
                "cumulative_return": strategy_return,
                "benchmark_return": benchmark_return,
                "excess_return": (1.0 + strategy_return) / (1.0 + benchmark_return) - 1.0,
                "volatility": volatility,
                "sharpe": (
                    None
                    if volatility == 0
                    else float(
                        np.mean(year_returns - daily_risk_free)
                        / session_std
                        * np.sqrt(settings.annualization_days)
                    )
                ),
                "max_drawdown": float(np.min(drawdown)),
                "win_rate": float(np.mean(year_returns > 0)),
                "turnover": float(frame["turnover"].astype(float).mean()),
            }
        )
    return {"overall": overall, "yearly": yearly}


def _run_id(
    model_id: str,
    start_date: str,
    end_date: str,
    feature_hash: str,
    top_n: tuple[int, ...],
) -> str:
    safe_model = "".join(character if character.isalnum() else "_" for character in model_id)
    suffix = feature_hash[:8]
    top = "_".join(str(value) for value in top_n)
    return f"{safe_model}_{start_date}_{end_date}_top{top}_{suffix}"


def _publish(
    output_dir: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    predictions: DataFrame,
    daily: DataFrame,
    holdings: DataFrame,
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        try:
            existing_manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            existing_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DataValidationError(
                f"incomplete immutable historical backtest exists: {output_dir}"
            ) from error
        if existing_manifest == manifest and existing_summary == summary:
            return
        raise DataValidationError(f"immutable historical backtest identity differs: {output_dir}")
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        predictions.to_parquet(staging / "predictions.parquet", index=False)
        daily.to_parquet(staging / "daily_returns.parquet", index=False)
        holdings.to_parquet(staging / "holdings.parquet", index=False)
        atomic_write_json(staging / "summary.json", summary)
        (staging / "backtest_report.md").write_text(_render_report(summary), encoding="utf-8")
        atomic_write_json(staging / "manifest.json", manifest)
        staging.rename(output_dir)


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Historical Champion Backtest Report",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Model ID: `{summary['model_id']}`",
        f"- Period: {summary['effective_start_date']} to {summary['effective_end_date']}",
        f"- Holding period: {summary['holding_period_days']} trading days",
        "- Scope: out-of-sample executable simulation",
        "",
        "> Labels are used only for post-selection chronology and coverage auditing. "
        "They do not affect model scores, selection, or simulated returns.",
    ]
    metrics = summary["metrics"]
    assert isinstance(metrics, dict)
    for top_n, result in metrics.items():
        assert isinstance(result, dict)
        overall = result["overall"]
        assert isinstance(overall, dict)
        lines.extend(
            [
                "",
                f"## Top {top_n}",
                "",
                f"- Cumulative return: {_percent(overall['cumulative_return'])}",
                f"- Annual return: {_percent(overall['annual_return'])}",
                f"- Excess return: {_percent(overall['excess_return'])}",
                f"- Sharpe: {_number(overall['sharpe'])}",
                f"- Maximum drawdown: {_percent(overall['max_drawdown'])}",
                f"- Annual volatility: {_percent(overall['volatility'])}",
                f"- Average turnover: {_percent(overall['turnover'])}",
                f"- Holding days: {overall['holding_days']}",
                f"- Win rate: {_percent(overall['win_rate'])}",
                "",
                "### Year and Market Regime",
                "",
                "| Year | Regime | Return | Benchmark | Excess | Sharpe | Max DD |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        yearly = result["yearly"]
        assert isinstance(yearly, list)
        for row in yearly:
            assert isinstance(row, dict)
            lines.append(
                f"| {row['year']} | {row['regime']} | {_percent(row['cumulative_return'])} | "
                f"{_percent(row['benchmark_return'])} | {_percent(row['excess_return'])} | "
                f"{_number(row['sharpe'])} | {_percent(row['max_drawdown'])} |"
            )
    return "\n".join(lines) + "\n"


def _percent(value: object) -> str:
    return "NA" if value is None else f"{float(value):.2%}"  # type: ignore[arg-type]


def _number(value: object) -> str:
    return "NA" if value is None else f"{float(value):.4f}"  # type: ignore[arg-type]
