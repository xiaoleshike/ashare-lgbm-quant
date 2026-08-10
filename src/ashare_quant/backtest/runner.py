"""Run and persist executable portfolio backtests for Ranker artifacts."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.backtest.data import load_backtest_inputs, load_model_and_features
from ashare_quant.backtest.engine import BacktestResult, simulate_portfolio
from ashare_quant.backtest.provenance import (
    ModelEvaluationBoundary,
    require_oos_evaluation,
    resolve_model_evaluation_boundary,
)
from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.utils.manifest import config_hash, current_git_info


@dataclass(frozen=True, slots=True)
class BacktestRunResult:
    """Summary of a persisted backtest run."""

    experiment_id: str
    output_dir: Path
    top_n: tuple[int, ...]
    metrics: dict[str, dict[str, float | int | None]]


class BacktestRunner:
    """Convert saved Ranker scores into executable portfolio simulations."""

    def __init__(
        self,
        raw_root: Path,
        processed_root: Path,
        model_root: Path,
        output_root: Path,
        settings: AppSettings,
        config_path: Path,
    ) -> None:
        self.raw_root = raw_root
        self.processed_root = processed_root
        self.model_root = model_root
        self.output_root = output_root
        self.settings = settings
        self.config_path = config_path

    def run(
        self,
        *,
        model_dir: Path | None,
        start_date: str,
        end_date: str,
        top_n: tuple[int, ...] | None = None,
    ) -> BacktestRunResult:
        """Run all requested Top-N variants and publish outputs atomically."""

        resolved_model_dir = self._resolve_model_dir(model_dir)
        boundary = resolve_model_evaluation_boundary(resolved_model_dir)
        require_oos_evaluation(
            boundary,
            model_dir=resolved_model_dir,
            evaluation_start=start_date,
            evaluation_end=end_date,
        )
        model, feature_names, feature_hash = load_model_and_features(resolved_model_dir)
        inputs = load_backtest_inputs(
            raw_root=self.raw_root,
            processed_root=self.processed_root,
            model=model,
            feature_names=feature_names,
            start_date=start_date,
            end_date=end_date,
            settings=self.settings,
        )
        top_values = top_n or tuple(int(value) for value in self.settings.backtest.top_n)
        results = [
            simulate_portfolio(
                inputs,
                top_n=value,
                settings=self.settings.backtest,
                purpose="oos_evidence",
            )
            for value in top_values
        ]
        experiment_id = (
            f"{resolved_model_dir.name}_backtest_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
        )
        output_dir = self._persist(
            experiment_id,
            resolved_model_dir,
            feature_hash,
            feature_names,
            start_date,
            end_date,
            inputs.signals,
            top_values,
            results,
            boundary,
        )
        return BacktestRunResult(
            experiment_id=experiment_id,
            output_dir=output_dir,
            top_n=top_values,
            metrics={str(result.top_n): result.metrics for result in results},
        )

    def _resolve_model_dir(self, model_dir: Path | None) -> Path:
        if model_dir is not None:
            if not model_dir.exists():
                raise DataValidationError(f"model directory does not exist: {model_dir}")
            return model_dir
        candidates = sorted(self.model_root.glob("experiment_b_robust_*"))
        if not candidates:
            raise DataValidationError(
                "no experiment_b_robust_* model directory found; pass --model-dir"
            )
        return candidates[-1]

    def _persist(
        self,
        experiment_id: str,
        model_dir: Path,
        feature_hash: str,
        feature_names: tuple[str, ...],
        start_date: str,
        end_date: str,
        signals: pd.DataFrame,
        top_n: tuple[int, ...],
        results: list[BacktestResult],
        boundary: ModelEvaluationBoundary,
    ) -> Path:
        final_dir = self.output_root / experiment_id
        self.output_root.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            raise DataValidationError(f"backtest output directory already exists: {final_dir}")
        with tempfile.TemporaryDirectory(dir=self.output_root) as temporary:
            directory = Path(temporary)
            pd.concat([result.daily_returns for result in results], ignore_index=True).to_csv(
                directory / "daily_returns.csv", index=False
            )
            pd.concat([result.trades for result in results], ignore_index=True).to_csv(
                directory / "trades.csv", index=False
            )
            pd.concat([result.holdings for result in results], ignore_index=True).to_csv(
                directory / "holdings.csv", index=False
            )
            build_predictions_frame(signals, top_n).to_csv(
                directory / "predictions.csv", index=False
            )
            metrics: dict[str, Any] = {
                "schema_version": 2,
                "accounting_schema_version": 2,
                "results": {str(result.top_n): result.metrics for result in results},
            }
            _write_json(directory / "metrics.json", metrics)
            _write_json(
                directory / "manifest.json",
                self._manifest(
                    experiment_id,
                    model_dir,
                    feature_hash,
                    feature_names,
                    start_date,
                    end_date,
                    results,
                    boundary,
                ),
            )
            directory.rename(final_dir)
        return final_dir

    def _manifest(
        self,
        experiment_id: str,
        model_dir: Path,
        feature_hash: str,
        feature_names: tuple[str, ...],
        start_date: str,
        end_date: str,
        results: list[BacktestResult],
        boundary: ModelEvaluationBoundary,
    ) -> dict[str, Any]:
        git_info = current_git_info()
        return {
            "schema_version": 2,
            "artifact_name": "ranker_executable_backtest",
            "experiment_id": experiment_id,
            "completed_at": datetime.now(UTC).isoformat(),
            "git_commit": git_info["commit"],
            "git_dirty": git_info["dirty"],
            "config_path": str(self.config_path),
            "config_hash": config_hash(self.config_path),
            "model_dir": str(model_dir),
            "model_boundary": boundary.to_dict(),
            "model_manifest_hash": boundary.manifest_hash,
            "feature_list_hash": feature_hash,
            "feature_count": len(feature_names),
            "start_date": start_date,
            "end_date": end_date,
            "top_n": [result.top_n for result in results],
            "purpose": "OOS_EVIDENCE",
            "accounting_schema_version": 2,
            "execution": "signal_close_t_next_open",
            "holding_period_days": self.settings.backtest.holding_period_days,
            "commission": self.settings.backtest.commission,
            "stamp_duty": self.settings.backtest.stamp_duty,
            "slippage": self.settings.backtest.slippage,
            "execution_cost_policy": results[0].cost_policy,
            "cost_policy_hash": results[0].cost_policy["cost_policy_hash"],
            "accounting_summaries": {
                str(result.top_n): result.accounting_summary for result in results
            },
            "benchmark_index_code": self.settings.backtest.benchmark_index_code,
            "prediction_file": "predictions.csv",
        }


def build_predictions_frame(signals: pd.DataFrame, top_n: tuple[int, ...]) -> pd.DataFrame:
    """Return per-date model rankings saved before portfolio execution."""

    if signals.empty:
        return pd.DataFrame(
            columns=["trade_date", "ts_code", "prediction_score", "rank", "selected_flag"]
        )
    max_top_n = max(top_n)
    frame = signals.rename(columns={"score": "prediction_score"}).copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["prediction_score"] = pd.to_numeric(frame["prediction_score"], errors="coerce")
    frame = frame.sort_values(
        ["trade_date", "prediction_score", "ts_code"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    frame["rank"] = frame.groupby("trade_date", sort=False).cumcount() + 1
    frame["selected_flag"] = frame["rank"] <= max_top_n
    return frame[["trade_date", "ts_code", "prediction_score", "rank", "selected_flag"]]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
