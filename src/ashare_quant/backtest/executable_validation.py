"""Executable OOS Champion-versus-Challenger portfolio validation."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.backtest.data import load_benchmark, load_calendar, load_execution_prices
from ashare_quant.backtest.engine import BacktestInputs, BacktestResult, simulate_portfolio
from ashare_quant.backtest.provenance import (
    require_oos_evaluation,
    resolve_model_evaluation_boundary,
)
from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.challenger_evaluation import (
    _load_predictions,
    _require_identical_prediction_keys,
    _validate_prediction_contract,
)
from ashare_quant.models.inference import (
    PredictionModel,
    load_registered_feature_list,
    score_registered_model_range,
)
from ashare_quant.models.registry import ModelRegistry, RegisteredModel
from ashare_quant.utils.manifest import atomic_write_json, config_hash, current_git_info

EXECUTABLE_VALIDATION_SCHEMA_VERSION = 2
REQUIRED_TOP_N = (10, 20, 50)

type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class ExecutableValidationResult:
    """One immutable executable OOS model comparison."""

    run_id: str
    champion_model_id: str
    challenger_model_id: str
    horizon: int
    output_dir: Path
    metrics: dict[str, dict[str, dict[str, float | int | None]]]


class ExecutableOOSValidationEngine:
    """Run two frozen models through identical next-open portfolio mechanics."""

    def __init__(
        self,
        *,
        raw_root: Path,
        processed_root: Path,
        models_root: Path,
        reports_root: Path,
        settings: AppSettings,
        config_path: Path,
        model_loader: Callable[[Path], PredictionModel] | None = None,
    ) -> None:
        self.raw_root = raw_root
        self.processed_root = processed_root
        self.models_root = models_root
        self.reports_root = reports_root
        self.settings = settings
        self.config_path = config_path
        self._model_loader = model_loader

    def run(
        self,
        model_ids: Sequence[str],
        *,
        top_n: tuple[int, ...] = REQUIRED_TOP_N,
    ) -> ExecutableValidationResult:
        """Compare the Champion and one Challenger on the Challenger's frozen OOS scope."""

        if tuple(sorted(set(top_n))) != tuple(sorted(REQUIRED_TOP_N)):
            raise DataValidationError("executable validation requires Top10, Top20, and Top50")
        champion, challenger = self._resolve_models(model_ids)
        registry_path = self.models_root / "registry.json"
        registry_before = _file_hash(registry_path)
        challenger_predictions, prediction_manifest, horizon = self._load_challenger_predictions(
            champion, challenger
        )
        requested_dates = tuple(sorted(challenger_predictions["trade_date"].astype(str).unique()))
        execution = self.settings.backtest.model_copy(
            update={"execution": "next_open", "holding_period_days": horizon, "top_n": top_n}
        )
        calendar = load_calendar(
            self.raw_root,
            requested_dates[0],
            requested_dates[-1],
            horizon + execution.sell_delay_max_days,
        )
        if not calendar or requested_dates[0] not in calendar:
            raise DataValidationError("executable validation has no authoritative trade calendar")
        prices = load_execution_prices(
            self.raw_root,
            self.processed_root,
            calendar[0],
            calendar[-1],
            self.settings.universe.price_tolerance,
        )
        maximum_price_date = str(prices["trade_date"].astype(str).max())
        calendar = [date for date in calendar if date <= maximum_price_date]
        dates = _fully_executable_signal_dates(
            requested_dates, calendar, horizon, execution.sell_delay_max_days
        )
        challenger_predictions = challenger_predictions.loc[
            challenger_predictions["trade_date"].astype(str).isin(dates)
        ].reset_index(drop=True)
        for model in (champion, challenger):
            boundary = resolve_model_evaluation_boundary(Path(model.artifact_path))
            require_oos_evaluation(
                boundary,
                model_dir=Path(model.artifact_path),
                evaluation_start=dates[0],
                evaluation_end=dates[-1],
            )
        champion_batch = score_registered_model_range(
            champion,
            processed_root=self.processed_root,
            start_date=dates[0],
            end_date=dates[-1],
            allowed_ranges=((dates[0], dates[-1]),),
            model_loader=self._model_loader,
        )
        champion_predictions = champion_batch.predictions.loc[
            champion_batch.predictions["trade_date"].astype(str).isin(dates)
        ].reset_index(drop=True)
        _require_identical_prediction_keys(challenger_predictions, champion_predictions)
        benchmark = load_benchmark(
            self.raw_root,
            execution.benchmark_index_code,
            calendar[0],
            calendar[-1],
        )
        signals = {
            champion.model_id: _signals(champion_predictions),
            challenger.model_id: _signals(challenger_predictions),
        }
        model_results = {
            model_id: tuple(
                simulate_portfolio(
                    BacktestInputs(
                        signals=model_signals,
                        prices=prices,
                        calendar=tuple(calendar),
                        benchmark=benchmark,
                    ),
                    top_n=value,
                    settings=execution,
                    purpose="executable_validation",
                )
                for value in top_n
            )
            for model_id, model_signals in signals.items()
        }
        for model_id, results in model_results.items():
            for result in results:
                if (
                    not result.holdings.empty
                    and result.holdings["trade_date"].astype(str).eq(calendar[-1]).any()
                ):
                    raise DataValidationError(
                        f"executable validation ends with unresolved holdings: "
                        f"model={model_id} top_n={result.top_n}"
                    )
        if _file_hash(registry_path) != registry_before:
            raise DataValidationError("model registry changed during executable validation")
        metrics = {
            model_id: {str(result.top_n): result.metrics for result in results}
            for model_id, results in model_results.items()
        }
        identity = self._identity(
            champion, challenger, prediction_manifest, horizon, top_n, execution.model_dump()
        )
        run_id = f"executable_oos_h{horizon}_{identity[:16]}"
        output_dir = self.reports_root / "executable_validation" / run_id
        existing = _existing_result(output_dir, identity)
        if existing is not None:
            return existing
        summary = _summary(
            run_id,
            champion,
            challenger,
            horizon,
            dates,
            top_n,
            execution.model_dump(mode="json"),
            metrics,
            model_results,
        )
        manifest = self._manifest(
            identity,
            run_id,
            champion,
            challenger,
            prediction_manifest,
            dates,
            top_n,
            execution.model_dump(mode="json"),
            model_results,
        )
        _publish(output_dir, summary, manifest, model_results)
        return ExecutableValidationResult(
            run_id,
            champion.model_id,
            challenger.model_id,
            horizon,
            output_dir,
            metrics,
        )

    def _resolve_models(self, model_ids: Sequence[str]) -> tuple[RegisteredModel, RegisteredModel]:
        if len(model_ids) != 2 or len(set(model_ids)) != 2:
            raise DataValidationError(
                "specify exactly two unique model IDs: champion and challenger"
            )
        registry = ModelRegistry(self.models_root)
        champion = registry.get_champion("lightgbm_ranker")
        if champion is None:
            raise DataValidationError("no lightgbm_ranker champion is registered")
        resolved = [
            champion if model_id == "champion" else _find_model(registry, model_id)
            for model_id in model_ids
        ]
        if champion.model_id not in {model.model_id for model in resolved}:
            raise DataValidationError("executable validation requires the current champion")
        challenger = next(
            (model for model in resolved if model.model_id != champion.model_id), None
        )
        if challenger is None or challenger.status != "candidate":
            raise DataValidationError("executable validation requires one candidate challenger")
        return champion, challenger

    def _load_challenger_predictions(
        self, champion: RegisteredModel, challenger: RegisteredModel
    ) -> tuple[DataFrame, dict[str, Any], int]:
        champion_features, champion_hash = load_registered_feature_list(
            Path(champion.artifact_path), champion
        )
        challenger_features, challenger_hash = load_registered_feature_list(
            Path(challenger.artifact_path), challenger
        )
        if champion_hash != challenger_hash or champion_features != challenger_features:
            raise DataValidationError("champion and challenger feature hashes differ")
        prediction_dir = self.reports_root / "challenger_predictions" / challenger.model_id
        manifest = _load_json(prediction_dir / "manifest.json", "challenger prediction manifest")
        horizon, _ = _validate_prediction_contract(
            manifest, challenger, processed_root=self.processed_root
        )
        predictions = _load_predictions(
            prediction_dir / "predictions.parquet", challenger.model_id, manifest
        )
        return predictions, manifest, horizon

    def _identity(
        self,
        champion: RegisteredModel,
        challenger: RegisteredModel,
        prediction_manifest: dict[str, Any],
        horizon: int,
        top_n: tuple[int, ...],
        execution: dict[str, Any],
    ) -> str:
        git = current_git_info()
        return _payload_hash(
            {
                "schema_version": EXECUTABLE_VALIDATION_SCHEMA_VERSION,
                "champion_model_id": champion.model_id,
                "champion_manifest_hash": _file_hash(
                    Path(champion.artifact_path) / "manifest.json"
                ),
                "challenger_model_id": challenger.model_id,
                "challenger_prediction_identity": prediction_manifest.get("prediction_identity"),
                "horizon": horizon,
                "top_n": top_n,
                "execution": execution,
                "config_hash": config_hash(self.config_path),
                "git_commit": git["commit"],
            }
        )

    def _manifest(
        self,
        identity: str,
        run_id: str,
        champion: RegisteredModel,
        challenger: RegisteredModel,
        prediction_manifest: dict[str, Any],
        dates: tuple[str, ...],
        top_n: tuple[int, ...],
        execution: dict[str, Any],
        model_results: dict[str, tuple[BacktestResult, ...]],
    ) -> dict[str, Any]:
        git = current_git_info()
        return {
            "schema_version": EXECUTABLE_VALIDATION_SCHEMA_VERSION,
            "artifact_name": "executable_oos_portfolio_validation_manifest",
            "validation_identity": identity,
            "run_id": run_id,
            "champion_model_id": champion.model_id,
            "challenger_model_id": challenger.model_id,
            "horizon": prediction_manifest["horizon"],
            "holding_period": prediction_manifest["holding_period"],
            "execution_rule": "signal_close_t_next_open_entry_and_horizon_open_exit",
            "accounting_schema_version": 2,
            "terminal_untradable_policy": "explicit_terminal_event_only; unresolved_fails_closed",
            "execution_cost_policy": next(iter(model_results.values()))[0].cost_policy,
            "cost_policy_hash": next(iter(model_results.values()))[0].cost_policy[
                "cost_policy_hash"
            ],
            "accounting_summaries": {
                model_id: {str(result.top_n): result.accounting_summary for result in results}
                for model_id, results in model_results.items()
            },
            "top_n": list(top_n),
            "minimum_signal_date": dates[0],
            "maximum_signal_date": dates[-1],
            "signal_dates": len(dates),
            "execution_config": execution,
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "config_path": str(self.config_path),
            "config_hash": config_hash(self.config_path),
            "input_manifests": {
                "champion_model": _file_hash(Path(champion.artifact_path) / "manifest.json"),
                "challenger_model": _file_hash(Path(challenger.artifact_path) / "manifest.json"),
                "challenger_predictions": prediction_manifest,
                "features_daily": _file_hash(
                    self.processed_root / "features_daily" / "_manifest.json"
                ),
                "universe_daily": _file_hash(
                    self.processed_root / "universe_daily" / "_manifest.json"
                ),
            },
            "isolation_contract": {
                "labels_loaded": False,
                "future_features_loaded": False,
                "same_dates": True,
                "same_universe_rows": True,
                "same_execution_rules": True,
                "registry_modified": False,
                "automatic_promotion": False,
                "models_modified": False,
            },
        }


def _signals(predictions: DataFrame) -> DataFrame:
    return predictions[["trade_date", "ts_code", "prediction_score"]].rename(
        columns={"prediction_score": "score"}
    )


def _fully_executable_signal_dates(
    requested_dates: tuple[str, ...],
    calendar: list[str],
    horizon: int,
    sell_delay_max_days: int,
) -> tuple[str, ...]:
    positions = {date: index for index, date in enumerate(calendar)}
    dates = tuple(
        date
        for date in requested_dates
        if date in positions and positions[date] + horizon + sell_delay_max_days + 1 < len(calendar)
    )
    if not dates:
        raise DataValidationError(
            "no OOS signal date has sufficient prices for next-open entry and horizon exit"
        )
    return dates


def _summary(
    run_id: str,
    champion: RegisteredModel,
    challenger: RegisteredModel,
    horizon: int,
    dates: tuple[str, ...],
    top_n: tuple[int, ...],
    execution: dict[str, Any],
    metrics: dict[str, dict[str, dict[str, float | int | None]]],
    model_results: dict[str, tuple[BacktestResult, ...]],
) -> dict[str, Any]:
    comparison: dict[str, dict[str, float | None]] = {}
    for value in top_n:
        champion_metrics = metrics[champion.model_id][str(value)]
        challenger_metrics = metrics[challenger.model_id][str(value)]
        comparison[str(value)] = {
            name: _metric_delta(challenger_metrics, champion_metrics, name)
            for name in (
                "annual_return",
                "sharpe",
                "maximum_drawdown",
                "average_turnover",
                "trade_win_rate",
                "profit_loss_ratio",
            )
        }
    return {
        "schema_version": EXECUTABLE_VALIDATION_SCHEMA_VERSION,
        "artifact_name": "executable_oos_portfolio_validation",
        "run_id": run_id,
        "champion_model_id": champion.model_id,
        "challenger_model_id": challenger.model_id,
        "horizon": horizon,
        "holding_period": horizon,
        "minimum_signal_date": dates[0],
        "maximum_signal_date": dates[-1],
        "signal_dates": len(dates),
        "top_n": list(top_n),
        "execution_config": execution,
        "accounting_schema_version": 2,
        "terminal_untradable_policy": "explicit_terminal_event_only; unresolved_fails_closed",
        "execution_cost_policy": next(iter(model_results.values()))[0].cost_policy,
        "cost_policy_hash": next(iter(model_results.values()))[0].cost_policy["cost_policy_hash"],
        "accounting_summaries": {
            model_id: {str(result.top_n): result.accounting_summary for result in results}
            for model_id, results in model_results.items()
        },
        "metrics": metrics,
        "challenger_minus_champion": comparison,
        "interpretation": (
            "Executable next-open OOS comparison. Positive deltas are not uniformly better for "
            "maximum_drawdown or turnover; no model is promoted by this report."
        ),
    }


def _metric_delta(
    challenger: dict[str, float | int | None],
    champion: dict[str, float | int | None],
    name: str,
) -> float | None:
    left = challenger.get(name)
    right = champion.get(name)
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return float(left) - float(right)


def _publish(
    output_dir: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    model_results: dict[str, tuple[BacktestResult, ...]],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix=".executable-") as temporary:
        staging = Path(temporary)
        atomic_write_json(staging / "summary.json", summary)
        (staging / "report.md").write_text(_render_report(summary), encoding="utf-8")
        _combined_frame(model_results, "daily_returns").to_parquet(
            staging / "daily_returns.parquet", index=False
        )
        _combined_frame(model_results, "trades").to_parquet(staging / "trades.parquet", index=False)
        _combined_frame(model_results, "holdings").to_parquet(
            staging / "holdings.parquet", index=False
        )
        atomic_write_json(staging / "manifest.json", manifest)
        if output_dir.exists():
            raise DataValidationError(f"immutable executable validation exists: {output_dir}")
        staging.rename(output_dir)


def _combined_frame(model_results: dict[str, tuple[BacktestResult, ...]], field: str) -> DataFrame:
    frames: list[DataFrame] = []
    for model_id, results in model_results.items():
        for result in results:
            frame = getattr(result, field).copy()
            frame["model_id"] = model_id
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Executable OOS Portfolio Validation",
        "",
        f"- Champion: `{summary['champion_model_id']}`",
        f"- Challenger: `{summary['challenger_model_id']}`",
        f"- Signal dates: {summary['minimum_signal_date']} to {summary['maximum_signal_date']}",
        f"- Holding period: {summary['holding_period']} trading days",
        "- Execution: signal close, next-open entry, horizon next-open exit",
        "- Unresolved positions fail closed; write-off requires a verified terminal event",
        "",
    ]
    for value in summary["top_n"]:
        lines.extend(
            [
                f"## Top {value}",
                "",
                "| Model | Annual return | Sharpe | Max drawdown | Turnover | Trade win | P/L |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for model_id in (summary["champion_model_id"], summary["challenger_model_id"]):
            metrics = summary["metrics"][model_id][str(value)]
            lines.append(
                f"| `{model_id}` | {_number(metrics.get('annual_return'))} | "
                f"{_number(metrics.get('sharpe'))} | "
                f"{_number(metrics.get('maximum_drawdown'))} | "
                f"{_number(metrics.get('average_turnover'))} | "
                f"{_number(metrics.get('trade_win_rate'))} | "
                f"{_number(metrics.get('profit_loss_ratio'))} |"
            )
        lines.append("")
    lines.extend(
        [
            "This validation does not modify models, change the registry, promote a model, or "
            "generate live orders.",
            "",
        ]
    )
    return "\n".join(lines)


def _number(value: object) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{float(value):.6f}"


def _existing_result(output_dir: Path, identity: str) -> ExecutableValidationResult | None:
    if not output_dir.exists():
        return None
    manifest = _load_json(output_dir / "manifest.json", "executable validation manifest")
    if manifest.get("validation_identity") != identity:
        raise DataValidationError(f"immutable executable validation identity differs: {output_dir}")
    summary = _load_json(output_dir / "summary.json", "executable validation summary")
    return ExecutableValidationResult(
        str(summary["run_id"]),
        str(summary["champion_model_id"]),
        str(summary["challenger_model_id"]),
        int(summary["horizon"]),
        output_dir,
        summary["metrics"],
    )


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
