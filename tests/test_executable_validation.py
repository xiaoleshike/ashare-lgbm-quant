from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ashare_quant.backtest.engine import calculate_metrics
from ashare_quant.backtest.executable_validation import (
    ExecutableOOSValidationEngine,
    ExecutableValidationResult,
)
from ashare_quant.cli import main
from ashare_quant.config.settings import AppSettings, BacktestSettings
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.utils.manifest import atomic_write_json

SIGNAL_DATES = ("20240102", "20240103", "20240104")
CALENDAR = ("20240102", "20240103", "20240104", "20240105", "20240108", "20240109")


class _FixtureModel:
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return data["f1"].to_numpy(dtype=float)


def test_executable_validation_uses_next_open_horizon_exit_and_costs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, sources = _validation_fixture(tmp_path, monkeypatch)

    result = engine.run(("champion", "challenger_h2"))

    trades = pd.read_parquet(result.output_dir / "trades.parquet")
    filled = trades.loc[trades["status"] == "filled"]
    assert filled.loc[filled["side"] == "buy", "trade_date"].min() == "20240103"
    first_buys = filled.loc[(filled["side"] == "buy") & (filled["trade_date"] == "20240103")]
    first_codes = set(first_buys["ts_code"].astype(str))
    first_sells = filled.loc[(filled["side"] == "sell") & (filled["trade_date"] == "20240105")]
    assert first_codes <= set(first_sells["ts_code"].astype(str))
    assert filled["cost"].sum() > 0
    assert filled.loc[filled["side"] == "sell", "stamp_duty"].sum() > 0
    daily = pd.read_parquet(result.output_dir / "daily_returns.parquet")
    assert daily["cost"].sum() == pytest.approx(filled["cost"].sum())
    summary = _read_json(result.output_dir / "summary.json")
    assert summary["holding_period"] == 2
    assert (
        summary["terminal_untradable_policy"]
        == "explicit_terminal_event_only; unresolved_fails_closed"
    )
    assert summary["accounting_schema_version"] == 2
    assert summary["top_n"] == [10, 20, 50]
    assert {
        "trade_win_rate",
        "profit_loss_ratio",
        "annual_return",
        "sharpe",
        "maximum_drawdown",
        "average_turnover",
    } <= set(summary["metrics"]["challenger_h2"]["10"])


def test_executable_validation_is_oos_label_free_deterministic_and_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, sources = _validation_fixture(tmp_path, monkeypatch)
    registry_before = sources["registry"].read_bytes()

    first = engine.run(("champion", "challenger_h2"))
    before = {path.name: path.read_bytes() for path in first.output_dir.iterdir()}
    second = engine.run(("challenger_h2", "champion"))

    assert second.output_dir == first.output_dir
    assert {path.name: path.read_bytes() for path in second.output_dir.iterdir()} == before
    assert sources["registry"].read_bytes() == registry_before
    assert not (sources["processed"] / "labels_forward").exists()
    manifest = _read_json(first.output_dir / "manifest.json")
    assert manifest["minimum_signal_date"] > "20231231"
    assert manifest["isolation_contract"]["labels_loaded"] is False
    assert manifest["isolation_contract"]["future_features_loaded"] is False
    assert manifest["isolation_contract"]["registry_modified"] is False
    assert set(before) == {
        "summary.json",
        "report.md",
        "daily_returns.parquet",
        "trades.parquet",
        "holdings.parquet",
        "manifest.json",
    }


def test_executable_validation_cli_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "reports" / "executable_validation" / "run"
    monkeypatch.setattr(
        ExecutableOOSValidationEngine,
        "run",
        lambda self, model_ids, top_n: ExecutableValidationResult(
            run_id="run",
            champion_model_id="champion-id",
            challenger_model_id="challenger-id",
            horizon=10,
            output_dir=output,
            metrics={},
        ),
    )
    command = [
        "--config",
        "config/default.yaml",
        "backtest",
        "--storage-root",
        str(tmp_path / "raw"),
        "--processed-root",
        str(tmp_path / "processed"),
        "--models-root",
        str(tmp_path / "models"),
        "--output-root",
        str(tmp_path / "reports"),
        "executable-validation",
        "--model-id",
        "champion",
        "--model-id",
        "challenger-id",
    ]

    assert main(command) == 0
    assert "executable_validation: run_id=run" in capsys.readouterr().out


def test_trade_win_rate_and_profit_loss_ratio_use_net_closed_trade_pnl() -> None:
    daily = pd.DataFrame(
        {
            "net_return": [0.01, -0.01],
            "benchmark_return": [0.0, 0.0],
            "equity": [1_010.0, 1_000.0],
            "turnover": [0.2, 0.2],
        }
    )
    trades = pd.DataFrame(
        [
            _filled_trade("A", "buy", 100.0, 1.0),
            _filled_trade("A", "sell", 120.0, 1.0),
            _filled_trade("B", "buy", 100.0, 1.0),
            _filled_trade("B", "sell", 90.0, 1.0),
        ]
    )

    metrics = calculate_metrics(daily, trades, BacktestSettings(initial_cash=1_000.0))

    assert metrics["trade_win_rate"] == pytest.approx(0.5)
    assert metrics["profit_loss_ratio"] == pytest.approx(1.5)


def _validation_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ExecutableOOSValidationEngine, dict[str, Path]]:
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    reports = tmp_path / "reports"
    raw = tmp_path / "raw"
    config = tmp_path / "config.yaml"
    config.write_text("project_name: executable-validation-test\n", encoding="utf-8")
    features_dir = processed / "features_daily"
    universe_dir = processed / "universe_daily"
    features_dir.mkdir(parents=True)
    universe_dir.mkdir(parents=True)
    feature_rows: list[dict[str, object]] = []
    universe_rows: list[dict[str, object]] = []
    for date_index, trade_date in enumerate(SIGNAL_DATES):
        for stock_index in range(60):
            code = f"{stock_index:06d}.SZ"
            feature_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "f1": float(stock_index + date_index),
                    "f2": float(60 - stock_index),
                }
            )
            universe_rows.append(
                {"trade_date": trade_date, "ts_code": code, "in_model_universe": True}
            )
    pd.DataFrame(feature_rows).to_parquet(features_dir / "data.parquet", index=False)
    pd.DataFrame(universe_rows).to_parquet(universe_dir / "data.parquet", index=False)
    features_manifest = features_dir / "_manifest.json"
    universe_manifest = universe_dir / "_manifest.json"
    atomic_write_json(features_manifest, {"artifact_name": "features_daily"})
    atomic_write_json(universe_manifest, {"artifact_name": "universe_daily"})
    feature_names = ("f1", "f2")
    champion = _write_model(models, "champion-id", feature_names, champion=True)
    challenger = _write_model(models, "challenger_h2", feature_names, champion=False)
    registry = ModelRegistry(models)
    registry.register_model(champion)
    registry.promote_model("champion-id")
    registry.register_model(challenger)
    prediction_dir = reports / "challenger_predictions" / "challenger_h2"
    prediction_dir.mkdir(parents=True)
    predictions = pd.DataFrame(feature_rows)[["trade_date", "ts_code"]].copy()
    predictions["prediction_score"] = predictions["ts_code"].str[:6].astype(float)
    predictions["model_id"] = "challenger_h2"
    predictions = predictions.sort_values(
        ["trade_date", "prediction_score", "ts_code"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    predictions["rank"] = predictions.groupby("trade_date", sort=False).cumcount() + 1
    predictions.to_parquet(prediction_dir / "predictions.parquet", index=False)
    atomic_write_json(
        prediction_dir / "manifest.json",
        {
            "artifact_name": "challenger_predictions",
            "prediction_identity": "challenger-h2-predictions",
            "model_id": "challenger_h2",
            "feature_hash": feature_list_hash(feature_names),
            "universe_hash": _hash(universe_manifest),
            "horizon": 2,
            "holding_period": 2,
            "execution_rule": "next_open",
            "evaluation_ranges": [{"start_date": SIGNAL_DATES[0], "end_date": SIGNAL_DATES[-1]}],
            "maximum_mature_evaluation_date": SIGNAL_DATES[-1],
            "prediction_rows": len(predictions),
            "prediction_dates": len(SIGNAL_DATES),
            "input_manifests": {
                "model": _hash(challenger / "manifest.json"),
                "features_daily": _hash(features_manifest),
                "universe_daily": _hash(universe_manifest),
            },
        },
    )
    prices = pd.DataFrame(
        [
            {
                "trade_date": date,
                "ts_code": f"{stock_index:06d}.SZ",
                "open": 10.0 + CALENDAR.index(date) * 0.1 + stock_index * 0.001,
                "close": 10.05 + CALENDAR.index(date) * 0.1 + stock_index * 0.001,
                "can_buy": True,
                "can_sell": True,
            }
            for date in CALENDAR
            for stock_index in range(60)
        ]
    )
    benchmark = pd.DataFrame(
        {"trade_date": list(CALENDAR), "close": np.linspace(100.0, 101.0, len(CALENDAR))}
    )
    monkeypatch.setattr(
        "ashare_quant.backtest.executable_validation.load_calendar",
        lambda *_args, **_kwargs: list(CALENDAR),
    )
    monkeypatch.setattr(
        "ashare_quant.backtest.executable_validation.load_execution_prices",
        lambda *_args, **_kwargs: prices.copy(),
    )
    monkeypatch.setattr(
        "ashare_quant.backtest.executable_validation.load_benchmark",
        lambda *_args, **_kwargs: benchmark.copy(),
    )
    settings = AppSettings.model_validate({"backtest": {"sell_delay_max_days": 0}})
    engine = ExecutableOOSValidationEngine(
        raw_root=raw,
        processed_root=processed,
        models_root=models,
        reports_root=reports,
        settings=settings,
        config_path=config,
        model_loader=lambda _path: _FixtureModel(),
    )
    return engine, {
        "processed": processed,
        "registry": models / "registry.json",
    }


def _write_model(
    root: Path,
    model_id: str,
    features: tuple[str, ...],
    *,
    champion: bool,
) -> Path:
    artifact = root / model_id
    artifact.mkdir(parents=True)
    digest = feature_list_hash(features)
    (artifact / "model.txt").write_text("fixture model\n", encoding="utf-8")
    atomic_write_json(
        artifact / "feature_list.json", {"features": list(features), "feature_hash": digest}
    )
    atomic_write_json(
        artifact / "metrics.json",
        {
            "validation": {"rank_ic": 0.01},
            "test": {"rank_ic": 0.01} if champion else {},
        },
    )
    atomic_write_json(
        artifact / "manifest.json",
        {
            "artifact_name": (
                "lightgbm_ranker_baseline" if champion else "lightgbm_ranker_challenger"
            ),
            "experiment_id": model_id,
            "creation_time": "2026-07-22T00:00:00+00:00",
            "feature_list_hash": digest,
            "train_start": "20200101",
            "train_end": "20231231",
        },
    )
    return artifact


def _filled_trade(code: str, side: str, gross_value: float, cost: float) -> dict[str, object]:
    return {
        "trade_date": "20240102",
        "ts_code": code,
        "side": side,
        "status": "filled",
        "gross_value": gross_value,
        "cost": cost,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
