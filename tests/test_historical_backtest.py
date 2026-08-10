from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_quant.backtest.data import load_scored_signals
from ashare_quant.backtest.engine import BacktestInputs
from ashare_quant.backtest.historical import (
    HistoricalBacktestEngine,
    HistoricalBacktestResult,
)
from ashare_quant.cli import main
from ashare_quant.config.settings import (
    AppSettings,
    BacktestSettings,
    HistoricalBacktestSettings,
)
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.utils.manifest import atomic_write_json

MODEL_ID = "ranker_fixture"


class FakeModel:
    def predict(self, matrix: pd.DataFrame) -> np.ndarray:
        return matrix["f1"].to_numpy(dtype=float)


def test_historical_backtest_publishes_required_deterministic_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    engine, inputs = historical_fixture(tmp_path, monkeypatch)

    first = engine.run(start_date="20200102", end_date="20200103", top_n=(1, 2))

    assert first.model_id == MODEL_ID
    assert set(path.name for path in first.output_dir.iterdir()) == {
        "summary.json",
        "backtest_report.md",
        "predictions.parquet",
        "daily_returns.parquet",
        "holdings.parquet",
        "manifest.json",
    }
    summary = json.loads((first.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["metrics"]) == {"1", "2"}
    assert summary["label_audit"]["labels_used_for_selection"] is False
    assert summary["label_audit"]["labels_used_for_returns"] is False
    assert summary["metrics"]["1"]["overall"]["holding_days"] == 2
    assert summary["metrics"]["1"]["yearly"][0]["regime"] in {
        "bull",
        "bear",
        "neutral",
    }
    manifest = json.loads((first.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_id"] == MODEL_ID
    assert manifest["candidate_config"]["universe_filter"] == "same_date_in_model_universe"
    assert manifest["out_of_sample"] is True
    assert manifest["prediction_file"] == "predictions.parquet"
    predictions = pd.read_parquet(first.output_dir / "predictions.parquet")
    assert len(predictions) == len(inputs.signals)
    assert predictions.duplicated(["trade_date", "ts_code"]).sum() == 0
    assert not inputs.signals.empty

    before = {path.name: path.read_bytes() for path in first.output_dir.iterdir()}
    second = engine.run(start_date="20200102", end_date="20200103", top_n=(1, 2))
    after = {path.name: path.read_bytes() for path in second.output_dir.iterdir()}
    assert second.run_id == first.run_id
    assert after == before


def test_label_returns_cannot_change_selection_or_backtest_returns(
    tmp_path: Path, monkeypatch
) -> None:
    engine, _ = historical_fixture(tmp_path, monkeypatch)
    first = engine.run(start_date="20200102", end_date="20200103", top_n=(1,))
    first_daily = pd.read_parquet(first.output_dir / "daily_returns.parquet")
    label_path = next((tmp_path / "processed" / "labels_forward").glob("**/*.parquet"))
    labels = pd.read_parquet(label_path)
    labels["stock_forward_ret"] = 999999.0
    labels["future_excess_ret"] = -999999.0
    labels.to_parquet(label_path, index=False)

    second = engine.run(start_date="20200102", end_date="20200103", top_n=(1,))

    pd.testing.assert_frame_equal(
        pd.read_parquet(second.output_dir / "daily_returns.parquet"), first_daily
    )


def test_historical_feature_scoring_reads_no_future_dates(tmp_path: Path) -> None:
    feature_dir = tmp_path / "processed" / "features_daily" / "year=2020" / "month=01"
    universe_dir = tmp_path / "processed" / "universe_daily" / "year=2020" / "month=01"
    feature_dir.mkdir(parents=True)
    universe_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "trade_date": ["20200102", "20200103", "20200106"],
            "ts_code": ["000001.SZ"] * 3,
            "f1": [1.0, 2.0, 999999.0],
        }
    ).to_parquet(feature_dir / "data.parquet", index=False)
    pd.DataFrame(
        {
            "trade_date": ["20200102", "20200103", "20200106"],
            "ts_code": ["000001.SZ"] * 3,
            "in_model_universe": [True, True, True],
        }
    ).to_parquet(universe_dir / "data.parquet", index=False)

    signals = load_scored_signals(
        tmp_path / "processed",
        FakeModel(),
        ("f1",),
        "20200102",
        "20200103",  # type: ignore[arg-type]
    )

    assert signals["trade_date"].astype(str).tolist() == ["20200102", "20200103"]
    assert signals["score"].tolist() == [1.0, 2.0]


def test_historical_backtest_rejects_non_chronological_predictions(
    tmp_path: Path, monkeypatch
) -> None:
    engine, inputs = historical_fixture(tmp_path, monkeypatch)
    inputs.signals.sort_values("trade_date", ascending=False, inplace=True, ignore_index=True)

    with pytest.raises(DataValidationError, match="not chronologically ordered"):
        engine.run(start_date="20200102", end_date="20200103", top_n=(1,))


def test_historical_backtest_rejects_in_sample_period(tmp_path: Path, monkeypatch) -> None:
    engine, _ = historical_fixture(tmp_path, monkeypatch)

    with pytest.raises(DataValidationError, match="BACKTEST_IN_SAMPLE_OVERLAP"):
        engine.run(start_date="20190101", end_date="20191231", top_n=(1,))


def test_historical_backtest_cli_success_and_failure(tmp_path: Path, monkeypatch, capsys) -> None:
    class SuccessfulEngine:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, **kwargs: object) -> HistoricalBacktestResult:
            return HistoricalBacktestResult(
                "run_fixture",
                tmp_path / "reports" / "backtest" / "run_fixture",
                MODEL_ID,
                "20200101",
                "20221231",
                {},
            )

    monkeypatch.setattr("ashare_quant.cli.HistoricalBacktestEngine", SuccessfulEngine)
    arguments = [
        "--config",
        "config/default.yaml",
        "backtest",
        "historical",
        "--period",
        "2020-2023",
    ]

    assert main(arguments) == 0
    assert "historical_backtest: run_id=run_fixture" in capsys.readouterr().out

    class FailingEngine(SuccessfulEngine):
        def run(self, **kwargs: object) -> HistoricalBacktestResult:
            raise DataValidationError("not OOS")

    monkeypatch.setattr("ashare_quant.cli.HistoricalBacktestEngine", FailingEngine)
    assert main(arguments) == 2
    assert "historical backtest failed: not OOS" in capsys.readouterr().err


def historical_fixture(
    tmp_path: Path, monkeypatch
) -> tuple[HistoricalBacktestEngine, BacktestInputs]:
    models_root = tmp_path / "models"
    artifact = _model_artifact(models_root)
    registry = ModelRegistry(models_root)
    registry.register_model(artifact)
    registry.promote_model(MODEL_ID)
    settings = AppSettings(
        backtest=BacktestSettings(
            initial_cash=1000.0,
            top_n=(1, 2),
            holding_period_days=2,
            annualization_days=252,
            historical=HistoricalBacktestSettings(
                top_n=(1, 2),
                holding_period_days=2,
                periods={},
            ),
        )
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project_name: fixture\n", encoding="utf-8")
    inputs = _inputs()
    monkeypatch.setattr(
        "ashare_quant.backtest.historical._effective_end_date",
        lambda processed, start, end, horizon: end,
    )
    monkeypatch.setattr(
        "ashare_quant.backtest.historical.load_model_and_features",
        lambda path: (object(), ("f1",), feature_list_hash(("f1",))),
    )
    monkeypatch.setattr(
        "ashare_quant.backtest.historical.load_backtest_inputs",
        lambda **kwargs: inputs,
    )
    _write_labels(tmp_path / "processed")
    return (
        HistoricalBacktestEngine(
            raw_root=tmp_path / "raw",
            processed_root=tmp_path / "processed",
            output_root=tmp_path / "reports" / "backtest",
            models_root=models_root,
            settings=settings,
            config_path=config_path,
        ),
        inputs,
    )


def _inputs() -> BacktestInputs:
    calendar = (
        "20200102",
        "20200103",
        "20200106",
        "20200107",
        "20200108",
        "20200109",
    )
    signals = pd.DataFrame(
        [
            {"trade_date": "20200102", "ts_code": "000001.SZ", "score": 0.9},
            {"trade_date": "20200102", "ts_code": "000002.SZ", "score": 0.8},
            {"trade_date": "20200103", "ts_code": "000001.SZ", "score": 0.7},
            {"trade_date": "20200103", "ts_code": "000002.SZ", "score": 0.6},
        ]
    )
    price_rows = []
    for date_index, date in enumerate(calendar):
        for code_index, code in enumerate(("000001.SZ", "000002.SZ")):
            price = 10.0 + date_index + code_index
            price_rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "open": price,
                    "close": price + 0.2,
                    "can_buy": True,
                    "can_sell": True,
                }
            )
    benchmark = pd.DataFrame(
        {"trade_date": list(calendar), "close": [100, 101, 102, 103, 104, 105]}
    )
    return BacktestInputs(signals, pd.DataFrame(price_rows), calendar, benchmark)


def _write_labels(processed_root: Path) -> None:
    directory = processed_root / "labels_forward" / "year=2020" / "month=01"
    directory.mkdir(parents=True)
    rows = []
    for trade_date, entry_date, exit_date in (
        ("20200102", "20200103", "20200107"),
        ("20200103", "20200106", "20200108"),
    ):
        for code in ("000001.SZ", "000002.SZ"):
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "horizon": 2,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "stock_forward_ret": 0.1,
                    "future_excess_ret": 0.05,
                    "is_label_available": True,
                    "label_unavailable_reason": "",
                }
            )
    pd.DataFrame(rows).to_parquet(directory / "data.parquet", index=False)


def _model_artifact(models_root: Path) -> Path:
    artifact = models_root / MODEL_ID
    artifact.mkdir(parents=True)
    digest = feature_list_hash(("f1",))
    (artifact / "model.txt").write_text("fixture\n", encoding="utf-8")
    atomic_write_json(artifact / "feature_list.json", {"features": ["f1"], "feature_hash": digest})
    atomic_write_json(
        artifact / "metrics.json",
        {"validation": {"rank_ic": 0.1}, "test": {"rank_ic": 0.1}},
    )
    atomic_write_json(
        artifact / "manifest.json",
        {
            "artifact_name": "lightgbm_ranker_baseline",
            "experiment_id": MODEL_ID,
            "completed_at": "2020-01-01T00:00:00+00:00",
            "git_commit": "fixture",
            "config_hash": "fixture",
            "feature_list_hash": digest,
            "train_start": "20150101",
            "train_end": "20191231",
        },
    )
    return artifact
