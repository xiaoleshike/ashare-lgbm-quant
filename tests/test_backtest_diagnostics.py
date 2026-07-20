from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_quant.backtest.diagnostic_metrics import (
    assign_score_layers,
    daily_layer_returns,
    daily_prediction_ic,
    monthly_stability,
    summarize_ic,
    summarize_score_layers,
)
from ashare_quant.backtest.diagnostics import (
    BacktestDiagnosticEngine,
    BacktestDiagnosticResult,
)
from ashare_quant.cli import main
from ashare_quant.config.settings import (
    AppSettings,
    BacktestDiagnosticSettings,
    BacktestSettings,
    HistoricalBacktestSettings,
)
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.utils.manifest import atomic_write_json

RUN_ID = "historical_fixture"
MODEL_ID = "ranker_fixture"


class FakeBooster:
    def predict(self, matrix: pd.DataFrame) -> np.ndarray:
        return matrix["f1"].to_numpy(dtype=float)

    def feature_name(self) -> list[str]:
        return ["f1"]

    def feature_importance(self, importance_type: str) -> np.ndarray:
        return np.array([2.0 if importance_type == "gain" else 3])


def test_score_layers_and_rank_ic_use_frozen_full_cross_section() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20240102"] * 100,
            "ts_code": [f"{index:06d}.SZ" for index in range(100)],
            "prediction_score": np.arange(100, 0, -1, dtype=float),
            "rank": np.arange(1, 101),
            "cross_section_size": [100] * 100,
            "future_excess_ret": np.arange(100, 0, -1, dtype=float) / 1000,
        }
    )
    layered = assign_score_layers(frame, (0.01, 0.05, 0.10, 0.20), 0.20)

    assert (layered["layer"] == "top_1pct").sum() == 1
    assert (layered["layer"] == "bottom").sum() == 20
    daily_ic = daily_prediction_ic(frame, minimum_cross_section=20)
    assert daily_ic.iloc[0]["rank_ic"] == pytest.approx(1.0)
    assert summarize_ic(daily_ic)["positive_ic_ratio"] == 1.0


def test_missing_top_label_does_not_promote_lower_rank_into_score_layer() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20240102"] * 10,
            "ts_code": [f"{index:06d}.SZ" for index in range(10)],
            "prediction_score": np.arange(10, 0, -1, dtype=float),
            "rank": np.arange(1, 11),
            "cross_section_size": [10] * 10,
            "future_excess_ret": [np.nan, *([0.01] * 9)],
        }
    )

    layered = assign_score_layers(frame, (0.10,), 0.20)
    top = layered.loc[layered["layer"] == "top_10pct"]

    assert top["rank"].tolist() == [1]
    assert daily_layer_returns(layered).query("layer == 'top_10pct'").empty


def test_layer_metrics_and_monthly_stability_are_deterministic() -> None:
    dates = [f"202401{day:02d}" for day in range(2, 12)]
    frame = pd.DataFrame(
        {
            "trade_date": np.repeat(dates, 10),
            "ts_code": [f"{index % 10:06d}.SZ" for index in range(100)],
            "prediction_score": np.tile(np.arange(10, 0, -1), 10),
            "rank": np.tile(np.arange(1, 11), 10),
            "future_excess_ret": np.tile(np.linspace(0.05, -0.05, 10), 10),
        }
    )
    layered = assign_score_layers(frame, (0.10,), 0.20)
    daily = daily_layer_returns(layered)
    score_summary = summarize_score_layers(daily, horizon=5, annualization_days=252)
    ic = daily_prediction_ic(frame, minimum_cross_section=5)
    monthly = monthly_stability(daily, ic)

    assert next(row for row in score_summary if row["layer"] == "top_10pct")[
        "mean_forward_excess_return"
    ] == pytest.approx(0.05)
    top_month = monthly.loc[monthly["layer"] == "top_10pct"].iloc[0]
    assert top_month["ic"] == pytest.approx(1.0)
    assert top_month["win_rate"] == 1.0


def test_diagnostics_publish_complete_read_only_artifacts(tmp_path: Path, monkeypatch) -> None:
    engine = diagnostic_fixture(tmp_path)
    prediction_bytes = (
        tmp_path / "reports" / "backtest" / RUN_ID / "predictions.parquet"
    ).read_bytes()
    label_path = next((tmp_path / "processed" / "labels_forward").glob("**/*.parquet"))
    label_bytes = label_path.read_bytes()
    monkeypatch.setattr(
        "ashare_quant.backtest.diagnostics.load_model_and_features",
        lambda path: (FakeBooster(), ("f1",), feature_list_hash(("f1",))),
    )
    monkeypatch.setattr(
        "ashare_quant.backtest.diagnostics.shap_importance",
        lambda model, sample, features, prediction_tolerance: (
            [{"feature": "f1", "mean_abs_shap": 1.0, "mean_shap": 0.0}],
            "fixture_shap",
            0.0,
        ),
    )

    result = engine.run(RUN_ID)

    assert result.prediction_rows == 12
    assert result.labelled_rows == 12
    assert result.ic_days == 2
    assert set(path.name for path in result.output_dir.iterdir()) == {
        "daily_ic.csv",
        "score_layer_returns.csv",
        "monthly_stability.csv",
        "single_factor_groups.csv",
        "summary.json",
        "diagnostics_report.md",
        "manifest.json",
    }
    summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["scientific_scope"]["labels_used_for_training"] is False
    assert summary["factor_attribution"]["shap_method"] == "fixture_shap"
    manifest = json.loads((result.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_backtest_manifest"]["model_id"] == MODEL_ID
    assert (
        tmp_path / "reports" / "backtest" / RUN_ID / "predictions.parquet"
    ).read_bytes() == prediction_bytes
    assert label_path.read_bytes() == label_bytes


def test_diagnostics_reject_old_run_without_full_predictions(tmp_path: Path) -> None:
    engine = diagnostic_fixture(tmp_path)
    (tmp_path / "reports" / "backtest" / RUN_ID / "predictions.parquet").unlink()

    with pytest.raises(DataValidationError, match="lacks predictions.parquet"):
        engine.run(RUN_ID)


def test_diagnostics_cli_success_and_failure(tmp_path: Path, monkeypatch, capsys) -> None:
    class SuccessfulEngine:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, run_id: str) -> BacktestDiagnosticResult:
            return BacktestDiagnosticResult(run_id, tmp_path / "output", 12, 11, 2)

    monkeypatch.setattr("ashare_quant.cli.BacktestDiagnosticEngine", SuccessfulEngine)
    arguments = [
        "--config",
        "config/default.yaml",
        "backtest",
        "diagnostics",
        "--run-id",
        RUN_ID,
    ]
    assert main(arguments) == 0
    assert "backtest_diagnostics: run_id=historical_fixture" in capsys.readouterr().out

    class FailingEngine(SuccessfulEngine):
        def run(self, run_id: str) -> BacktestDiagnosticResult:
            raise DataValidationError("missing predictions")

    monkeypatch.setattr("ashare_quant.cli.BacktestDiagnosticEngine", FailingEngine)
    assert main(arguments) == 2
    assert "backtest diagnostics failed" in capsys.readouterr().err


def diagnostic_fixture(tmp_path: Path) -> BacktestDiagnosticEngine:
    processed = tmp_path / "processed"
    backtest = tmp_path / "reports" / "backtest" / RUN_ID
    backtest.mkdir(parents=True)
    predictions = []
    labels = []
    features = []
    for trade_date in ("20240102", "20240103"):
        for rank in range(1, 7):
            code = f"{rank:06d}.SZ"
            score = float(7 - rank)
            predictions.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "prediction_score": score,
                    "rank": rank,
                    "selected_flag": rank <= 2,
                }
            )
            labels.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "horizon": 5,
                    "future_excess_ret": score / 100,
                    "is_label_available": True,
                }
            )
            features.append({"trade_date": trade_date, "ts_code": code, "f1": score})
    pd.DataFrame(predictions).to_parquet(backtest / "predictions.parquet", index=False)
    label_dir = processed / "labels_forward" / "year=2024" / "month=01"
    feature_dir = processed / "features_daily" / "year=2024" / "month=01"
    label_dir.mkdir(parents=True)
    feature_dir.mkdir(parents=True)
    pd.DataFrame(labels).to_parquet(label_dir / "data.parquet", index=False)
    pd.DataFrame(features).to_parquet(feature_dir / "data.parquet", index=False)
    digest = feature_list_hash(("f1",))
    model_dir = tmp_path / "models" / MODEL_ID
    model_dir.mkdir(parents=True)
    atomic_write_json(model_dir / "feature_list.json", {"features": ["f1"], "feature_hash": digest})
    (model_dir / "model.txt").write_text("fixture\n", encoding="utf-8")
    atomic_write_json(
        backtest / "manifest.json",
        {
            "artifact_name": "historical_champion_backtest",
            "model_id": MODEL_ID,
            "model_artifact": str(model_dir),
            "feature_hash": digest,
            "out_of_sample": True,
            "backtest_config": {"historical": {"holding_period_days": 5}},
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project_name: fixture\n", encoding="utf-8")
    settings = AppSettings(
        backtest=BacktestSettings(
            historical=HistoricalBacktestSettings(holding_period_days=5, periods={}),
            diagnostics=BacktestDiagnosticSettings(
                horizon=5,
                score_layers=(0.10, 0.20),
                bottom_fraction=0.20,
                minimum_cross_section=2,
                factor_quantiles=3,
                shap_sample_rows=12,
            ),
        )
    )
    return BacktestDiagnosticEngine(
        processed_root=processed,
        backtest_root=tmp_path / "reports" / "backtest",
        output_root=tmp_path / "reports" / "backtest_diagnostics",
        settings=settings,
        config_path=config_path,
    )
