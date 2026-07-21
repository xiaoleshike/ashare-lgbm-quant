from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts.predict_stock_outlook import main as script_main

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.models.stock_outlook import StockOutlookPredictor, StockOutlookResult
from ashare_quant.utils.manifest import atomic_write_json

AS_OF = "20260105"


class FakeModel:
    def predict(self, matrix: pd.DataFrame) -> np.ndarray:
        return matrix["f1"].to_numpy(dtype=float)


def test_horizon_10_model_returns_cross_sectional_relative_outlook(tmp_path: Path) -> None:
    predictor = outlook_fixture(tmp_path, horizon=10)

    result = predictor.predict(
        model_id="ranker_h10",
        ts_code="000003.sz",
        as_of=AS_OF,
    )

    assert result.ts_code == "000003.SZ"
    assert result.target == "future_excess_ret_10d"
    assert result.horizon_trading_days == 10
    assert result.entry_date == "20260106"
    assert result.exit_date == "20260116"
    assert result.rank == 1
    assert result.universe_size == 3
    assert result.score_percentile == 1.0
    assert result.relative_outlook == "very_strong_relative"
    assert "not an absolute price path" in result.interpretation


def test_horizon_5_model_cannot_claim_a_10_day_outlook(tmp_path: Path) -> None:
    predictor = outlook_fixture(tmp_path, horizon=5)

    with pytest.raises(DataValidationError, match="model_horizon=5 requested_horizon=10"):
        predictor.predict(model_id="ranker_h5", ts_code="000003.SZ", as_of=AS_OF)


def test_stock_outside_model_universe_is_rejected(tmp_path: Path) -> None:
    predictor = outlook_fixture(tmp_path, horizon=10)

    with pytest.raises(DataValidationError, match="not in_model_universe"):
        predictor.predict(model_id="ranker_h10", ts_code="000004.SZ", as_of=AS_OF)


def test_outlook_requires_sufficient_future_trade_calendar(tmp_path: Path) -> None:
    predictor = outlook_fixture(tmp_path, horizon=10, calendar_sessions=5)

    with pytest.raises(DataValidationError, match="lacks 11 future open sessions"):
        predictor.predict(model_id="ranker_h10", ts_code="000003.SZ", as_of=AS_OF)


def test_outlook_does_not_require_or_read_labels(tmp_path: Path) -> None:
    predictor = outlook_fixture(tmp_path, horizon=10)
    assert not (tmp_path / "processed" / "labels_forward").exists()

    result = predictor.predict(model_id="ranker_h10", ts_code="000001.SZ", as_of=AS_OF)

    assert result.rank == 3


def test_standalone_script_outputs_json(tmp_path: Path, monkeypatch, capsys) -> None:
    result = StockOutlookResult(
        as_of=AS_OF,
        ts_code="000001.SZ",
        model_id="ranker_h10",
        model_status="candidate",
        target="future_excess_ret_10d",
        horizon_trading_days=10,
        entry_date="20260106",
        exit_date="20260116",
        prediction_score=0.5,
        rank=2,
        universe_size=100,
        score_percentile=0.99,
        relative_outlook="very_strong_relative",
        interpretation="relative only",
    )

    class FakePredictor:
        def __init__(self, **kwargs: object) -> None:
            pass

        def predict(self, **kwargs: object) -> StockOutlookResult:
            return result

    monkeypatch.setattr("scripts.predict_stock_outlook.StockOutlookPredictor", FakePredictor)
    assert (
        script_main(
            [
                "--config",
                "config/default.yaml",
                "--model-id",
                "ranker_h10",
                "--ts-code",
                "000001.SZ",
                "--as-of",
                AS_OF,
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "future_excess_ret_10d"
    assert payload["rank"] == 2


def outlook_fixture(
    tmp_path: Path,
    *,
    horizon: int,
    calendar_sessions: int = 11,
) -> StockOutlookPredictor:
    model_id = f"ranker_h{horizon}"
    models_root = tmp_path / "models"
    artifact = models_root / model_id
    artifact.mkdir(parents=True)
    digest = feature_list_hash(("f1",))
    (artifact / "model.txt").write_text("fixture\n", encoding="utf-8")
    atomic_write_json(
        artifact / "feature_list.json",
        {"features": ["f1"], "feature_hash": digest},
    )
    atomic_write_json(
        artifact / "metrics.json",
        {"validation": {"rank_ic": 0.1}, "test": {"rank_ic": 0.1}},
    )
    atomic_write_json(
        artifact / "manifest.json",
        {
            "artifact_name": "lightgbm_ranker_baseline",
            "experiment_id": model_id,
            "completed_at": "2026-01-01T00:00:00+00:00",
            "git_commit": "fixture",
            "config_hash": "fixture",
            "feature_list_hash": digest,
            "label_horizon": horizon,
            "target": f"future_excess_ret_{horizon}d",
            "train_start": "20200101",
            "train_end": "20251231",
        },
    )
    ModelRegistry(models_root).register_model(artifact)

    processed = tmp_path / "processed"
    feature_dir = processed / "features_daily" / "year=2026"
    universe_dir = processed / "universe_daily" / "year=2026"
    feature_dir.mkdir(parents=True)
    universe_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "trade_date": [AS_OF] * 4,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
            "f1": [1.0, 2.0, 3.0, 100.0],
        }
    ).to_parquet(feature_dir / "data.parquet", index=False)
    pd.DataFrame(
        {
            "trade_date": [AS_OF] * 4,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
            "in_model_universe": [True, True, True, False],
        }
    ).to_parquet(universe_dir / "data.parquet", index=False)

    raw = tmp_path / "raw"
    calendar_dir = raw / "trade_cal" / "year=2026"
    calendar_dir.mkdir(parents=True)
    all_dates = [f"202601{day:02d}" for day in range(6, 17)]
    pd.DataFrame(
        {
            "cal_date": all_dates[:calendar_sessions],
            "is_open": [1] * calendar_sessions,
        }
    ).to_parquet(calendar_dir / "data.parquet", index=False)
    return StockOutlookPredictor(
        raw_root=raw,
        processed_root=processed,
        models_root=models_root,
        model_loader=lambda path: FakeModel(),
    )
