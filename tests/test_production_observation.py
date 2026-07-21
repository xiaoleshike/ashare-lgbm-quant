from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ashare_quant.cli import main
from ashare_quant.models.production_observation import ProductionObservationRecorder

AS_OF = "20260717"
MODEL_ID = "champion_fixture"


def test_production_observation_is_deterministic_and_reserves_future_returns(
    tmp_path: Path,
) -> None:
    reports_root = tmp_path / "reports"
    _write_report_inputs(reports_root)
    recorder = ProductionObservationRecorder(reports_root)

    first = recorder.record(AS_OF)
    first_bytes = first.output_path.read_bytes()
    second = recorder.record(AS_OF)
    payload = json.loads(second.output_path.read_text(encoding="utf-8"))

    assert second.output_path.read_bytes() == first_bytes
    assert payload["model_id"] == MODEL_ID
    assert payload["prediction_date"] == AS_OF
    assert payload["candidate_count"] == 55
    assert len(payload["top10_rank"]) == 10
    assert len(payload["top20_rank"]) == 20
    assert len(payload["top50_rank"]) == 50
    assert payload["top10_rank"][0]["ts_code"] == "000001.SZ"
    assert payload["future_returns"] == {
        "status": "pending",
        "5d": None,
        "10d": None,
        "20d": None,
        "60d": None,
    }
    assert payload["constraints"] == {
        "future_returns_calculated": False,
        "orders_generated": False,
        "trading_signal_generated": False,
    }
    assert not (tmp_path / "processed" / "labels_forward").exists()


def test_production_observation_cli(tmp_path: Path, capsys) -> None:
    reports_root = tmp_path / "reports"
    _write_report_inputs(reports_root)

    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "models",
            "--reports-root",
            str(reports_root),
            "observation-log",
            "--as-of",
            AS_OF,
        ]
    )

    assert exit_code == 0
    assert "production_observation:" in capsys.readouterr().out
    assert (reports_root / "production_observation" / f"{AS_OF}.json").is_file()


def _write_report_inputs(reports_root: Path) -> None:
    report_dir = reports_root / AS_OF
    report_dir.mkdir(parents=True)
    rows = [
        {
            "trade_date": AS_OF,
            "ts_code": f"{index:06d}.SZ",
            "prediction_score": 1.0 - index / 1000,
            "model_id": MODEL_ID,
        }
        for index in range(1, 61)
    ]
    pd.DataFrame(rows).to_parquet(report_dir / "predictions.parquet", index=False)
    pd.DataFrame(
        [
            {
                "rank": index,
                "ts_code": f"{index:06d}.SZ",
                "prediction_score": 1.0 - index / 1000,
                "selection_reason": "eligible",
                "trade_date": AS_OF,
                "model_id": MODEL_ID,
            }
            for index in range(1, 56)
        ]
    ).to_csv(report_dir / "candidates.csv", index=False)
