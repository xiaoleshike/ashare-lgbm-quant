from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.cli import main
from ashare_quant.config.settings import DecisionSupportSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.research import DecisionSupportResult, InvestmentDecisionSupport
from ashare_quant.utils.manifest import atomic_write_json

AS_OF = "20240110"
MODEL_ID = "ranker_fixture"


def test_decision_support_preserves_candidate_ranking_and_reports_observations(
    tmp_path: Path,
) -> None:
    support, report_dir = decision_fixture(tmp_path)
    candidate_bytes = (report_dir / "candidates.csv").read_bytes()

    result = support.generate(AS_OF)

    assert result.candidate_count == 2
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert [stock["ts_code"] for stock in payload["stocks"]] == ["000002.SZ", "000001.SZ"]
    assert [stock["candidate_rank"] for stock in payload["stocks"]] == [1, 2]
    assert [stock["model_rank"] for stock in payload["stocks"]] == [2, 3]
    first = payload["stocks"][0]
    assert first["prediction_score"] == 3.0
    assert first["signal_strength"] == "strong"
    assert first["positive_contributions"][0]["feature"] == "market_excess_ret_120d"
    assert first["negative_contributions"][0]["feature"] == "amihud_20d"
    assert first["technical_state"]["ma20_status"] == "above_or_equal"
    statuses = {row["rule"]: row["status"] for row in first["watch_entry_conditions"]}
    assert statuses["open_gap_within_range"] == "met"
    assert statuses["amount_activity_sufficient"] == "met"
    risks = {row["rule"] for row in first["risk_observations"]}
    assert risks == {"short_return_elevated", "volatility_elevated"}
    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert "Human-review support only" in markdown
    assert "create orders" in markdown
    assert (report_dir / "candidates.csv").read_bytes() == candidate_bytes


def test_decision_support_is_deterministic_does_not_read_labels_or_future_rows(
    tmp_path: Path,
) -> None:
    support, report_dir = decision_fixture(tmp_path)
    labels = tmp_path / "processed" / "labels_forward"
    labels.mkdir(parents=True)
    (labels / "unreadable.parquet").write_bytes(b"not parquet")

    first = support.generate(AS_OF)
    first_json = first.json_path.read_bytes()
    first_markdown = first.markdown_path.read_bytes()
    payload = json.loads(first_json)
    first_stock = payload["stocks"][0]
    assert first_stock["technical_state"]["short_return"] == 0.2
    assert first_stock["technical_state"]["ma20_ratio"] == 0.03

    second = support.generate(AS_OF)

    assert second.json_path.read_bytes() == first_json
    assert second.markdown_path.read_bytes() == first_markdown
    assert report_dir.exists()


def test_decision_support_rejects_changed_score_or_rank(tmp_path: Path) -> None:
    support, report_dir = decision_fixture(tmp_path)
    candidates = pd.read_csv(report_dir / "candidates.csv")
    candidates.loc[0, "prediction_score"] = 9.0
    candidates.to_csv(report_dir / "candidates.csv", index=False)

    with pytest.raises(DataValidationError, match="score identity mismatch"):
        support.generate(AS_OF)

    candidates.loc[0, "prediction_score"] = 3.0
    candidates.loc[0, "rank"] = 9
    candidates.to_csv(report_dir / "candidates.csv", index=False)
    with pytest.raises(DataValidationError, match="rank differs"):
        support.generate(AS_OF)

    candidates.loc[0, "rank"] = 1
    candidates.to_csv(report_dir / "candidates.csv", index=False)
    explanations = json.loads((report_dir / "explanations.json").read_text(encoding="utf-8"))
    explanations["stocks"][0]["model_rank"] = 99
    atomic_write_json(report_dir / "explanations.json", explanations)
    with pytest.raises(DataValidationError, match="model rank differs"):
        support.generate(AS_OF)


def test_decision_support_rejects_mixed_input_dates(tmp_path: Path) -> None:
    support, report_dir = decision_fixture(tmp_path)
    candidates = pd.read_csv(report_dir / "candidates.csv", dtype={"trade_date": str})
    candidates["trade_date"] = "20240111"
    candidates.to_csv(report_dir / "candidates.csv", index=False)

    with pytest.raises(DataValidationError, match="date other than --as-of"):
        support.generate(AS_OF)


def test_decision_support_cli_success_and_failure(tmp_path: Path, monkeypatch, capsys) -> None:
    class SuccessfulSupport:
        def __init__(self, **kwargs: object) -> None:
            pass

        def generate(self, as_of: str) -> DecisionSupportResult:
            return DecisionSupportResult(
                as_of,
                MODEL_ID,
                2,
                tmp_path / "decision.json",
                tmp_path / "decision_report.md",
            )

    monkeypatch.setattr("ashare_quant.cli.InvestmentDecisionSupport", SuccessfulSupport)
    arguments = [
        "--config",
        "config/default.yaml",
        "research",
        "--storage-root",
        str(tmp_path / "raw"),
        "--processed-root",
        str(tmp_path / "processed"),
        "--reports-root",
        str(tmp_path / "reports"),
        "decision",
        "--as-of",
        AS_OF,
    ]

    assert main(arguments) == 0
    assert "decision_support: date=20240110 candidates=2" in capsys.readouterr().out

    class FailingSupport(SuccessfulSupport):
        def generate(self, as_of: str) -> DecisionSupportResult:
            raise DataValidationError("missing explanations")

    monkeypatch.setattr("ashare_quant.cli.InvestmentDecisionSupport", FailingSupport)
    assert main(arguments) == 2
    assert "investment decision support failed" in capsys.readouterr().err


def decision_fixture(tmp_path: Path) -> tuple[InvestmentDecisionSupport, Path]:
    report_dir = tmp_path / "reports" / AS_OF
    report_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "trade_date": [AS_OF] * 3,
            "ts_code": ["000003.SZ", "000002.SZ", "000001.SZ"],
            "prediction_score": [4.0, 3.0, 2.0],
            "model_id": [MODEL_ID] * 3,
        }
    ).to_parquet(report_dir / "predictions.parquet", index=False)
    pd.DataFrame(
        {
            "rank": [1, 2],
            "ts_code": ["000002.SZ", "000001.SZ"],
            "prediction_score": [3.0, 2.0],
            "selection_reason": ["passed_configured_filters"] * 2,
            "trade_date": [AS_OF] * 2,
            "model_id": [MODEL_ID] * 2,
        }
    ).to_csv(report_dir / "candidates.csv", index=False)
    atomic_write_json(report_dir / "explanations.json", _explanations())
    _write_inputs(tmp_path)
    return (
        InvestmentDecisionSupport(
            raw_root=tmp_path / "raw",
            processed_root=tmp_path / "processed",
            reports_root=tmp_path / "reports",
            settings=DecisionSupportSettings(),
        ),
        report_dir,
    )


def _explanations() -> dict[str, object]:
    stocks = []
    for code, model_rank, candidate_rank, score in (
        ("000002.SZ", 2, 1, 3.0),
        ("000001.SZ", 3, 2, 2.0),
    ):
        stocks.append(
            {
                "ts_code": code,
                "model_rank": model_rank,
                "candidate_rank": candidate_rank,
                "prediction_score": score,
                "signal_strength": "strong",
                "confidence": "medium",
                "positive_contributions": [
                    {
                        "feature": "market_excess_ret_120d",
                        "value": 0.1,
                        "shap": 0.2,
                        "description": "长期市场超额收益",
                    }
                ],
                "negative_contributions": [
                    {
                        "feature": "amihud_20d",
                        "value": 0.000001,
                        "shap": -0.1,
                        "description": "流动性压力",
                    }
                ],
            }
        )
    return {
        "schema_version": 1,
        "artifact_name": "daily_model_explanations",
        "as_of": AS_OF,
        "model_id": MODEL_ID,
        "feature_hash": "fixture-feature-hash",
        "stocks": stocks,
    }


def _write_inputs(tmp_path: Path) -> None:
    feature_columns = {
        "trade_date": [AS_OF, AS_OF, "20240111", "20240111"],
        "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
        "gap_mean_1d": [-0.04, 0.01, 0.99, 0.99],
        "ma_ratio_20d": [-0.02, 0.03, 0.99, 0.99],
        "amount_ratio_20d": [0.5, 1.2, 9.0, 9.0],
        "amihud_20d": [0.00002, 0.000001, 0.0, 0.0],
        "ret_5d": [0.02, 0.2, 9.0, 9.0],
        "realized_vol_20d": [0.02, 0.05, 9.0, 9.0],
    }
    _write_partition(tmp_path / "processed", "features_daily", pd.DataFrame(feature_columns))
    daily_basic = pd.DataFrame(
        {
            "trade_date": [AS_OF, AS_OF, "20240111", "20240111"],
            "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
            "turnover_rate": [0.3, 1.2, 99.0, 99.0],
            "total_mv": [800_000.0, 2_000_000.0, 99.0, 99.0],
        }
    )
    _write_partition(tmp_path / "raw", "daily_basic", daily_basic)


def _write_partition(root: Path, dataset: str, frame: pd.DataFrame) -> None:
    directory = root / dataset / "year=2024" / "month=01"
    directory.mkdir(parents=True)
    frame.to_parquet(directory / "data.parquet", index=False)
