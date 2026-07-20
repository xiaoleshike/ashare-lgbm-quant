from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ashare_quant.cli import main
from ashare_quant.config.settings import DailyResearchReportSettings
from ashare_quant.research import DailyResearchReportGenerator

AS_OF = "20240110"
MODEL_ID = "ranker_champion"


def test_daily_report_generation_contains_rankings_statistics_and_risks(
    tmp_path: Path,
) -> None:
    generator = report_fixture(tmp_path)

    result = generator.generate(AS_OF)
    markdown = result.report_path.read_text(encoding="utf-8")
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert result.candidate_count == 2
    assert "## Market Summary" in markdown
    assert "## Candidate Ranking" in markdown
    assert "## Statistics" in markdown
    assert "## Risk Flags" in markdown
    assert "## Model Explanation" in markdown
    assert "This report describes model ranking only" in markdown
    assert markdown.index("`000001.SH`") < markdown.index("`300001.SZ`")
    assert summary["model_id"] == MODEL_ID
    assert summary["candidate_count"] == 2
    assert summary["statistics"]["board_distribution"] == {
        "ChiNext": 1,
        "Shanghai Main": 1,
    }
    assert summary["statistics"]["industry_distribution"] == {"Bank": 1, "Software": 1}
    risks = {row["ts_code"]: row["flags"] for row in summary["risk_flags"]}
    assert "abnormal_recent_return" in risks["000001.SH"]
    assert "high_volatility" in risks["000001.SH"]
    assert "low_liquidity" in risks["300001.SZ"]
    assert "limit_up" in risks["300001.SZ"]


def test_daily_report_is_byte_deterministic_for_same_inputs(tmp_path: Path) -> None:
    generator = report_fixture(tmp_path)

    first = generator.generate(AS_OF)
    first_markdown = first.report_path.read_bytes()
    first_summary = first.summary_path.read_bytes()
    second = generator.generate(AS_OF)

    assert second.report_path.read_bytes() == first_markdown
    assert second.summary_path.read_bytes() == first_summary


def test_missing_optional_market_data_produces_warnings_without_dropping_candidates(
    tmp_path: Path,
) -> None:
    generator = report_fixture(tmp_path)
    for directory in (
        tmp_path / "processed" / "universe_daily",
        tmp_path / "raw" / "daily_basic",
        tmp_path / "raw" / "daily",
    ):
        for path in directory.glob("**/*.parquet"):
            path.unlink()

    result = generator.generate(AS_OF)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert result.candidate_count == 2
    assert any("universe_daily" in warning for warning in result.warnings)
    assert any("daily_basic" in warning for warning in result.warnings)
    assert any("daily" in warning for warning in result.warnings)
    assert summary["statistics"]["industry_distribution"] == {"Unknown": 2}
    risks = {row["ts_code"]: set(row["flags"]) for row in summary["risk_flags"]}
    assert "missing_daily_data" in risks["000001.SH"]
    assert "missing_universe_metadata" in risks["000001.SH"]
    assert "missing_market_cap" in risks["000001.SH"]


def test_research_report_cli_success_and_missing_candidate_failure(tmp_path: Path, capsys) -> None:
    report_fixture(tmp_path)
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
        "report",
        "--as-of",
        AS_OF,
    ]

    assert main(arguments) == 0
    output = capsys.readouterr().out
    assert "daily_research_report: date=20240110 candidates=2" in output
    assert output.rstrip().endswith("daily_report.md")

    assert main([*arguments[:-1], "20240111"]) == 2
    assert "daily research report failed" in capsys.readouterr().err


def report_fixture(tmp_path: Path) -> DailyResearchReportGenerator:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    reports_root = tmp_path / "reports"
    report_dir = reports_root / AS_OF
    report_dir.mkdir(parents=True)
    candidates = pd.DataFrame(
        {
            "rank": [1, 2],
            "ts_code": ["000001.SH", "300001.SZ"],
            "prediction_score": [0.9, 0.8],
            "selection_reason": ["passed_configured_filters"] * 2,
            "trade_date": [AS_OF] * 2,
            "model_id": [MODEL_ID] * 2,
        }
    )
    candidates.to_csv(report_dir / "candidates.csv", index=False)
    pd.DataFrame(
        {
            "trade_date": [AS_OF] * 3,
            "ts_code": ["000001.SH", "300001.SZ", "600001.SH"],
            "prediction_score": [0.9, 0.8, 0.7],
            "model_id": [MODEL_ID] * 3,
        }
    ).to_parquet(report_dir / "predictions.parquet", index=False)
    _write_partition(
        processed_root,
        "universe_daily",
        pd.DataFrame(
            {
                "trade_date": [AS_OF, AS_OF],
                "ts_code": ["000001.SH", "300001.SZ"],
                "market": ["Main Board", "ChiNext"],
                "industry": ["Bank", "Software"],
                "is_limit_up": [False, True],
            }
        ),
    )
    _write_partition(
        raw_root,
        "daily_basic",
        pd.DataFrame(
            {
                "trade_date": [AS_OF, AS_OF],
                "ts_code": ["000001.SH", "300001.SZ"],
                "total_mv": [2_000_000.0, 800_000.0],
            }
        ),
    )
    dates = pd.bdate_range(end="2024-01-10", periods=20).strftime("%Y%m%d").tolist()
    daily_rows: list[dict[str, object]] = []
    for index, trade_date in enumerate(dates):
        daily_rows.append(
            {
                "trade_date": trade_date,
                "ts_code": "000001.SH",
                "pct_chg": 10.0 if trade_date == AS_OF else (8.0 if index % 2 else -8.0),
                "amount": 100_000.0,
            }
        )
        daily_rows.append(
            {
                "trade_date": trade_date,
                "ts_code": "300001.SZ",
                "pct_chg": 1.0,
                "amount": 40_000.0,
            }
        )
    _write_partition(raw_root, "daily", pd.DataFrame(daily_rows))
    return DailyResearchReportGenerator(
        raw_root=raw_root,
        processed_root=processed_root,
        reports_root=reports_root,
        settings=DailyResearchReportSettings(),
    )


def _write_partition(root: Path, dataset: str, frame: pd.DataFrame) -> None:
    directory = root / dataset / "year=2024" / "month=01"
    directory.mkdir(parents=True)
    frame.to_parquet(directory / "data.parquet", index=False)
