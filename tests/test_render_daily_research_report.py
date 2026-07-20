from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from subprocess import run

SCRIPT = Path("scripts/reports/render_daily_research_report.py")


def test_render_daily_research_report_creates_readable_html(tmp_path: Path) -> None:
    report_dir = _report_fixture(tmp_path)

    completed = run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPT),
            "--as-of",
            "20240110",
            "--reports-root",
            str(tmp_path / "reports"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    output = report_dir / "daily_report.html"
    html = output.read_text(encoding="utf-8")
    assert "A股量化研究日报" in html
    assert "ranker_fixture" in html
    assert "000001.SZ" in html
    assert "当日涨跌幅异常" in html
    assert "板块分布" in html
    assert "原始 Markdown 报告" in html


def test_render_daily_research_report_fails_for_missing_inputs(tmp_path: Path) -> None:
    completed = run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPT),
            "--as-of",
            "20240110",
            "--reports-root",
            str(tmp_path / "reports"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "required research summary is missing" in completed.stdout


def _report_fixture(tmp_path: Path) -> Path:
    report_dir = tmp_path / "reports" / "20240110"
    report_dir.mkdir(parents=True)
    summary = {
        "as_of": "20240110",
        "model_id": "ranker_fixture",
        "prediction_count": 100,
        "candidate_count": 1,
        "top_candidate_count": 1,
        "statistics": {
            "board_distribution": {"Shenzhen Main": 1},
            "market_cap_distribution": {"5bn_to_10bn_cny": 1},
            "industry_distribution": {"Bank": 1},
        },
        "risk_flags": [
            {
                "rank": 1,
                "ts_code": "000001.SZ",
                "flags": ["abnormal_recent_return"],
                "pct_chg": -9.5,
                "recent_volatility_pct": 4.2,
                "amount": 100000.0,
            }
        ],
        "warnings": [],
    }
    (report_dir / "research_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (report_dir / "candidates.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank",
                "ts_code",
                "prediction_score",
                "selection_reason",
                "trade_date",
                "model_id",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "rank": 1,
                "ts_code": "000001.SZ",
                "prediction_score": 0.8,
                "selection_reason": "passed_configured_filters",
                "trade_date": "20240110",
                "model_id": "ranker_fixture",
            }
        )
    (report_dir / "daily_report.md").write_text(
        "# Daily Quantitative Research Report\n", encoding="utf-8"
    )
    return report_dir
