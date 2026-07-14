from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_quant.config.settings import AppSettings, DiagnosticSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.diagnostics.metrics import (
    daily_ic_table,
    greedy_correlation_prune,
    pairwise_correlations,
    summarize_ic,
)
from ashare_quant.diagnostics.model import choose_feature_count
from ashare_quant.diagnostics.pipeline import ChronologicalSplit, FeatureDiagnosticPipeline
from ashare_quant.features.registry import FEATURE_REGISTRY


def test_daily_cross_sectional_ic_and_coverage_summary() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20240102"] * 4 + ["20240103"] * 4,
            "feature_a": [1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0],
            "target": [2.0, 4.0, 6.0, 8.0, 1.0, 2.0, 3.0, 4.0],
        }
    )

    daily = daily_ic_table(frame, ["feature_a"], minimum_cross_section=3)
    summary = summarize_ic(daily, {"feature_a": 1.0}).set_index("feature")

    assert len(daily) == 2
    assert daily["pearson_ic"].tolist() == pytest.approx([1.0, 1.0])
    assert daily["rank_ic"].tolist() == pytest.approx([1.0, 1.0])
    assert summary.loc["feature_a", "coverage"] == 1.0
    assert summary.loc["feature_a", "positive_ic_ratio"] == 1.0


def test_pairwise_correlation_and_greedy_pruning_keep_first_ranked_feature() -> None:
    frame = pd.DataFrame(
        {
            "strong": [1.0, 2.0, 3.0, 4.0],
            "duplicate": [2.0, 4.0, 6.0, 8.0],
            "independent": [1.0, 0.0, 1.0, 0.0],
        }
    )

    correlations = pairwise_correlations(frame, ["strong", "duplicate", "independent"])
    kept, removed = greedy_correlation_prune(
        ["strong", "duplicate", "independent"], correlations, threshold=0.9
    )

    assert kept == ["strong", "independent"]
    assert removed.iloc[0]["removed_feature"] == "duplicate"
    assert removed.iloc[0]["kept_feature"] == "strong"


def test_candidate_count_choice_uses_balanced_metrics_not_single_return() -> None:
    validation = pd.DataFrame(
        {
            "set_name": ["top_30", "top_50"],
            "feature_count": [30, 50],
            "rank_ic": [0.04, 0.01],
            "rank_icir": [0.4, 0.1],
            "top_decile_excess_return": [0.002, 0.02],
            "sharpe": [1.2, 0.2],
            "maximum_drawdown": [-0.1, -0.4],
            "turnover": [0.3, 0.8],
            "year_positive_ratio": [1.0, 0.5],
        }
    )

    assert choose_feature_count(validation) == "top_30"


def test_chronological_split_rejects_overlap() -> None:
    split = ChronologicalSplit(
        train_start="20200101",
        train_end="20211231",
        validation_start="20210101",
        validation_end="20221231",
        test_start="20230101",
        test_end="20231231",
    )

    with pytest.raises(DataValidationError, match="chronological and non-overlapping"):
        split.validate()


def test_test_period_targets_cannot_change_recommended_feature_set(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project_name: test\n", encoding="utf-8")
    settings = diagnostic_test_settings()
    write_diagnostic_fixture(processed, invert_test_target=False)
    split = fixture_split()

    first = FeatureDiagnosticPipeline(processed, tmp_path / "reports_a", settings, config_path).run(
        split, horizon=5
    )
    write_diagnostic_fixture(processed, invert_test_target=True)
    second = FeatureDiagnosticPipeline(
        processed, tmp_path / "reports_b", settings, config_path
    ).run(split, horizon=5)

    assert first.recommended_features == second.recommended_features
    selection = json.loads(
        (first.report_dir / "recommended_features.json").read_text(encoding="utf-8")
    )
    assert selection["selection_uses_test_period"] is False


def test_diagnostics_status_reports_missing_run(tmp_path: Path, capsys) -> None:
    from ashare_quant.cli import main

    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "diagnostics",
            "--reports-root",
            str(tmp_path),
            "status",
        ]
    )

    assert exit_code == 0
    assert "feature_diagnostics: exists=False" in capsys.readouterr().out


def diagnostic_test_settings() -> AppSettings:
    return AppSettings(
        diagnostics=DiagnosticSettings(
            minimum_coverage=0.1,
            minimum_daily_cross_section=3,
            minimum_ic_days=2,
            correlation_threshold=0.95,
            model_sample_rows=1000,
            correlation_sample_rows=1000,
            candidate_feature_counts=(1, 2),
            lgbm_num_boost_round=3,
            lgbm_num_leaves=3,
            lgbm_min_data_in_leaf=1,
            lgbm_feature_fraction=1.0,
            lgbm_bagging_fraction=1.0,
            permutation_repeats=1,
        )
    )


def fixture_split() -> ChronologicalSplit:
    return ChronologicalSplit(
        train_start="20240102",
        train_end="20240104",
        validation_start="20240201",
        validation_end="20240203",
        test_start="20240301",
        test_end="20240303",
    )


def write_diagnostic_fixture(processed: Path, *, invert_test_target: bool) -> None:
    dates = [
        "20240102",
        "20240103",
        "20240104",
        "20240201",
        "20240202",
        "20240203",
        "20240301",
        "20240302",
        "20240303",
    ]
    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    universe_rows: list[dict[str, object]] = []
    for date_index, trade_date in enumerate(dates):
        for stock_index in range(8):
            code = f"{stock_index:06d}.SZ"
            row: dict[str, object] = {"trade_date": trade_date, "ts_code": code}
            for feature_index, spec in enumerate(FEATURE_REGISTRY):
                phase = (feature_index % 7) + 1
                row[spec.name] = (
                    stock_index * phase
                    + date_index * ((feature_index % 5) - 2)
                    + np.sin(stock_index + feature_index)
                )
            feature_rows.append(row)
            target = float(stock_index - 3.5)
            if trade_date.startswith("202403") and invert_test_target:
                target = -target
            label_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "horizon": 5,
                    "future_excess_ret": target / 100.0,
                    "benchmark_forward_ret": 0.001,
                    "is_label_available": True,
                }
            )
            universe_rows.append(
                {"trade_date": trade_date, "ts_code": code, "in_model_universe": True}
            )
    write_frame(processed / "features_daily" / "part.parquet", pd.DataFrame(feature_rows))
    write_frame(processed / "labels_forward" / "part.parquet", pd.DataFrame(label_rows))
    write_frame(processed / "universe_daily" / "part.parquet", pd.DataFrame(universe_rows))


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
