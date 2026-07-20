from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ashare_quant.cli import main
from ashare_quant.config import load_settings
from ashare_quant.config.settings import CandidateSelectionSettings
from ashare_quant.strategy import CandidateSelector
from ashare_quant.utils.manifest import atomic_write_json, config_hash

AS_OF = "20240110"
MODEL_ID = "ranker_champion"
STOCKS = tuple(f"00000{index}.SZ" for index in range(1, 8))


def test_configured_filters_and_deterministic_ranking(tmp_path: Path) -> None:
    selector = candidate_fixture(tmp_path)

    result = selector.select(AS_OF)

    assert result.candidates["ts_code"].tolist() == ["000001.SZ", "000002.SZ"]
    assert result.candidates["rank"].tolist() == [1, 2]
    assert result.candidates["selection_reason"].eq("passed_configured_filters").all()
    assert result.filtered_counts == {
        "abnormal_price_limits": 1,
        "insufficient_liquidity": 1,
        "newly_listed": 1,
        "st": 1,
        "suspended": 1,
    }


def test_candidate_count_limit_preserves_score_order_and_ts_code_ties(tmp_path: Path) -> None:
    selector = candidate_fixture(
        tmp_path,
        settings=CandidateSelectionSettings(
            max_candidates=1,
            exclude_st=False,
            exclude_suspended=False,
            exclude_low_liquidity=False,
            exclude_bj_market=False,
            min_list_trading_days=0,
            min_total_mv=None,
            min_daily_amount=None,
            require_valid_price_limits=False,
        ),
    )

    result = selector.select(AS_OF)

    assert result.candidates["ts_code"].tolist() == ["000001.SZ"]
    assert result.filtered_counts == {"below_candidate_cutoff": 6}


def test_bj_market_is_excluded_while_star_and_chinext_remain_configurable(
    tmp_path: Path,
) -> None:
    selector = candidate_fixture(tmp_path)
    _replace_ts_code(tmp_path, "000002.SZ", "920225.BJ")

    result = selector.select(AS_OF)

    assert "920225.BJ" not in set(result.candidates["ts_code"])
    assert result.filtered_counts["bj_market"] == 1

    selector = candidate_fixture(tmp_path / "boards")
    _replace_ts_code(tmp_path / "boards", "000001.SZ", "688001.SH")
    _replace_ts_code(tmp_path / "boards", "000002.SZ", "300001.SZ")
    board_result = selector.select(AS_OF)
    assert board_result.candidates["ts_code"].tolist() == ["300001.SZ", "688001.SH"]


def test_market_cap_filter_excludes_below_configured_threshold(tmp_path: Path) -> None:
    selector = candidate_fixture(tmp_path)
    path = next((tmp_path / "raw" / "daily_basic").glob("**/*.parquet"))
    frame = pd.read_parquet(path)
    frame.loc[frame["ts_code"].eq("000002.SZ"), "total_mv"] = 499_999.0
    frame.to_parquet(path, index=False)

    result = selector.select(AS_OF)

    assert result.candidates["ts_code"].tolist() == ["000001.SZ"]
    assert result.filtered_counts["insufficient_market_cap"] == 1


def test_daily_amount_filter_excludes_illiquid_stock(tmp_path: Path) -> None:
    selector = candidate_fixture(tmp_path)
    path = next((tmp_path / "raw" / "daily").glob("**/*.parquet"))
    frame = pd.read_parquet(path)
    frame.loc[frame["ts_code"].eq("000001.SZ"), "amount"] = 29_999.0
    frame.to_parquet(path, index=False)

    result = selector.select(AS_OF)

    assert result.candidates["ts_code"].tolist() == ["000002.SZ"]
    assert result.filtered_counts["insufficient_daily_amount"] == 1


def test_empty_candidate_output_has_stable_schema(tmp_path: Path) -> None:
    selector = candidate_fixture(tmp_path)
    universe_path = next((tmp_path / "processed" / "universe_daily").glob("**/*.parquet"))
    universe = pd.read_parquet(universe_path)
    universe["is_st"] = True
    universe.to_parquet(universe_path, index=False)

    result = selector.select(AS_OF)

    assert result.candidate_count == 0
    assert list(result.candidates.columns) == [
        "rank",
        "ts_code",
        "prediction_score",
        "selection_reason",
        "trade_date",
        "model_id",
    ]
    persisted = pd.read_csv(result.output_path)
    assert persisted.empty
    assert list(persisted.columns) == list(result.candidates.columns)


def test_candidate_manifest_and_summary_preserve_prediction_provenance(tmp_path: Path) -> None:
    selector = candidate_fixture(tmp_path)

    result = selector.select(AS_OF)
    report_dir = result.output_path.parent
    manifest = json.loads((report_dir / "candidates_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))

    assert manifest["artifact_name"] == "production_candidates"
    assert manifest["model_id"] == MODEL_ID
    assert manifest["feature_hash"] == "feature-hash"
    assert manifest["config_hash"] == config_hash(selector.config_path)
    assert manifest["prediction_manifest"]["artifact_name"] == "production_predictions"
    assert manifest["filtering_rules"]["max_candidates"] == 50
    assert manifest["filtering_rules"]["exclude_bj_market"] is True
    assert manifest["filtering_rules"]["min_total_mv"] == 500_000.0
    assert manifest["filtering_rules"]["min_daily_amount"] == 30_000.0
    assert manifest["filtered_counts"] == result.filtered_counts
    assert summary["existing_inference_field"] == "preserved"
    assert summary["prediction_count"] == 7
    assert summary["candidate_count"] == 2
    assert summary["filtered_counts"] == result.filtered_counts
    assert (report_dir / "candidate.csv").read_bytes() == (
        report_dir / "candidates.csv"
    ).read_bytes()


def test_default_config_declares_candidate_rules() -> None:
    settings = load_settings("config/default.yaml").strategy.candidate_selection

    assert settings.max_candidates == 50
    assert settings.exclude_st
    assert settings.exclude_suspended
    assert settings.exclude_low_liquidity
    assert settings.exclude_bj_market
    assert not settings.exclude_star_market
    assert not settings.exclude_chinext_market
    assert settings.min_list_trading_days == 180
    assert settings.min_total_mv == 500_000.0
    assert settings.min_daily_amount == 30_000.0
    assert settings.require_valid_price_limits


def test_strategy_candidates_cli_success_and_failure(tmp_path: Path, capsys) -> None:
    candidate_fixture(tmp_path)
    arguments = [
        "--config",
        "config/default.yaml",
        "strategy",
        "--storage-root",
        str(tmp_path / "raw"),
        "--processed-root",
        str(tmp_path / "processed"),
        "--reports-root",
        str(tmp_path / "reports"),
        "candidates",
        "--as-of",
        AS_OF,
    ]

    assert main(arguments) == 0
    output = capsys.readouterr().out
    assert "strategy_candidates: date=20240110 candidates=2" in output
    assert output.rstrip().endswith("candidates.csv")

    assert main([*arguments[:-1], "20240111"]) == 2
    assert "strategy candidate selection failed" in capsys.readouterr().err


def candidate_fixture(
    tmp_path: Path,
    *,
    settings: CandidateSelectionSettings | None = None,
) -> CandidateSelector:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    reports_root = tmp_path / "reports"
    config_path = tmp_path / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("project_name: candidate-test\n", encoding="utf-8")
    scores = [0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    report_dir = reports_root / AS_OF
    report_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "trade_date": [AS_OF] * len(STOCKS),
            "ts_code": STOCKS,
            "prediction_score": scores,
            "model_id": [MODEL_ID] * len(STOCKS),
        }
    ).to_parquet(report_dir / "predictions.parquet", index=False)
    atomic_write_json(
        report_dir / "manifest.json",
        {
            "artifact_name": "production_predictions",
            "as_of": AS_OF,
            "model_id": MODEL_ID,
            "feature_hash": "feature-hash",
        },
    )
    atomic_write_json(
        report_dir / "summary.json",
        {"as_of": AS_OF, "prediction_count": len(STOCKS), "existing_inference_field": "preserved"},
    )

    _write_partition(
        processed_root,
        "universe_daily",
        pd.DataFrame(
            {
                "trade_date": [AS_OF] * len(STOCKS),
                "ts_code": STOCKS,
                "is_st": [False, False, True, False, False, False, False],
                "is_suspended": [False, False, False, True, False, False, False],
                "in_model_universe": [True] * len(STOCKS),
                "is_low_liquidity": [False, False, False, False, True, False, False],
                "list_days": [300, 300, 300, 300, 300, 20, 300],
            }
        ),
    )
    _write_partition(
        raw_root,
        "daily",
        pd.DataFrame(
            {
                "trade_date": [AS_OF] * len(STOCKS),
                "ts_code": STOCKS,
                "open": [10.0] * len(STOCKS),
                "high": [10.5] * len(STOCKS),
                "low": [9.5] * len(STOCKS),
                "close": [10.0] * len(STOCKS),
                "amount": [100_000.0] * len(STOCKS),
            }
        ),
    )
    _write_partition(
        raw_root,
        "daily_basic",
        pd.DataFrame(
            {
                "trade_date": [AS_OF] * len(STOCKS),
                "ts_code": STOCKS,
                "turnover_rate": [2.0] * len(STOCKS),
                "total_mv": [1_000_000.0] * len(STOCKS),
            }
        ),
    )
    _write_partition(
        raw_root,
        "stk_limit",
        pd.DataFrame(
            {
                "trade_date": [AS_OF] * len(STOCKS),
                "ts_code": STOCKS,
                "up_limit": [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 0.0],
                "down_limit": [9.0] * len(STOCKS),
            }
        ),
    )
    return CandidateSelector(
        raw_root=raw_root,
        processed_root=processed_root,
        reports_root=reports_root,
        config_path=config_path,
        settings=settings or CandidateSelectionSettings(),
    )


def _write_partition(root: Path, dataset: str, frame: pd.DataFrame) -> None:
    directory = root / dataset / "year=2024" / "month=01"
    directory.mkdir(parents=True)
    frame.to_parquet(directory / "data.parquet", index=False)


def _replace_ts_code(root: Path, old: str, new: str) -> None:
    for path in root.glob("**/*.parquet"):
        frame = pd.read_parquet(path)
        if "ts_code" not in frame.columns:
            continue
        frame.loc[frame["ts_code"].astype(str).eq(old), "ts_code"] = new
        frame.to_parquet(path, index=False)
