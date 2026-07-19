from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.datasets import get_dataset_spec
from ashare_quant.data.storage import ParquetDataStore
from ashare_quant.features.storage import FeatureStore
from ashare_quant.orchestration.freshness import FreshnessService
from ashare_quant.universe.storage import UNIVERSE_COLUMNS, UniverseStore
from ashare_quant.utils.manifest import (
    parquet_artifact_statistics,
    write_build_manifest,
)

DATES = ("20240102", "20240103", "20240104", "20240105", "20240108", "20240109", "20240110")
AS_OF = DATES[-1]


def test_explicit_open_completed_session_and_incomplete_session(tmp_path: Path) -> None:
    service = freshness_fixture(tmp_path)

    completed = service.check_session(AS_OF)
    incomplete = freshness_fixture(
        tmp_path / "incomplete",
        now=datetime(2024, 1, 10, 14, 30, tzinfo=ZoneInfo("Asia/Shanghai")),
    ).check_session(AS_OF)

    assert completed.ready
    assert not incomplete.ready
    assert any("future or incomplete" in failure for failure in incomplete.hard_failures)


def test_hard_raw_dataset_stale_by_one_session_fails(tmp_path: Path) -> None:
    result = freshness_fixture(tmp_path, stale_daily=True).check_raw(AS_OF)

    assert not result.ready
    assert any("daily" in failure and "as-of" in failure for failure in result.hard_failures)


def test_missing_benchmark_entity_fails(tmp_path: Path) -> None:
    result = freshness_fixture(tmp_path, missing_benchmark=True).check_raw(AS_OF)

    assert not result.ready
    assert any("benchmark index entities missing" in failure for failure in result.hard_failures)


def test_legitimate_empty_suspend_is_explicit_and_optional_financial_data_warns(
    tmp_path: Path,
) -> None:
    result = freshness_fixture(tmp_path).check_raw(AS_OF)

    assert result.ready
    assert result.details["legitimate_empty_sessions"]["suspend_d"] == "explicit_empty_marker"
    assert any(
        "optional low-frequency dataset is missing" in warning for warning in result.warnings
    )


def test_stale_optional_financial_dataset_warns_without_failure(tmp_path: Path) -> None:
    service = freshness_fixture(tmp_path)
    service.settings.production.freshness.soft_dataset_max_lag_calendar_days = {"income": 1}
    service.raw_store.write(
        get_dataset_spec("income"),
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["20230101"],
                "f_ann_date": ["20230102"],
                "end_date": ["20221231"],
                "report_type": ["1"],
                "update_flag": ["0"],
            }
        ),
    )

    result = service.check_raw(AS_OF)

    assert result.ready
    assert any(
        "optional low-frequency dataset is stale: income" in warning for warning in result.warnings
    )


def test_missing_as_of_universe_and_duplicate_keys_fail(tmp_path: Path) -> None:
    missing = freshness_fixture(tmp_path / "missing", universe_current_count=0).check_universe(
        AS_OF
    )
    service = freshness_fixture(tmp_path / "duplicate")
    path = next(service.universe_store.dataset_dir.glob("**/*.parquet"))
    frame = pd.read_parquet(path)
    pd.concat([frame, frame.loc[frame["trade_date"].eq(AS_OF)].iloc[[0]]]).to_parquet(
        path, index=False
    )
    duplicate = service.check_universe(AS_OF)

    assert not missing.ready
    assert any("lacks as-of" in failure for failure in missing.hard_failures)
    assert not duplicate.ready
    assert any("duplicate keys" in failure for failure in duplicate.hard_failures)


def test_universe_moderate_drift_warns_and_severe_drift_fails(tmp_path: Path) -> None:
    moderate = freshness_fixture(tmp_path / "moderate", universe_current_count=4).check_universe(
        AS_OF
    )
    severe = freshness_fixture(tmp_path / "severe", universe_current_count=2).check_universe(AS_OF)

    assert moderate.ready
    assert any("moderate baseline deviation" in warning for warning in moderate.warnings)
    assert not severe.ready
    assert any("severe baseline deviation" in failure for failure in severe.hard_failures)


def test_feature_universe_mismatch_and_missing_hard_feature_fail(tmp_path: Path) -> None:
    mismatch = freshness_fixture(tmp_path / "mismatch", feature_current_count=4).check_features(
        AS_OF
    )
    missing_feature = freshness_fixture(
        tmp_path / "missing-feature", hard_required_features=("required_signal",)
    ).check_features(AS_OF)

    assert not mismatch.ready
    assert any("row mismatch" in failure for failure in mismatch.hard_failures)
    assert not missing_feature.ready
    assert any(
        "hard-required feature columns" in failure for failure in missing_feature.hard_failures
    )


def test_structurally_sparse_features_do_not_fail_and_missingness_is_reported(
    tmp_path: Path,
) -> None:
    result = freshness_fixture(tmp_path).check_features(AS_OF)

    assert result.ready
    assert result.details["missingness_summary"]["current_ratio"] == 1.0
    assert not any("current_ratio" in failure for failure in result.hard_failures)


def test_repeated_readiness_is_read_only_and_deterministic(tmp_path: Path) -> None:
    service = freshness_fixture(tmp_path)
    files = sorted(
        path
        for root in (
            service.raw_store.root,
            service.universe_store.root,
            service.feature_store.root,
        )
        for path in root.glob("**/*")
        if path.is_file()
    )
    before = {path: path.stat().st_mtime_ns for path in files}

    first = tuple(result.to_dict() for result in service.check_all(AS_OF))
    second = tuple(result.to_dict() for result in service.check_all(AS_OF))

    assert first == second
    assert before == {path: path.stat().st_mtime_ns for path in files}


def freshness_fixture(
    tmp_path: Path,
    *,
    stale_daily: bool = False,
    missing_benchmark: bool = False,
    universe_current_count: int = 5,
    feature_current_count: int | None = None,
    hard_required_features: tuple[str, ...] = (),
    now: datetime | None = None,
) -> FreshnessService:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    config_path = tmp_path / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("project_name: freshness-test\n", encoding="utf-8")
    settings = AppSettings.model_validate(
        {
            "data": {"index_codes": ["000300.SH"]},
            "production": {
                "freshness": {
                    "baseline_sessions": 5,
                    "minimum_baseline_sessions": 3,
                    "minimum_daily_rows": 1,
                    "minimum_universe_rows": 1,
                    "minimum_base_universe_rows": 1,
                    "minimum_model_universe_rows": 1,
                    "moderate_count_ratio_low": 0.9,
                    "moderate_count_ratio_high": 1.1,
                    "severe_count_ratio_low": 0.5,
                    "severe_count_ratio_high": 1.5,
                    "hard_required_features": list(hard_required_features),
                    "warning_features": ["ret_1d"],
                    "structurally_sparse_features": ["current_ratio"],
                    "git_dirty_policy": "ignore",
                }
            },
        }
    )
    raw_store = ParquetDataStore(raw_root)
    raw_store.write(
        get_dataset_spec("trade_cal"),
        pd.DataFrame(
            {"exchange": ["SSE"] * len(DATES), "cal_date": DATES, "is_open": [1] * len(DATES)}
        ),
    )
    market_dates = DATES[:-1] if stale_daily else DATES
    _write_stock_dataset(raw_store, "daily", market_dates)
    for name in ("adj_factor", "daily_basic", "stk_limit"):
        _write_stock_dataset(raw_store, name, DATES)
    index_dates = DATES[:-1] if missing_benchmark else DATES
    raw_store.write(
        get_dataset_spec("index_daily"),
        pd.DataFrame(
            {
                "ts_code": ["000300.SH"] * len(index_dates),
                "trade_date": index_dates,
                "close": [100.0] * len(index_dates),
            }
        ),
    )
    raw_store.mark_empty_result(get_dataset_spec("suspend_d"), AS_OF)

    universe_store = UniverseStore(processed_root)
    universe_store.write(_universe_frame(universe_current_count))
    feature_store = FeatureStore(processed_root)
    current_feature_count = (
        universe_current_count if feature_current_count is None else feature_current_count
    )
    feature_store.write(_feature_frame(current_feature_count))
    _write_processed_manifest(
        universe_store.dataset_dir, "universe_daily", config_path, feature_count=None
    )
    _write_processed_manifest(
        feature_store.dataset_dir, "features_daily", config_path, feature_count=2
    )
    return FreshnessService(
        settings,
        raw_store,
        universe_store,
        feature_store,
        config_path=config_path,
        now=now or datetime(2024, 1, 10, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )


def _write_stock_dataset(store: ParquetDataStore, name: str, dates: tuple[str, ...]) -> None:
    rows: list[dict[str, object]] = []
    for trade_date in dates:
        for index in range(5):
            base: dict[str, object] = {
                "ts_code": f"00000{index + 1}.SZ",
                "trade_date": trade_date,
            }
            if name == "daily":
                base.update(open=10.0, high=10.2, low=9.8, close=10.0, vol=100.0)
            elif name == "adj_factor":
                base["adj_factor"] = 1.0
            elif name == "daily_basic":
                base.update(turnover_rate=1.0, pe=10.0, pb=1.0)
            elif name == "stk_limit":
                base.update(up_limit=11.0, down_limit=9.0)
            rows.append(base)
    store.write(get_dataset_spec(name), pd.DataFrame(rows))


def _universe_frame(current_count: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trade_date in DATES:
        count = current_count if trade_date == AS_OF else 5
        for index in range(count):
            row = {column: None for column in UNIVERSE_COLUMNS}
            row.update(
                trade_date=trade_date,
                ts_code=f"00000{index + 1}.SZ",
                name=f"Stock {index + 1}",
                market="Main",
                exchange="SZSE",
                list_days=500,
                is_listed=True,
                is_new_stock=False,
                is_st=False,
                is_suspended=False,
                is_low_liquidity=False,
                is_limit_up=False,
                is_limit_down=False,
                can_buy=True,
                can_sell=True,
                in_base_universe=True,
                in_model_universe=True,
                exclude_reason="",
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _feature_frame(current_count: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trade_date in DATES:
        count = current_count if trade_date == AS_OF else 5
        for index in range(count):
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": f"00000{index + 1}.SZ",
                    "ret_1d": 0.01,
                    "current_ratio": None,
                }
            )
    return pd.DataFrame(rows)


def _write_processed_manifest(
    artifact_dir: Path,
    artifact_name: str,
    config_path: Path,
    *,
    feature_count: int | None,
) -> None:
    statistics = parquet_artifact_statistics(artifact_dir)
    extra = None if feature_count is None else {"feature_count": feature_count}
    write_build_manifest(
        artifact_dir,
        artifact_name=artifact_name,
        build_started_at="2024-01-10T08:00:00+00:00",
        config_path=config_path,
        start_date=DATES[0],
        end_date=AS_OF,
        row_count=statistics.row_count,
        canonical_statistics=statistics,
        partitions_changed=1,
        source_fingerprints={},
        extra=extra,
    )
