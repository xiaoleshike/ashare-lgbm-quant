from __future__ import annotations

import pandas as pd

from ashare_quant.data.datasets import get_dataset_spec
from ashare_quant.data.storage import ParquetDataStore
from ashare_quant.data.validation import DataValidator


def test_parquet_store_is_idempotent_by_primary_key(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    spec = get_dataset_spec("daily")
    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20240102"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.5],
            "close": [10.5],
            "vol": [100.0],
        }
    )

    store.write(spec, frame)
    store.write(spec, frame)

    stored = store.read_dataset(spec)
    assert len(stored) == 1
    assert store.status(spec).partitions == 1


def test_validator_detects_duplicate_primary_keys(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    spec = get_dataset_spec("stock_basic")
    path = tmp_path / "stock_basic" / "snapshot=latest"
    path.mkdir(parents=True)
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "symbol": ["000001", "000001"],
            "name": ["Ping An", "Ping An"],
            "list_date": ["19910403", "19910403"],
        }
    ).to_parquet(path / "data.parquet", index=False)

    result = DataValidator(store).validate_dataset(spec)

    assert result.ok is False
    assert result.status == "invalid"
    assert any("primary key" in error for error in result.errors)


def test_validator_accepts_valid_non_empty_required_dataset(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    spec = get_dataset_spec("stock_basic")
    store.write(
        spec,
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "symbol": ["000001"],
                "name": ["Ping An"],
                "list_date": ["19910403"],
            }
        ),
    )

    result = DataValidator(store).validate_dataset(spec)

    assert result.ok is True
    assert result.status == "valid"
    assert result.errors == ()


def test_validator_fails_missing_required_dataset(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    spec = get_dataset_spec("stock_basic")

    result = DataValidator(store).validate_dataset(spec)

    assert result.ok is False
    assert result.status == "missing"
    assert result.errors == ("required dataset is not downloaded",)


def test_validator_fails_empty_required_dataset(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    spec = get_dataset_spec("stock_basic")
    path = tmp_path / "stock_basic" / "snapshot=latest"
    path.mkdir(parents=True)
    pd.DataFrame(columns=list(spec.required_columns)).to_parquet(path / "data.parquet", index=False)

    result = DataValidator(store).validate_dataset(spec)

    assert result.ok is False
    assert result.status == "empty"
    assert result.errors == ("required dataset is empty",)


def test_validator_skips_missing_optional_dataset(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    spec = get_dataset_spec("fund_basic")

    result = DataValidator(store).validate_dataset(spec)

    assert spec.optional is True
    assert result.ok is True
    assert result.status == "skipped_optional"
    assert result.errors == ()
    assert any("optional dataset is not downloaded" in warning for warning in result.warnings)


def test_validator_fails_when_any_required_dataset_fails(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    stock_basic = get_dataset_spec("stock_basic")
    store.write(
        stock_basic,
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "symbol": ["000001"],
                "name": ["Ping An"],
                "list_date": ["19910403"],
            }
        ),
    )

    results = DataValidator(store).validate_all(("stock_basic", "daily", "fund_basic"))

    assert [result.status for result in results] == ["valid", "missing", "skipped_optional"]
    assert not all(result.ok for result in results)


def test_validator_warns_on_missing_open_trading_days(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    trade_cal = get_dataset_spec("trade_cal")
    daily = get_dataset_spec("daily")
    store.write(
        trade_cal,
        pd.DataFrame(
            {
                "exchange": ["SSE", "SSE"],
                "cal_date": ["20240102", "20240103"],
                "is_open": [1, 1],
            }
        ),
    )
    store.write(
        daily,
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20240102"],
                "open": [10.0],
                "high": [11.0],
                "low": [9.5],
                "close": [10.5],
                "vol": [100.0],
            }
        ),
    )

    result = DataValidator(store).validate_dataset(daily)

    assert result.ok is True
    assert any("missing 1 open trading days" in warning for warning in result.warnings)


def test_validator_warns_invalid_daily_ohlc_cleaning_policy(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    trade_cal = get_dataset_spec("trade_cal")
    daily = get_dataset_spec("daily")
    store.write(
        trade_cal,
        pd.DataFrame(
            {
                "exchange": ["SSE", "SSE"],
                "cal_date": ["20140618", "20240102"],
                "is_open": [1, 1],
            }
        ),
    )
    store.write(
        daily,
        pd.DataFrame(
            {
                "ts_code": ["920489.BJ", "000001.SZ"],
                "trade_date": ["20140618", "20240102"],
                "open": [10.88, 10.0],
                "high": [10.88, 10.0],
                "low": [10.88, 10.0],
                "close": [10.81, 9.9],
                "vol": [410.0, 100.0],
            }
        ),
    )

    result = DataValidator(store).validate_dataset(daily)

    assert result.ok is True
    assert any("pre-2020 invalid OHLC" in warning for warning in result.warnings)
    assert any("post-2020 invalid OHLC" in warning for warning in result.warnings)


def test_validator_classifies_stk_limit_special_values(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    stk_limit = get_dataset_spec("stk_limit")
    suspend_d = get_dataset_spec("suspend_d")
    store.write(
        suspend_d,
        pd.DataFrame(
            {
                "ts_code": ["601607.SH"],
                "trade_date": ["20100204"],
                "suspend_type": ["S"],
            }
        ),
    )
    store.write(
        stk_limit,
        pd.DataFrame(
            {
                "ts_code": ["920092.BJ", "601607.SH", "000001.SZ"],
                "trade_date": ["20211115", "20100204", "20240102"],
                "up_limit": [99999.99, 0.0, 11.0],
                "down_limit": [0.0, 0.0, 9.0],
            }
        ),
    )

    result = DataValidator(store).validate_dataset(stk_limit)

    assert result.ok is True
    assert any("no-price-limit sentinel" in warning for warning in result.warnings)
    assert any("suspended zero-limit rows" in warning for warning in result.warnings)
