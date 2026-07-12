from __future__ import annotations

import os
import time

import pandas as pd

from ashare_quant.config import load_settings
from ashare_quant.data.exceptions import TushareRequestError
from ashare_quant.data.ingestion import DataIngestionService
from ashare_quant.data.storage import ParquetDataStore


class MockClient:
    def query(self, endpoint: str, **params: object) -> pd.DataFrame:
        if endpoint == "trade_cal":
            return pd.DataFrame(
                {
                    "exchange": ["SSE", "SSE"],
                    "cal_date": ["20240102", "20240103"],
                    "is_open": [1, 1],
                }
            )
        if endpoint == "daily":
            trade_date = str(params["trade_date"])
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "trade_date": [trade_date],
                    "open": [10.0],
                    "high": [11.0],
                    "low": [9.5],
                    "close": [10.5],
                    "vol": [100.0],
                }
            )
        raise AssertionError(f"unexpected endpoint: {endpoint}")


def test_init_downloads_calendar_then_daily_with_mock_client(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    store = ParquetDataStore(tmp_path)
    service = DataIngestionService(settings=settings, store=store, client=MockClient())  # type: ignore[arg-type]

    results = service.init(("daily",), start_date="20240102", end_date="20240103")

    assert [result.dataset for result in results] == ["trade_cal", "daily"]
    assert store.status(store_spec("daily")).rows == 2


def store_spec(name: str):
    from ashare_quant.data.datasets import get_dataset_spec

    return get_dataset_spec(name)


class FailingDailyClient:
    def __init__(self) -> None:
        self.daily_calls = 0

    def query(self, endpoint: str, **params: object) -> pd.DataFrame:
        if endpoint == "trade_cal":
            return pd.DataFrame(
                {
                    "exchange": ["SSE", "SSE"],
                    "cal_date": ["20240102", "20240103"],
                    "is_open": [1, 1],
                }
            )
        if endpoint == "daily":
            self.daily_calls += 1
            if self.daily_calls == 2:
                raise TushareRequestError("simulated failure")
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "trade_date": [str(params["trade_date"])],
                    "open": [10.0],
                    "high": [11.0],
                    "low": [9.5],
                    "close": [10.5],
                    "vol": [100.0],
                }
            )
        raise AssertionError(f"unexpected endpoint: {endpoint}")


def test_trade_date_download_streams_successful_days_before_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    store = ParquetDataStore(tmp_path)
    service = DataIngestionService(
        settings=settings,
        store=store,
        client=FailingDailyClient(),  # type: ignore[arg-type]
    )

    try:
        service.init(("daily",), start_date="20240102", end_date="20240103")
    except TushareRequestError:
        pass

    assert store.status(store_spec("daily")).rows == 1


class RevisionLookbackClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def query(self, endpoint: str, **params: object) -> pd.DataFrame:
        self.calls.append((endpoint, params))
        if endpoint == "trade_cal":
            return pd.DataFrame(
                {
                    "exchange": ["SSE"] * 8,
                    "cal_date": [
                        "20260701",
                        "20260702",
                        "20260703",
                        "20260706",
                        "20260707",
                        "20260708",
                        "20260709",
                        "20260710",
                    ],
                    "is_open": [1] * 8,
                }
            )
        if endpoint == "top_list":
            trade_date = str(params["trade_date"])
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "trade_date": [trade_date],
                    "reason": ["日涨幅偏离值达到7%的前5只证券"],
                }
            )
        raise AssertionError(f"unexpected endpoint: {endpoint}")


def test_update_rechecks_revision_lookback_open_days(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    store = ParquetDataStore(tmp_path)
    trade_cal = store_spec("trade_cal")
    top_list = store_spec("top_list")
    store.write(
        trade_cal,
        pd.DataFrame(
            {
                "exchange": ["SSE"] * 8,
                "cal_date": [
                    "20260701",
                    "20260702",
                    "20260703",
                    "20260706",
                    "20260707",
                    "20260708",
                    "20260709",
                    "20260710",
                ],
                "is_open": [1] * 8,
            }
        ),
    )
    store.write(
        top_list,
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20260708"],
                "reason": ["日涨幅偏离值达到7%的前5只证券"],
            }
        ),
    )
    client = RevisionLookbackClient()
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    service.update(("top_list",), "20260710")

    top_list_dates = [
        str(params["trade_date"]) for endpoint, params in client.calls if endpoint == "top_list"
    ]
    assert top_list_dates == [
        "20260702",
        "20260703",
        "20260706",
        "20260707",
        "20260708",
        "20260709",
        "20260710",
    ]


class SnapshotMockClient:
    def __init__(self, prefix: str = "new", fail: bool = False, empty: bool = False) -> None:
        self.prefix = prefix
        self.fail = fail
        self.empty = empty
        self.calls: list[tuple[str, dict[str, object]]] = []

    def query(self, endpoint: str, **params: object) -> pd.DataFrame:
        self.calls.append((endpoint, params))
        if endpoint != "stock_basic":
            raise AssertionError(f"unexpected endpoint: {endpoint}")
        if self.fail:
            raise TushareRequestError("simulated snapshot failure")
        if self.empty:
            return pd.DataFrame(columns=["ts_code", "symbol", "name", "list_date"])
        status = str(params["list_status"])
        return pd.DataFrame(
            {
                "ts_code": [f"{self.prefix}_{status}.SZ"],
                "symbol": [f"{self.prefix}_{status}"],
                "name": [f"{self.prefix}-{status}"],
                "list_date": ["20200101"],
            }
        )


def write_stock_basic_snapshot(store: ParquetDataStore, prefix: str = "old") -> None:
    store.write(
        store_spec("stock_basic"),
        pd.DataFrame(
            {
                "ts_code": [f"{prefix}_L.SZ"],
                "symbol": [f"{prefix}_L"],
                "name": [f"{prefix}-L"],
                "list_date": ["20200101"],
            }
        ),
    )


def set_snapshot_age_days(store: ParquetDataStore, days: int) -> None:
    path = store.dataset_dir(store_spec("stock_basic")) / "snapshot=latest" / "data.parquet"
    timestamp = time.time() - days * 24 * 60 * 60
    os.utime(path, (timestamp, timestamp))


def test_snapshot_ttl_policy_skips_existing_fresh_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    settings.data.snapshot_refresh_policy = "ttl_days"
    settings.data.snapshot_refresh_ttl_days = 7
    store = ParquetDataStore(tmp_path)
    write_stock_basic_snapshot(store)
    set_snapshot_age_days(store, 1)
    client = SnapshotMockClient()
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    result = service.update(("stock_basic",), "20240201")

    assert result[0].skipped is True
    assert result[0].message == "snapshot already exists"
    assert client.calls == []
    assert store.read_dataset(store_spec("stock_basic"))["ts_code"].tolist() == ["old_L.SZ"]


def test_snapshot_ttl_policy_refreshes_stale_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    settings.data.snapshot_refresh_policy = "ttl_days"
    settings.data.snapshot_refresh_ttl_days = 7
    store = ParquetDataStore(tmp_path)
    write_stock_basic_snapshot(store)
    set_snapshot_age_days(store, 30)
    client = SnapshotMockClient(prefix="new")
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    result = service.update(("stock_basic",), "20240201")

    stored_codes = set(store.read_dataset(store_spec("stock_basic"))["ts_code"].astype(str))
    assert result[0].skipped is False
    assert result[0].message == "snapshot refreshed"
    assert stored_codes == {"new_L.SZ", "new_D.SZ", "new_P.SZ"}
    assert [params["list_status"] for _, params in client.calls] == ["L", "D", "P"]


def test_snapshot_manual_policy_skips_unless_explicitly_refreshed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    settings.data.snapshot_refresh_policy = "manual"
    store = ParquetDataStore(tmp_path)
    write_stock_basic_snapshot(store)
    client = SnapshotMockClient(prefix="new")
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    skipped = service.update(("stock_basic",), "20240201")
    refreshed = service.update(("stock_basic",), "20240201", refresh_snapshots=True)

    stored_codes = set(store.read_dataset(store_spec("stock_basic"))["ts_code"].astype(str))
    assert skipped[0].skipped is True
    assert refreshed[0].message == "snapshot refreshed"
    assert stored_codes == {"new_L.SZ", "new_D.SZ", "new_P.SZ"}


def test_snapshot_always_policy_refreshes_existing_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    settings.data.snapshot_refresh_policy = "always"
    store = ParquetDataStore(tmp_path)
    write_stock_basic_snapshot(store)
    client = SnapshotMockClient(prefix="new")
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    result = service.update(("stock_basic",), "20240201")

    stored_codes = set(store.read_dataset(store_spec("stock_basic"))["ts_code"].astype(str))
    assert result[0].message == "snapshot refreshed"
    assert stored_codes == {"new_L.SZ", "new_D.SZ", "new_P.SZ"}


def test_failed_snapshot_refresh_preserves_existing_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    store = ParquetDataStore(tmp_path)
    write_stock_basic_snapshot(store)
    client = SnapshotMockClient(fail=True)
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    result = service.update(("stock_basic",), "20240201", refresh_snapshots=True)

    stored = store.read_dataset(store_spec("stock_basic"))
    assert result[0].skipped is True
    assert "existing snapshot preserved" in result[0].message
    assert stored["ts_code"].tolist() == ["old_L.SZ"]


def test_empty_snapshot_refresh_preserves_existing_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    store = ParquetDataStore(tmp_path)
    write_stock_basic_snapshot(store)
    client = SnapshotMockClient(empty=True)
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    result = service.update(("stock_basic",), "20240201", refresh_snapshots=True)

    stored = store.read_dataset(store_spec("stock_basic"))
    assert result[0].skipped is True
    assert result[0].message == "snapshot refresh returned no rows; existing snapshot preserved"
    assert stored["ts_code"].tolist() == ["old_L.SZ"]


def test_snapshot_status_reports_age_and_updated_at(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    write_stock_basic_snapshot(store)
    set_snapshot_age_days(store, 3)

    status = store.status(store_spec("stock_basic"))

    assert status.snapshot_updated_at is not None
    assert status.snapshot_age_days is not None
    assert status.snapshot_age_days >= 2
