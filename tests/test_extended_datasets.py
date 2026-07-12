from __future__ import annotations

import pandas as pd
import pytest

from ashare_quant.cli import main
from ashare_quant.config import load_settings
from ashare_quant.data.datasets import ALL_DATASETS, DEFAULT_DATASETS, get_dataset_spec
from ashare_quant.data.exceptions import (
    DataValidationError,
    TushareNoDataError,
    TusharePermissionError,
)
from ashare_quant.data.ingestion import DataIngestionService
from ashare_quant.data.storage import ParquetDataStore


class ExtendedMockClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def query(self, endpoint: str, **params: object) -> pd.DataFrame:
        self.calls.append((endpoint, params))
        if endpoint == "trade_cal":
            return pd.DataFrame(
                {
                    "exchange": ["SSE", "SSE"],
                    "cal_date": ["20240105", "20240131"],
                    "is_open": [1, 1],
                }
            )
        if endpoint == "fund_basic":
            return pd.DataFrame({"ts_code": ["510300.SH"], "name": ["ETF"]})
        if endpoint == "hs_const":
            return pd.DataFrame({"ts_code": ["600000.SH"], "hs_type": [params["hs_type"]]})
        if endpoint == "weekly":
            return pd.DataFrame(
                {"ts_code": ["000001.SZ"], "trade_date": ["20240105"], "close": [10.0]}
            )
        raise AssertionError(f"unexpected endpoint: {endpoint}")


def test_extended_dataset_specs_include_user_permission_scope() -> None:
    expected = {
        "fund_basic",
        "fund_daily",
        "opt_basic",
        "weekly",
        "monthly",
        "income",
        "balancesheet",
        "cashflow",
        "fina_indicator",
        "forecast",
        "express",
        "namechange",
        "hs_const",
        "pledge_stat",
        "pledge_detail",
        "share_float",
        "repurchase",
        "stk_holdertrade",
        "top_list",
        "top_inst",
        "margin",
        "margin_detail",
        "concept",
        "concept_detail",
        "moneyflow",
        "moneyflow_hsgt",
        "broker_recommend",
        "cyq_chips",
        "cyq_perf",
        "stk_factor",
        "cn_gdp",
        "cn_cpi",
        "cn_ppi",
        "cn_m",
    }

    assert expected.issubset(set(ALL_DATASETS))
    assert set(DEFAULT_DATASETS).issubset(set(ALL_DATASETS))


def test_longhubang_dataset_primary_keys_preserve_multiple_reasons() -> None:
    assert get_dataset_spec("top_list").primary_key == ("ts_code", "trade_date", "reason")
    assert get_dataset_spec("top_inst").primary_key == (
        "ts_code",
        "trade_date",
        "exalter",
        "side",
        "reason",
    )


def test_extended_fetch_modes_use_expected_parameters(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    store = ParquetDataStore(tmp_path)
    client = ExtendedMockClient()
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    results = service.init(("fund_basic", "hs_const", "weekly"), "20240101", "20240131")

    assert [result.dataset for result in results] == [
        "trade_cal",
        "fund_basic",
        "hs_const",
        "weekly",
    ]
    assert store.status(get_dataset_spec("fund_basic")).rows == 1
    assert store.status(get_dataset_spec("hs_const")).rows == 2
    assert store.status(get_dataset_spec("weekly")).rows == 1
    assert ("fund_basic", {"market": "E"}) in client.calls
    assert ("hs_const", {"hs_type": "SH"}) in client.calls
    assert ("hs_const", {"hs_type": "SZ"}) in client.calls
    weekly_calls = [params for endpoint, params in client.calls if endpoint == "weekly"]
    assert weekly_calls == [{"trade_date": "20240105"}, {"trade_date": "20240131"}]


class SpecialDatasetMockClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def query(self, endpoint: str, **params: object) -> pd.DataFrame:
        self.calls.append((endpoint, params))
        if endpoint == "trade_cal":
            return pd.DataFrame(
                {
                    "exchange": ["SSE", "SSE"],
                    "cal_date": ["20240103", "20240131"],
                    "is_open": [1, 1],
                }
            )
        if endpoint == "stock_basic":
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "symbol": ["000001", "000002"],
                    "name": ["one", "two"],
                    "list_date": ["20000101", "20000101"],
                }
            )
        if endpoint == "ths_index":
            return pd.DataFrame(
                {
                    "ts_code": ["885001.TI", "885002.TI"],
                    "name": ["concept one", "concept two"],
                    "count": [1, 1],
                }
            )
        if endpoint == "ths_member":
            return pd.DataFrame(
                {
                    "ts_code": [params["ts_code"]],
                    "con_code": ["000001.SZ"],
                    "con_name": ["one"],
                }
            )
        if endpoint == "broker_recommend":
            return pd.DataFrame(
                {
                    "month": [params["month"]],
                    "broker": ["broker"],
                    "ts_code": ["000001.SZ"],
                }
            )
        if endpoint in {"cyq_chips", "cyq_perf"}:
            return pd.DataFrame(
                {
                    "ts_code": [params["ts_code"]],
                    "trade_date": ["20240103"],
                    "price": [10.0],
                    "percent": [1.0],
                }
            )
        raise AssertionError(f"unexpected endpoint: {endpoint}")


class CyqNoDataMockClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def query(self, endpoint: str, **params: object) -> pd.DataFrame:
        self.calls.append((endpoint, params))
        if endpoint == "trade_cal":
            return pd.DataFrame(
                {
                    "exchange": ["SSE"],
                    "cal_date": ["20240103"],
                    "is_open": [1],
                }
            )
        if endpoint == "stock_basic":
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000003.SZ", "000004.SZ"],
                    "symbol": ["000001", "000003", "000004"],
                    "name": ["one", "missing", "four"],
                    "list_date": ["20000101", "20000101", "20000101"],
                }
            )
        if endpoint == "cyq_chips":
            if params["ts_code"] == "000003.SZ":
                raise TushareNoDataError("No data for Tushare endpoint 'cyq_chips'")
            return pd.DataFrame(
                {
                    "ts_code": [params["ts_code"]],
                    "trade_date": ["20240103"],
                    "price": [10.0],
                    "percent": [1.0],
                }
            )
        raise AssertionError(f"unexpected endpoint: {endpoint}")


def test_special_dataset_fetch_modes_use_current_tushare_parameters(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    store = ParquetDataStore(tmp_path)
    client = SpecialDatasetMockClient()
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    service.init(
        ("concept_detail", "broker_recommend", "cyq_chips"),
        "20240101",
        "20240215",
    )

    assert ("ths_index", {"exchange": "A", "type": "N"}) in client.calls
    ths_member_calls = [params for endpoint, params in client.calls if endpoint == "ths_member"]
    assert ths_member_calls == [{"ts_code": "885001.TI"}, {"ts_code": "885002.TI"}]
    broker_calls = [params for endpoint, params in client.calls if endpoint == "broker_recommend"]
    assert broker_calls == [{"month": "202401"}, {"month": "202402"}]
    cyq_calls = [params for endpoint, params in client.calls if endpoint == "cyq_chips"]
    assert cyq_calls == [
        {"ts_code": "000001.SZ", "start_date": "20240101", "end_date": "20240215"},
        {"ts_code": "000002.SZ", "start_date": "20240101", "end_date": "20240215"},
    ]


def test_ts_code_date_range_skips_no_data_entities(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    store = ParquetDataStore(tmp_path)
    client = CyqNoDataMockClient()
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    service.init(("cyq_chips",), "20240101", "20240131")

    cyq_calls = [params["ts_code"] for endpoint, params in client.calls if endpoint == "cyq_chips"]
    assert cyq_calls == ["000001.SZ", "000003.SZ", "000004.SZ"]
    assert store.status(get_dataset_spec("cyq_chips")).rows == 2


class FinanceByStockMockClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def query(self, endpoint: str, **params: object) -> pd.DataFrame:
        self.calls.append((endpoint, params))
        if endpoint == "income_vip":
            raise TusharePermissionError("VIP permission denied")
        if endpoint == "stock_basic":
            status = str(params["list_status"])
            number = {"L": 1, "D": 2, "P": 3}[status]
            return pd.DataFrame(
                {
                    "ts_code": [f"00000{number}.SZ"],
                    "symbol": [f"00000{number}"],
                    "name": [status],
                    "list_date": ["20000101"],
                }
            )
        if endpoint == "income":
            return pd.DataFrame(
                {
                    "ts_code": [params["ts_code"], params["ts_code"]],
                    "ann_date": ["20240131", "20240201"],
                    "f_ann_date": ["20240131", "20240201"],
                    "end_date": ["20231231", "20231231"],
                    "report_type": ["1", "1"],
                    "update_flag": ["1", "1"],
                }
            )
        raise AssertionError(f"unexpected endpoint: {endpoint}")


def test_finance_falls_back_to_stock_code_and_server_date_range(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    store = ParquetDataStore(tmp_path)
    client = FinanceByStockMockClient()
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    service.init(("income",), "20240101", "20240131")

    stock_statuses = [
        params["list_status"] for endpoint, params in client.calls if endpoint == "stock_basic"
    ]
    income_calls = [params for endpoint, params in client.calls if endpoint == "income"]
    assert stock_statuses == ["L", "D", "P"]
    assert income_calls == [
        {"ts_code": "000001.SZ", "start_date": "20240101", "end_date": "20240131"},
        {"ts_code": "000002.SZ", "start_date": "20240101", "end_date": "20240131"},
        {"ts_code": "000003.SZ", "start_date": "20240101", "end_date": "20240131"},
    ]
    assert store.status(get_dataset_spec("income")).rows == 3


def test_finance_update_uses_bounded_revision_window_for_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    store = ParquetDataStore(tmp_path)
    income_spec = get_dataset_spec("income")
    store.write(
        income_spec,
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["20240131"],
                "f_ann_date": ["20240131"],
                "end_date": ["20231231"],
                "report_type": ["1"],
                "update_flag": ["1"],
            }
        ),
    )
    client = FinanceByStockMockClient()
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    service.update(("income",), "20240131")

    income_calls = [params for endpoint, params in client.calls if endpoint == "income"]
    assert income_calls == [
        {"ts_code": "000001.SZ", "start_date": "20220730", "end_date": "20240131"},
        {"ts_code": "000002.SZ", "start_date": "20220730", "end_date": "20240131"},
        {"ts_code": "000003.SZ", "start_date": "20220730", "end_date": "20240131"},
    ]
    assert store.status(income_spec).rows == 3


class FinanceVipMockClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def query(self, endpoint: str, **params: object) -> pd.DataFrame:
        self.calls.append((endpoint, params))
        if endpoint != "income_vip" or params["period"] != "20231231":
            return pd.DataFrame()
        offset = int(params["offset"])
        if offset == 0:
            codes = ["000001.SZ", "000002.SZ"]
        elif offset == 2:
            codes = ["000003.SZ"]
        else:
            codes = []
        return pd.DataFrame(
            {
                "ts_code": codes,
                "ann_date": ["20240131"] * len(codes),
                "f_ann_date": ["20240131"] * len(codes),
                "end_date": ["20231231"] * len(codes),
                "report_type": ["1"] * len(codes),
                "update_flag": ["1"] * len(codes),
            }
        )


def test_finance_vip_fetches_report_periods_with_pagination(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    settings.data.tushare_page_size = 2
    store = ParquetDataStore(tmp_path)
    client = FinanceVipMockClient()
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    service.init(("income",), "20240101", "20240131")

    income_calls = [params for endpoint, params in client.calls if endpoint == "income_vip"]
    report_calls = [params for params in income_calls if params["period"] == "20231231"]
    assert report_calls == [
        {"period": "20231231", "limit": 2, "offset": 0},
        {"period": "20231231", "limit": 2, "offset": 2},
    ]
    assert store.status(get_dataset_spec("income")).rows == 3


def test_cli_accepts_extended_dataset_for_status(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "data",
            "--storage-root",
            str(tmp_path),
            "validate",
            "--dataset",
            "fund_basic",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "fund_basic: ok=True" in captured.out


class RangeAndSnapshotMockClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def query(self, endpoint: str, **params: object) -> pd.DataFrame:
        self.calls.append((endpoint, params))
        if endpoint == "trade_cal":
            return pd.DataFrame(
                {
                    "exchange": ["SSE", "SSE", "SSE"],
                    "cal_date": ["20231229", "20241231", "20250115"],
                    "is_open": [1, 1, 1],
                }
            )
        if endpoint == "stock_basic":
            status = str(params["list_status"])
            return pd.DataFrame(
                {
                    "ts_code": [f"00000{len(self.calls)}.SZ"],
                    "symbol": [f"00000{len(self.calls)}"],
                    "name": [status],
                    "list_date": ["20000101"],
                }
            )
        if endpoint == "weekly":
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "trade_date": [str(params["trade_date"])],
                    "close": [10.0],
                }
            )
        raise AssertionError(f"unexpected endpoint: {endpoint}")


def test_stock_basic_fetches_all_configured_list_statuses(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    store = ParquetDataStore(tmp_path)
    client = RangeAndSnapshotMockClient()
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    service.init(("stock_basic",), "20240101", "20240131")

    statuses = [
        params["list_status"] for endpoint, params in client.calls if endpoint == "stock_basic"
    ]
    assert statuses == ["L", "D", "P"]
    assert store.status(get_dataset_spec("stock_basic")).rows == 3


def test_weekly_uses_year_end_trade_dates_from_calendar(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    store = ParquetDataStore(tmp_path)
    client = RangeAndSnapshotMockClient()
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    service.init(("weekly",), "20230101", "20250115")

    weekly_params = [params for endpoint, params in client.calls if endpoint == "weekly"]
    assert weekly_params == [
        {"trade_date": "20231229"},
        {"trade_date": "20241231"},
        {"trade_date": "20250115"},
    ]


def test_update_skips_existing_snapshot_dataset(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    store = ParquetDataStore(tmp_path)
    client = RangeAndSnapshotMockClient()
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    service.init(("stock_basic",), "20240101", "20240131")
    client.calls.clear()
    result = service.update(("stock_basic",), "20240201")

    assert result[0].skipped is True
    assert result[0].message == "snapshot already exists"
    assert client.calls == []


class PeriodModeMockClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def query(self, endpoint: str, **params: object) -> pd.DataFrame:
        self.calls.append((endpoint, params))
        if endpoint == "trade_cal":
            return pd.DataFrame(
                {
                    "exchange": ["SSE"] * 6,
                    "cal_date": [
                        "20240102",
                        "20240103",
                        "20240104",
                        "20240105",
                        "20240108",
                        "20240131",
                    ],
                    "is_open": [1, 1, 1, 1, 1, 1],
                }
            )
        if endpoint == "fund_daily":
            return pd.DataFrame(
                {"ts_code": ["510300.SH"], "trade_date": [params["trade_date"]], "close": [1.0]}
            )
        if endpoint in {"weekly", "monthly"}:
            return pd.DataFrame(
                {"ts_code": ["000001.SZ"], "trade_date": [params["trade_date"]], "close": [10.0]}
            )
        raise AssertionError(f"unexpected endpoint: {endpoint}")


def test_fund_daily_uses_trade_date_not_date_range(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    store = ParquetDataStore(tmp_path)
    client = PeriodModeMockClient()
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    service.init(("fund_daily",), "20240102", "20240105")

    fund_calls = [params for endpoint, params in client.calls if endpoint == "fund_daily"]
    assert fund_calls == [
        {"trade_date": "20240102"},
        {"trade_date": "20240103"},
        {"trade_date": "20240104"},
        {"trade_date": "20240105"},
    ]


def test_weekly_and_monthly_use_period_end_trade_dates(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    store = ParquetDataStore(tmp_path)
    client = PeriodModeMockClient()
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    service.init(("weekly", "monthly"), "20240102", "20240131")

    weekly_calls = [params for endpoint, params in client.calls if endpoint == "weekly"]
    monthly_calls = [params for endpoint, params in client.calls if endpoint == "monthly"]
    assert weekly_calls == [
        {"trade_date": "20240105"},
        {"trade_date": "20240108"},
        {"trade_date": "20240131"},
    ]
    assert monthly_calls == [{"trade_date": "20240131"}]


class CyqRowLimitMockClient:
    def __init__(self, always_at_limit: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.always_at_limit = always_at_limit

    def query(self, endpoint: str, **params: object) -> pd.DataFrame:
        self.calls.append((endpoint, params))
        if endpoint == "trade_cal":
            return pd.DataFrame(
                {
                    "exchange": ["SSE"] * 4,
                    "cal_date": ["20240102", "20240103", "20240104", "20240105"],
                    "is_open": [1] * 4,
                }
            )
        if endpoint == "stock_basic":
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "symbol": ["000001"],
                    "name": ["one"],
                    "list_date": ["20000101"],
                }
            )
        if endpoint != "cyq_chips":
            raise AssertionError(f"unexpected endpoint: {endpoint}")
        start = str(params["start_date"])
        end = str(params["end_date"])
        if self.always_at_limit or (start == "20240102" and end == "20240105"):
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"] * 6000,
                    "trade_date": [end] * 6000,
                    "price": [float(value) for value in range(6000)],
                    "percent": [0.0] * 6000,
                }
            )
        dates = ["20240102", "20240103"] if end == "20240103" else ["20240104", "20240105"]
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"] * len(dates),
                "trade_date": dates,
                "price": [10.0, 11.0],
                "percent": [50.0, 50.0],
            }
        )


def test_cyq_chips_splits_responses_that_reach_the_row_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    store = ParquetDataStore(tmp_path)
    client = CyqRowLimitMockClient()
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    service.init(("cyq_chips",), "20240102", "20240105")

    calls = [params for endpoint, params in client.calls if endpoint == "cyq_chips"]
    assert [(call["start_date"], call["end_date"]) for call in calls] == [
        ("20240102", "20240105"),
        ("20240102", "20240103"),
        ("20240104", "20240105"),
    ]
    assert store.status(get_dataset_spec("cyq_chips")).rows == 4


def test_cyq_chips_rejects_single_day_response_at_row_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    settings = load_settings("config/default.yaml")
    client = CyqRowLimitMockClient(always_at_limit=True)
    service = DataIngestionService(
        settings=settings,
        store=ParquetDataStore(tmp_path),
        client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(DataValidationError, match="completeness cannot be guaranteed"):
        service.init(("cyq_chips",), "20240102", "20240102")
