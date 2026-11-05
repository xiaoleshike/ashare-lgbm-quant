from __future__ import annotations

import json

import pandas as pd
import pytest

from ashare_quant.cli import main
from ashare_quant.config import load_settings
from ashare_quant.data.datasets import ALL_DATASETS, DEFAULT_DATASETS, get_dataset_spec
from ashare_quant.data.exceptions import (
    DataValidationError,
    TushareNoDataError,
    TushareParameterError,
    TusharePermissionError,
    TushareRequestError,
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


class ShareFloatMockClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.responses: dict[str | tuple[str, int], pd.DataFrame | Exception] = {}

    def query(self, endpoint: str, **params: object) -> pd.DataFrame:
        self.calls.append((endpoint, params))
        assert endpoint == "share_float"
        if "ann_date" not in params:
            raise TushareParameterError("ts_code or ann_date is required")
        ann_date = str(params["ann_date"])
        response = self.responses.get(
            (ann_date, int(params.get("offset", 0))),
            self.responses.get(
                ann_date,
                share_float_rows(ann_date, ["holder"]),
            ),
        )
        if isinstance(response, Exception):
            raise response
        return response.copy()


def share_float_rows(ann_date: str, holders: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * len(holders),
            "ann_date": [ann_date] * len(holders),
            "float_date": ["20270115"] * len(holders),
            "holder_name": holders,
            "share_type": ["placement"] * len(holders),
            "float_share": [1000.0] * len(holders),
        }
    )


def test_share_float_init_queries_announcement_dates_and_preserves_future_unlock(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    client = ShareFloatMockClient()
    service = DataIngestionService(
        settings=load_settings("config/default.yaml"),
        store=store,
        client=client,  # type: ignore[arg-type]
    )

    results = service.init(("share_float",), "20261104", "20261105")

    assert client.calls == [
        ("share_float", {"ann_date": "20261104", "limit": 6000, "offset": 0}),
        ("share_float", {"ann_date": "20261105", "limit": 6000, "offset": 0}),
    ]
    assert [result.dataset for result in results] == ["share_float"]
    assert results[0].rows_written == 2
    frame = store.read_dataset(get_dataset_spec("share_float"))
    assert frame["ann_date"].tolist() == ["20261104", "20261105"]
    assert frame["float_date"].tolist() == ["20270115", "20270115"]


@pytest.mark.parametrize(
    ("start", "end", "dates"),
    [
        ("20240105", "20240108", ["20240105", "20240106", "20240107", "20240108"]),
        ("20231231", "20240101", ["20231231", "20240101"]),
        ("20240228", "20240301", ["20240228", "20240229", "20240301"]),
    ],
)
def test_share_float_queries_every_calendar_day(tmp_path, start, end, dates) -> None:
    client = ShareFloatMockClient()
    service = DataIngestionService(
        settings=load_settings("config/default.yaml"),
        store=ParquetDataStore(tmp_path),
        client=client,  # type: ignore[arg-type]
    )

    service.init(("share_float",), start, end)

    assert client.calls == [
        ("share_float", {"ann_date": date, "limit": 6000, "offset": 0}) for date in dates
    ]


def test_share_float_update_resumes_from_announcement_not_unlock_date(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    spec = get_dataset_spec("share_float")
    store.write(
        spec,
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["20261103"],
                "float_date": ["20270115"],
                "holder_name": ["holder"],
                "share_type": ["placement"],
            }
        ),
    )
    client = ShareFloatMockClient()
    service = DataIngestionService(
        settings=load_settings("config/default.yaml"),
        store=store,
        client=client,  # type: ignore[arg-type]
    )

    service.update(("share_float",), "20261105")
    service.update(("share_float",), "20261105")

    assert client.calls == [
        ("share_float", {"ann_date": "20261104", "limit": 6000, "offset": 0}),
        ("share_float", {"ann_date": "20261105", "limit": 6000, "offset": 0}),
    ]
    assert store.status(spec).rows == 3


@pytest.mark.parametrize("empty_response", [pd.DataFrame(), TushareNoDataError("no data")])
def test_share_float_empty_announcement_day_does_not_block_following_days(
    tmp_path, empty_response
) -> None:
    store = ParquetDataStore(tmp_path)
    client = ShareFloatMockClient()
    client.responses["20261104"] = empty_response
    service = DataIngestionService(
        settings=load_settings("config/default.yaml"),
        store=store,
        client=client,  # type: ignore[arg-type]
    )

    results = service.init(("share_float",), "20261104", "20261105")

    assert results[0].rows_written == 1
    assert store.read_dataset(get_dataset_spec("share_float"))["ann_date"].tolist() == ["20261105"]


def test_share_float_keeps_completed_days_and_resumes_after_request_failure(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    client = ShareFloatMockClient()
    client.responses["20261105"] = TushareRequestError("simulated timeout")
    service = DataIngestionService(
        settings=load_settings("config/default.yaml"),
        store=store,
        client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(TushareRequestError, match="simulated timeout"):
        service.init(("share_float",), "20261104", "20261105")
    assert store.status(get_dataset_spec("share_float")).max_date == "20261104"

    client.responses.clear()
    client.calls.clear()
    service.update(("share_float",), "20261105")

    assert client.calls == [("share_float", {"ann_date": "20261105", "limit": 6000, "offset": 0})]
    assert store.status(get_dataset_spec("share_float")).rows == 2


def test_share_float_rejects_oversized_page(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    client = ShareFloatMockClient()
    client.responses["20261105"] = share_float_rows(
        "20261105", [f"holder_{i}" for i in range(6001)]
    )
    service = DataIngestionService(
        settings=load_settings("config/default.yaml"),
        store=store,
        client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(DataValidationError, match="exceeds requested page size"):
        service.init(("share_float",), "20261105", "20261105")

    assert not store.status(get_dataset_spec("share_float")).exists


@pytest.mark.parametrize("ann_dates", [["20261104"], [None], ["20261105", "20261104"]])
def test_share_float_rejects_rows_for_another_announcement_date(tmp_path, ann_dates) -> None:
    store = ParquetDataStore(tmp_path)
    client = ShareFloatMockClient()
    client.responses["20261105"] = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * len(ann_dates),
            "ann_date": ann_dates,
            "float_date": ["20270115"] * len(ann_dates),
            "holder_name": ["holder"] * len(ann_dates),
            "share_type": ["placement"] * len(ann_dates),
        }
    )
    service = DataIngestionService(
        settings=load_settings("config/default.yaml"),
        store=store,
        client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(DataValidationError, match="announcement date"):
        service.init(("share_float",), "20261105", "20261105")

    assert not store.status(get_dataset_spec("share_float")).exists


def test_share_float_missing_announcement_column_fails_closed(tmp_path) -> None:
    client = ShareFloatMockClient()
    client.responses["20261105"] = pd.DataFrame(
        {"ts_code": ["000001.SZ"], "float_date": ["20270115"]}
    )
    service = DataIngestionService(
        settings=load_settings("config/default.yaml"),
        store=ParquetDataStore(tmp_path),
        client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(DataValidationError, match="ann_date"):
        service.init(("share_float",), "20261105", "20261105")


def test_share_float_parameter_errors_are_not_swallowed_as_empty_days(tmp_path) -> None:
    client = ShareFloatMockClient()
    client.responses["20261105"] = TushareParameterError("invalid announcement query")
    service = DataIngestionService(
        settings=load_settings("config/default.yaml"),
        store=ParquetDataStore(tmp_path),
        client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(TushareParameterError, match="invalid announcement query"):
        service.init(("share_float",), "20261105", "20261105")


def test_share_float_pages_use_configured_size_and_preserve_holder_and_share_type(tmp_path) -> None:
    settings = load_settings("config/default.yaml")
    settings.data.tushare_page_size = 2
    store = ParquetDataStore(tmp_path)
    client = ShareFloatMockClient()
    first = share_float_rows("20261105", ["holder_a", "holder_a"])
    first.loc[1, "share_type"] = "employee"
    client.responses[("20261105", 0)] = first
    client.responses[("20261105", 2)] = share_float_rows("20261105", ["holder_b"])
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    results = service.init(("share_float",), "20261105", "20261105")
    service.init(("share_float",), "20261105", "20261105")

    assert client.calls[:2] == [
        ("share_float", {"ann_date": "20261105", "limit": 2, "offset": 0}),
        ("share_float", {"ann_date": "20261105", "limit": 2, "offset": 2}),
    ]
    assert results[0].rows_written == 3
    frame = store.read_dataset(get_dataset_spec("share_float"))
    assert len(frame) == 3
    assert set(zip(frame["holder_name"], frame["share_type"], strict=True)) == {
        ("holder_a", "placement"),
        ("holder_a", "employee"),
        ("holder_b", "placement"),
    }


def test_share_float_full_6000_row_page_continues_to_next_page(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    client = ShareFloatMockClient()
    client.responses[("20261105", 0)] = share_float_rows(
        "20261105", [f"holder_{i}" for i in range(6000)]
    )
    client.responses[("20261105", 0)].loc[1, "holder_name"] = "holder_0"
    client.responses[("20261105", 0)].loc[1, "float_share"] = 2000.0
    client.responses[("20261105", 6000)] = share_float_rows("20261105", ["last_holder"])
    service = DataIngestionService(
        settings=load_settings("config/default.yaml"),
        store=store,
        client=client,  # type: ignore[arg-type]
    )

    results = service.init(("share_float",), "20261105", "20261105")

    assert [params["offset"] for _, params in client.calls] == [0, 6000]
    assert results[0].rows_written == 6001
    assert store.status(get_dataset_spec("share_float")).rows == 6001
    stored = store.read_dataset(get_dataset_spec("share_float"))
    assert stored.identity_status.eq("AMBIGUOUS").sum() == 2


@pytest.mark.parametrize("tail", [pd.DataFrame(), TushareNoDataError("no data")])
def test_share_float_exact_page_multiple_requires_empty_tail(tmp_path, tail) -> None:
    settings = load_settings("config/default.yaml")
    settings.data.tushare_page_size = 2
    store = ParquetDataStore(tmp_path)
    client = ShareFloatMockClient()
    client.responses[("20261105", 0)] = share_float_rows("20261105", ["a", "b"])
    client.responses[("20261105", 2)] = share_float_rows("20261105", ["c", "d"])
    client.responses[("20261105", 4)] = tail
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    service.init(("share_float",), "20261105", "20261105")

    assert [params["offset"] for _, params in client.calls] == [0, 2, 4]
    assert store.status(get_dataset_spec("share_float")).rows == 4


@pytest.mark.parametrize("second_holders", [["a", "b"], ["b", "a"], ["a"], ["b", "c"]])
def test_share_float_repeated_or_overlapping_pages_never_publish_partial_day(
    tmp_path, second_holders
) -> None:
    settings = load_settings("config/default.yaml")
    settings.data.tushare_page_size = 2
    store = ParquetDataStore(tmp_path)
    client = ShareFloatMockClient()
    client.responses[("20261105", 0)] = share_float_rows("20261105", ["a", "b"])
    client.responses[("20261105", 2)] = share_float_rows("20261105", second_holders)
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    with pytest.raises(DataValidationError, match="overlapping pagination"):
        service.init(("share_float",), "20261105", "20261105")

    assert len(client.calls) == 2
    assert not store.status(get_dataset_spec("share_float")).exists


def test_share_float_second_page_failure_does_not_advance_update_watermark(tmp_path) -> None:
    settings = load_settings("config/default.yaml")
    settings.data.tushare_page_size = 2
    store = ParquetDataStore(tmp_path)
    client = ShareFloatMockClient()
    client.responses[("20261105", 0)] = share_float_rows("20261105", ["a", "b"])
    client.responses[("20261105", 2)] = TushareRequestError("second page timeout")
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    with pytest.raises(TushareRequestError, match="second page timeout"):
        service.init(("share_float",), "20261104", "20261105")
    assert store.status(get_dataset_spec("share_float")).max_date == "20261104"

    client.responses[("20261105", 2)] = share_float_rows("20261105", ["c"])
    client.calls.clear()
    service.update(("share_float",), "20261105")

    assert [params["offset"] for _, params in client.calls] == [0, 2]
    assert store.status(get_dataset_spec("share_float")).rows == 4


def test_share_float_validates_identity_on_every_page_before_writing(tmp_path) -> None:
    settings = load_settings("config/default.yaml")
    settings.data.tushare_page_size = 2
    store = ParquetDataStore(tmp_path)
    client = ShareFloatMockClient()
    client.responses[("20261105", 0)] = share_float_rows("20261105", ["a", "b"])
    client.responses[("20261105", 2)] = share_float_rows("20261104", ["c"])
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    with pytest.raises(DataValidationError, match="announcement date"):
        service.init(("share_float",), "20261105", "20261105")
    assert not store.status(get_dataset_spec("share_float")).exists


def test_share_float_bounds_pages_without_publishing_unfinished_day(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("ashare_quant.data.ingestion.MAX_ANNOUNCEMENT_PAGES", 2, raising=False)
    settings = load_settings("config/default.yaml")
    settings.data.tushare_page_size = 1
    store = ParquetDataStore(tmp_path)
    client = ShareFloatMockClient()
    client.responses[("20261105", 0)] = share_float_rows("20261105", ["a"])
    client.responses[("20261105", 1)] = share_float_rows("20261105", ["b"])
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    with pytest.raises(DataValidationError, match="page budget"):
        service.init(("share_float",), "20261105", "20261105")
    assert len(client.calls) == 2
    assert not store.status(get_dataset_spec("share_float")).exists


@pytest.mark.parametrize("different_value", [2000.0, None])
def test_share_float_different_detail_values_are_retained(tmp_path, different_value) -> None:
    client = ShareFloatMockClient()
    frame = share_float_rows("20261105", ["a", "a"])
    frame.loc[1, "float_share"] = different_value
    client.responses[("20261105", 0)] = frame
    store = ParquetDataStore(tmp_path)
    service = DataIngestionService(
        settings=load_settings("config/default.yaml"),
        store=store,
        client=client,  # type: ignore[arg-type]
    )

    service.init(("share_float",), "20261105", "20261105")
    stored = store.read_dataset(get_dataset_spec("share_float"))
    assert len(stored) == 2
    assert set(stored.identity_status) == {"AMBIGUOUS"}
    assert set(stored.identity_variant_count) == {2}
    values = [json.loads(raw)["float_share"] for raw in stored.raw_row_json]
    assert 1000.0 in values
    assert different_value in values
    assert not stored.legacy_provenance.any()
    assert all(
        json.loads(value)
        == [
            {
                "endpoint": "share_float",
                "params": {"ann_date": "20261105", "limit": 6000, "offset": 0},
            }
        ]
        for value in stored.source_requests_json
    )


def test_share_float_ambiguous_complete_day_advances_watermark(tmp_path) -> None:
    client = ShareFloatMockClient()
    frame = share_float_rows("20261105", ["a"] * 10 + ["b"] * 2)
    frame["float_share"] = list(range(12))
    client.responses["20261105"] = frame
    store = ParquetDataStore(tmp_path)
    service = DataIngestionService(
        settings=load_settings("config/default.yaml"),
        store=store,
        client=client,  # type: ignore[arg-type]
    )
    service.init(("share_float",), "20261104", "20261105")
    spec = get_dataset_spec("share_float")
    stored = store.read_dataset(spec, "20261105", "20261105")
    assert len(stored) == 12
    assert set(stored.identity_status) == {"AMBIGUOUS"}
    assert set(stored.identity_variant_count) == {10, 2}
    assert store.status(spec).max_date == "20261105"


def test_share_float_distinct_observations_for_same_key_across_pages(tmp_path) -> None:
    settings = load_settings("config/default.yaml")
    settings.data.tushare_page_size = 2
    client = ShareFloatMockClient()
    client.responses[("20261105", 0)] = share_float_rows("20261105", ["a", "b"])
    second = share_float_rows("20261105", ["a"])
    second["float_share"] = 2000.0
    client.responses[("20261105", 2)] = second
    store = ParquetDataStore(tmp_path)
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]
    service.init(("share_float",), "20261105", "20261105")
    stored = store.read_dataset(get_dataset_spec("share_float"))
    assert len(stored) == 3
    assert set(stored.loc[stored.holder_name.eq("a"), "identity_status"]) == {"AMBIGUOUS"}
    assert set(stored.loc[stored.holder_name.eq("b"), "identity_status"]) == {"SINGLE_OBSERVATION"}
    assert set(
        json.loads(value)[0]["params"]["offset"] for value in stored.source_requests_json
    ) == {0, 2}


def test_share_float_ambiguous_page_does_not_bypass_completion_requirement(tmp_path) -> None:
    settings = load_settings("config/default.yaml")
    settings.data.tushare_page_size = 2
    client = ShareFloatMockClient()
    frame = share_float_rows("20261105", ["a", "a"])
    frame.loc[1, "float_share"] = 2000.0
    client.responses[("20261105", 0)] = frame
    client.responses[("20261105", 2)] = TushareRequestError("second page failed")
    store = ParquetDataStore(tmp_path)
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]
    with pytest.raises(TushareRequestError, match="second page failed"):
        service.init(("share_float",), "20261104", "20261105")
    assert store.status(get_dataset_spec("share_float")).max_date == "20261104"


def test_share_float_exact_duplicate_detail_is_idempotent(tmp_path) -> None:
    client = ShareFloatMockClient()
    client.responses[("20261105", 0)] = share_float_rows("20261105", ["a", "a"])
    store = ParquetDataStore(tmp_path)
    service = DataIngestionService(
        settings=load_settings("config/default.yaml"),
        store=store,
        client=client,  # type: ignore[arg-type]
    )

    service.init(("share_float",), "20261105", "20261105")

    assert store.status(get_dataset_spec("share_float")).rows == 1


@pytest.mark.parametrize("field", ["holder_name", "share_type"])
@pytest.mark.parametrize("issue", ["missing", "null"])
def test_share_float_missing_detail_identity_fails_closed(tmp_path, field, issue) -> None:
    frame = share_float_rows("20261105", ["holder"])
    if issue == "missing":
        frame = frame.drop(columns=[field])
    else:
        frame[field] = None
    client = ShareFloatMockClient()
    client.responses[("20261105", 0)] = frame
    store = ParquetDataStore(tmp_path)
    service = DataIngestionService(
        settings=load_settings("config/default.yaml"),
        store=store,
        client=client,  # type: ignore[arg-type]
    )

    message = "missing required columns" if issue == "missing" else "null detail identity"
    with pytest.raises(DataValidationError, match=message):
        service.init(("share_float",), "20261105", "20261105")
    assert not store.status(get_dataset_spec("share_float")).exists


def test_share_float_schema_changes_between_pages_fail_before_writing(tmp_path) -> None:
    settings = load_settings("config/default.yaml")
    settings.data.tushare_page_size = 2
    client = ShareFloatMockClient()
    client.responses[("20261105", 0)] = share_float_rows("20261105", ["a", "b"])
    client.responses[("20261105", 2)] = share_float_rows("20261105", ["c"]).drop(
        columns=["float_share"]
    )
    store = ParquetDataStore(tmp_path)
    service = DataIngestionService(settings=settings, store=store, client=client)  # type: ignore[arg-type]

    with pytest.raises(DataValidationError, match="schema changed between pages"):
        service.init(("share_float",), "20261105", "20261105")
    assert not store.status(get_dataset_spec("share_float")).exists


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
