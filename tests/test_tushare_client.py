from __future__ import annotations

import pandas as pd
import pytest

from ashare_quant.data.exceptions import (
    TushareNoDataError,
    TushareParameterError,
    TusharePermissionError,
)
from ashare_quant.data.tushare_client import TushareClient, TushareClientConfig


class FakeApi:
    def __init__(self) -> None:
        self.calls = 0

    def daily(self, **_: object) -> pd.DataFrame:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary network error")
        return pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20240102"]})

    def daily_basic(self, **_: object) -> pd.DataFrame:
        raise RuntimeError("permission denied for endpoint")

    def income(self, **_: object) -> pd.DataFrame:
        self.calls += 1
        raise RuntimeError("必填参数, ts_code")

    def cyq_chips(self, **_: object) -> pd.DataFrame:
        self.calls += 1
        raise RuntimeError("指定数据不存在，请确认参数！")

    def cyq_perf(self, **_: object) -> pd.DataFrame:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("抱歉，您访问接口(cyq_perf)频率超限(200次/分钟)")
        return pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20240102"]})


def fast_config() -> TushareClientConfig:
    return TushareClientConfig(
        retry_attempts=2,
        rate_limit_per_minute=1_000_000,
        request_interval_seconds=0.0,
        backoff_base_seconds=0.0,
        backoff_max_seconds=0.0,
    )


def test_tushare_client_retries_and_records_stats() -> None:
    api = FakeApi()
    client = TushareClient(token="token", config=fast_config(), api=api)

    frame = client.query("daily", trade_date="20240102")

    assert len(frame) == 1
    assert api.calls == 2
    assert client.stats.total == 2
    assert client.stats.retried == 1
    assert client.stats.succeeded == 1


def test_tushare_client_permission_error_is_explicit() -> None:
    client = TushareClient(token="token", config=fast_config(), api=FakeApi())

    with pytest.raises(TusharePermissionError, match="daily_basic"):
        client.query("daily_basic", trade_date="20240102")

    assert client.stats.permission_errors == 1


def test_tushare_client_does_not_retry_parameter_errors() -> None:
    api = FakeApi()
    client = TushareClient(token="token", config=fast_config(), api=api)

    with pytest.raises(TushareParameterError, match="ts_code"):
        client.query("income", period="20231231")

    assert api.calls == 1
    assert client.stats.total == 1
    assert client.stats.retried == 0
    assert client.stats.failed == 1


def test_tushare_client_does_not_retry_no_data_errors() -> None:
    api = FakeApi()
    client = TushareClient(token="token", config=fast_config(), api=api)

    with pytest.raises(TushareNoDataError, match="No data"):
        client.query("cyq_chips", ts_code="000003.SZ")

    assert api.calls == 1
    assert client.stats.total == 1
    assert client.stats.retried == 0
    assert client.stats.failed == 1


def test_tushare_client_cools_down_after_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("ashare_quant.data.tushare_client.time.sleep", sleeps.append)
    api = FakeApi()
    client = TushareClient(token="token", config=fast_config(), api=api)

    frame = client.query("cyq_perf", ts_code="000001.SZ")

    assert len(frame) == 1
    assert api.calls == 2
    assert client.stats.rate_limit_errors == 1
    assert client.stats.retried == 1
    assert max(sleeps) >= 60.0


def test_tushare_client_uses_endpoint_specific_rate_limit() -> None:
    config = fast_config()
    config = TushareClientConfig(
        retry_attempts=config.retry_attempts,
        rate_limit_per_minute=config.rate_limit_per_minute,
        request_interval_seconds=config.request_interval_seconds,
        backoff_base_seconds=config.backoff_base_seconds,
        backoff_max_seconds=config.backoff_max_seconds,
        endpoint_rate_limits_per_minute={"cyq_chips": 200},
    )

    client = TushareClient(token="token", config=config, api=FakeApi())

    assert client._endpoint_rate_limiters["cyq_chips"]._min_interval == pytest.approx(0.3)
