"""Official Tushare client wrapper with retry, pacing, and diagnostics."""

from __future__ import annotations

import logging
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, cast

import pandas as pd
import tushare as ts  # type: ignore[import-untyped]
from pydantic import SecretStr

from ashare_quant.data.exceptions import (
    TushareNoDataError,
    TushareParameterError,
    TusharePermissionError,
    TushareRequestError,
    TushareTokenError,
)

LOGGER = logging.getLogger(__name__)
type DataFrame = pd.DataFrame


class TushareApiProtocol(Protocol):
    """Minimal protocol for the dynamic Tushare Pro API object."""

    def __getattr__(self, name: str) -> Callable[..., object]:
        """Return a callable endpoint from the official client."""


@dataclass(slots=True)
class RequestStats:
    """Collect request counters for diagnostics and status output."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    retried: int = 0
    permission_errors: int = 0
    rate_limit_errors: int = 0


@dataclass(frozen=True, slots=True)
class TushareClientConfig:
    """Runtime controls for Tushare request behavior."""

    retry_attempts: int
    rate_limit_per_minute: int
    request_interval_seconds: float
    backoff_base_seconds: float
    backoff_max_seconds: float
    endpoint_rate_limits_per_minute: dict[str, int] = field(default_factory=dict)


class RateLimiter:
    """Simple process-local rate limiter and request pacer."""

    def __init__(self, requests_per_minute: int, min_interval_seconds: float) -> None:
        self._min_interval = max(min_interval_seconds, 60.0 / requests_per_minute)
        self._last_request_at = 0.0

    def slow_down_to(self, requests_per_minute: int) -> None:
        """Reduce the effective request rate for the rest of the process."""

        safe_requests_per_minute = max(int(requests_per_minute * 0.9), 1)
        self._min_interval = max(self._min_interval, 60.0 / safe_requests_per_minute)

    def wait(self) -> None:
        """Sleep until the next request is allowed."""

        now = time.monotonic()
        wait_seconds = self._last_request_at + self._min_interval - now
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        self._last_request_at = time.monotonic()


class TushareClient:
    """Resilient wrapper around the official Tushare Pro API client."""

    def __init__(
        self,
        token: SecretStr | str | None,
        config: TushareClientConfig,
        api: TushareApiProtocol | None = None,
    ) -> None:
        if token is None:
            raise TushareTokenError("TUSHARE_TOKEN is required for data ingestion commands")
        token_value = token.get_secret_value() if isinstance(token, SecretStr) else token
        if not token_value:
            raise TushareTokenError("TUSHARE_TOKEN is empty")

        self._api = api or cast(TushareApiProtocol, ts.pro_api(token_value))
        self._config = config
        self._rate_limiter = RateLimiter(
            requests_per_minute=config.rate_limit_per_minute,
            min_interval_seconds=config.request_interval_seconds,
        )
        self._endpoint_rate_limiters = {
            endpoint: RateLimiter(
                requests_per_minute=requests_per_minute,
                min_interval_seconds=config.request_interval_seconds,
            )
            for endpoint, requests_per_minute in config.endpoint_rate_limits_per_minute.items()
        }
        self.stats = RequestStats()

    def query(self, endpoint: str, **params: object) -> DataFrame:
        """Call one Tushare endpoint and return a DataFrame.

        Permission failures are not retried because retrying cannot make an account
        eligible for an endpoint. Other failures use exponential backoff.
        """

        last_error: Exception | None = None
        for attempt in range(1, self._config.retry_attempts + 1):
            self.stats.total += 1
            self._rate_limiter.wait()
            endpoint_rate_limiter = self._endpoint_rate_limiters.get(endpoint)
            if endpoint_rate_limiter is not None:
                endpoint_rate_limiter.wait()
            try:
                safe_params = {key: value for key, value in params.items() if key != "token"}
                LOGGER.info(
                    "tushare request started",
                    extra={"endpoint": endpoint, "attempt": attempt, "params": safe_params},
                )
                result = getattr(self._api, endpoint)(**params)
                if not isinstance(result, pd.DataFrame):
                    raise TushareRequestError(
                        f"Endpoint {endpoint} returned {type(result).__name__}, expected DataFrame"
                    )
                self.stats.succeeded += 1
                LOGGER.info(
                    "tushare request succeeded",
                    extra={
                        "endpoint": endpoint,
                        "rows": len(result),
                        "attempt": attempt,
                        "params": safe_params,
                    },
                )
                return result.copy()
            except Exception as error:  # noqa: BLE001 - official client raises broad exceptions.
                if self._is_permission_error(error):
                    self.stats.permission_errors += 1
                    self.stats.failed += 1
                    message = self._permission_message(endpoint, error)
                    LOGGER.warning(message, extra={"endpoint": endpoint})
                    raise TusharePermissionError(message) from error

                if self._is_parameter_error(error):
                    self.stats.failed += 1
                    message = f"Invalid parameters for Tushare endpoint '{endpoint}': {error}"
                    LOGGER.error(message, extra={"endpoint": endpoint, "params": safe_params})
                    raise TushareParameterError(message) from error

                if self._is_no_data_error(error):
                    self.stats.failed += 1
                    message = f"No data for Tushare endpoint '{endpoint}': {error}"
                    LOGGER.info(message, extra={"endpoint": endpoint, "params": safe_params})
                    raise TushareNoDataError(message) from error

                if self._is_rate_limit_error(error):
                    self.stats.rate_limit_errors += 1
                    limit_per_minute = self._extract_rate_limit_per_minute(error)
                    if limit_per_minute is not None:
                        limiter = self._endpoint_rate_limiters.setdefault(
                            endpoint,
                            RateLimiter(
                                requests_per_minute=self._config.rate_limit_per_minute,
                                min_interval_seconds=self._config.request_interval_seconds,
                            ),
                        )
                        limiter.slow_down_to(limit_per_minute)
                    last_error = error
                    if attempt >= self._config.retry_attempts:
                        break
                    self.stats.retried += 1
                    sleep_seconds = max(60.0, self._backoff_seconds(attempt))
                    LOGGER.warning(
                        "tushare request rate limited; retrying after cooldown",
                        extra={
                            "endpoint": endpoint,
                            "attempt": attempt,
                            "limit_per_minute": limit_per_minute,
                            "sleep_seconds": sleep_seconds,
                            "params": safe_params,
                        },
                        exc_info=error,
                    )
                    time.sleep(sleep_seconds)
                    continue

                last_error = error
                if attempt >= self._config.retry_attempts:
                    break
                self.stats.retried += 1
                sleep_seconds = self._backoff_seconds(attempt)
                LOGGER.warning(
                    "tushare request failed; retrying",
                    extra={
                        "endpoint": endpoint,
                        "attempt": attempt,
                        "sleep_seconds": sleep_seconds,
                        "params": safe_params,
                    },
                    exc_info=error,
                )
                time.sleep(sleep_seconds)

        self.stats.failed += 1
        message = f"Endpoint {endpoint} failed after {self._config.retry_attempts} attempts"
        raise TushareRequestError(message) from last_error

    def _backoff_seconds(self, attempt: int) -> float:
        base = self._config.backoff_base_seconds * (2 ** (attempt - 1))
        jitter = random.uniform(0, self._config.backoff_base_seconds)  # noqa: S311
        return float(min(base + jitter, self._config.backoff_max_seconds))

    @staticmethod
    def _is_permission_error(error: Exception) -> bool:
        text = str(error).lower()
        markers = (
            "permission",
            "privilege",
            "forbidden",
            "没有权限",
            "无权限",
            "权限",
            "积分",
            "access denied",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _permission_message(endpoint: str, error: Exception) -> str:
        return (
            f"Tushare account lacks permission for endpoint '{endpoint}'. Original message: {error}"
        )

    @staticmethod
    def _is_parameter_error(error: Exception) -> bool:
        text = str(error).lower()
        markers = (
            "参数校验失败",
            "必填参数",
            "至少填写一个",
            "至少输入一个",
            "invalid parameter",
            "missing parameter",
            "parameter required",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _is_no_data_error(error: Exception) -> bool:
        text = str(error).lower()
        markers = (
            "指定数据不存在",
            "数据不存在",
            "no data",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        text = str(error).lower()
        markers = (
            "频率超限",
            "rate limit",
            "too many requests",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _extract_rate_limit_per_minute(error: Exception) -> int | None:
        match = re.search(r"(\d+)\s*次/分钟", str(error))
        if match is None:
            return None
        return int(match.group(1))
