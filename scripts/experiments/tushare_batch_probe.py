#!/usr/bin/env python3
"""Benchmark safe Tushare batch-query alternatives without writing project data."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from ashare_quant.config import load_settings
from ashare_quant.data.tushare_client import TushareClient, TushareClientConfig

DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"


@dataclass(frozen=True)
class MethodResult:
    name: str
    requests: int
    seconds: float
    rows: int
    duplicate_keys: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Tushare daily cross-section and multi-code range queries."
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--start-date", required=True, help="YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="YYYYMMDD")
    parser.add_argument("--sample-stocks", type=int, default=20)
    parser.add_argument(
        "--finance-period",
        help="Optionally probe a VIP financial endpoint for a report period, e.g. 20260331.",
    )
    parser.add_argument(
        "--finance-endpoint",
        choices=("income_vip", "balancesheet_vip", "cashflow_vip", "fina_indicator_vip"),
        default="income_vip",
    )
    return parser.parse_args()


def build_client(config_path: str) -> TushareClient:
    settings = load_settings(config_path)
    return TushareClient(
        token=settings.tushare_token,
        config=TushareClientConfig(
            retry_attempts=settings.data.retry_attempts,
            rate_limit_per_minute=settings.data.rate_limit_per_minute,
            request_interval_seconds=settings.data.request_interval_seconds,
            backoff_base_seconds=settings.data.backoff_base_seconds,
            backoff_max_seconds=settings.data.backoff_max_seconds,
        ),
    )


def open_dates(client: TushareClient, start_date: str, end_date: str) -> list[str]:
    frame = client.query(
        "trade_cal",
        exchange="SSE",
        start_date=start_date,
        end_date=end_date,
        is_open="1",
        fields="cal_date,is_open",
    )
    return sorted(frame.loc[frame["is_open"].astype(int) == 1, "cal_date"].astype(str).unique())


def sample_codes(client: TushareClient, limit: int) -> list[str]:
    frame = client.query("stock_basic", list_status="L", fields="ts_code")
    return sorted(frame["ts_code"].dropna().astype(str).unique())[:limit]


def timed_daily_cross_sections(
    client: TushareClient, dates: list[str], selected_codes: set[str]
) -> tuple[pd.DataFrame, MethodResult]:
    started = time.perf_counter()
    frames = [client.query("daily", trade_date=date, fields=DAILY_FIELDS) for date in dates]
    elapsed = time.perf_counter() - started
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    selected = combined.loc[combined["ts_code"].astype(str).isin(selected_codes)].copy()
    result = MethodResult(
        name="trade_date_cross_sections",
        requests=len(dates),
        seconds=elapsed,
        rows=len(combined),
        duplicate_keys=duplicate_count(combined),
    )
    return selected, result


def timed_multi_code_range(
    client: TushareClient, start_date: str, end_date: str, codes: list[str]
) -> tuple[pd.DataFrame, MethodResult]:
    started = time.perf_counter()
    frame = client.query(
        "daily",
        ts_code=",".join(codes),
        start_date=start_date,
        end_date=end_date,
        fields=DAILY_FIELDS,
    )
    elapsed = time.perf_counter() - started
    result = MethodResult(
        name="multi_code_date_range",
        requests=1,
        seconds=elapsed,
        rows=len(frame),
        duplicate_keys=duplicate_count(frame),
    )
    return frame, result


def duplicate_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int(frame.duplicated(["ts_code", "trade_date"]).sum())


def compare_frames(expected: pd.DataFrame, actual: pd.DataFrame) -> dict[str, Any]:
    keys = ["ts_code", "trade_date"]
    value_columns = ["open", "high", "low", "close", "vol", "amount"]
    expected_keys = set(map(tuple, expected[keys].astype(str).itertuples(index=False, name=None)))
    actual_keys = set(map(tuple, actual[keys].astype(str).itertuples(index=False, name=None)))
    merged = expected[keys + value_columns].merge(
        actual[keys + value_columns], on=keys, how="inner", suffixes=("_daily", "_batch")
    )
    mismatches: dict[str, int] = {}
    for column in value_columns:
        left = pd.to_numeric(merged[f"{column}_daily"], errors="coerce")
        right = pd.to_numeric(merged[f"{column}_batch"], errors="coerce")
        mismatches[column] = int((~left.eq(right) & ~(left.isna() & right.isna())).sum())
    return {
        "expected_sample_rows": len(expected),
        "batch_rows": len(actual),
        "missing_keys": len(expected_keys - actual_keys),
        "unexpected_keys": len(actual_keys - expected_keys),
        "value_mismatches": mismatches,
    }


def probe_finance_period(
    client: TushareClient, endpoint: str, period: str, page_size: int
) -> dict[str, Any]:
    started = time.perf_counter()
    frames: list[pd.DataFrame] = []
    offset = 0
    try:
        while True:
            frame = client.query(
                endpoint, period=period, limit=page_size, offset=offset
            )
            frames.append(frame)
            if len(frame) < page_size:
                break
            offset += page_size
    except Exception as error:  # noqa: BLE001 - this is a diagnostic probe.
        return {
            "supported": False,
            "endpoint": endpoint,
            "period": period,
            "error": str(error),
        }
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return {
        "supported": True,
        "endpoint": endpoint,
        "period": period,
        "seconds": time.perf_counter() - started,
        "requests": len(frames),
        "rows": len(combined),
        "stocks": int(combined["ts_code"].nunique()) if "ts_code" in combined else 0,
        "duplicate_rows": int(combined.duplicated().sum()),
    }


def main() -> int:
    args = parse_args()
    if args.sample_stocks < 1:
        raise SystemExit("--sample-stocks must be positive")
    client = build_client(args.config)
    dates = open_dates(client, args.start_date, args.end_date)
    codes = sample_codes(client, args.sample_stocks)
    if not dates:
        raise SystemExit("No open trading dates found in the requested range")
    if not codes:
        raise SystemExit("No listed stock codes returned")

    daily_sample, daily_result = timed_daily_cross_sections(client, dates, set(codes))
    batch_frame, batch_result = timed_multi_code_range(
        client, args.start_date, args.end_date, codes
    )
    output: dict[str, Any] = {
        "range": {"start": args.start_date, "end": args.end_date, "open_dates": dates},
        "sample_stock_count": len(codes),
        "methods": [asdict(daily_result), asdict(batch_result)],
        "comparison": compare_frames(daily_sample, batch_frame),
        "interpretation": (
            "Multi-code ranges are safe only while expected rows remain below the endpoint's "
            "6000-row cap; full-market daily cross-sections avoid silent truncation."
        ),
    }
    if args.finance_period:
        settings = load_settings(args.config)
        output["finance_period_probe"] = probe_finance_period(
            client,
            args.finance_endpoint,
            args.finance_period,
            settings.data.tushare_page_size,
        )
    output["request_stats"] = asdict(client.stats)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
