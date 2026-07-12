#!/usr/bin/env python3
"""Run non-blocking baostock previous-trading-day checks and log quality issues."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ashare_quant.config import load_settings
from ashare_quant.data.quality_logging import append_quality_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baostock previous-day quote check.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--log-root")
    return parser.parse_args()


def extract_json_array(output: str) -> list[dict[str, object]]:
    start = output.find("[")
    end = output.rfind("]")
    if start < 0 or end < start:
        return []
    parsed = json.loads(output[start : end + 1])
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def has_issue(result: dict[str, object]) -> bool:
    keys = (
        "only_baostock_count",
        "only_local_count",
        "mismatched_rows",
    )
    return any(int(result.get(key, 0) or 0) > 0 for key in keys)


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    log_root = Path(args.log_root) if args.log_root else settings.paths.data_quality_logs
    checker = Path(__file__).with_name("compare_baostock_stock_list.py")
    command = [
        sys.executable,
        str(checker),
        "--config",
        args.config,
        "--check",
        "quotes",
        "--recent-trading-days",
        "1",
        "--baostock-status",
        "trading",
        "--format",
        "json",
        "--sample-size",
        "20",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    if completed.returncode != 0:
        append_quality_event(
            log_root,
            {
                "event": "baostock_previous_day_check_failed",
                "severity": "error",
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            },
        )
        return completed.returncode
    results = extract_json_array(completed.stdout)
    for result in results:
        if has_issue(result):
            append_quality_event(
                log_root,
                {
                    "event": "baostock_previous_day_check_issue",
                    "severity": "error",
                    "result": result,
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
