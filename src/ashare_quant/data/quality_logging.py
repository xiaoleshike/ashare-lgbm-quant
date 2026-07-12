"""Data-quality error logging utilities."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from ashare_quant.data.validation import ValidationResult


def today_yyyymmdd() -> str:
    """Return local calendar date for organizing quality logs."""

    return datetime.now().strftime("%Y%m%d")


def quality_log_path(log_root: Path, date: str | None = None) -> Path:
    """Return the JSONL data-quality error log path for one local date."""

    day = date or today_yyyymmdd()
    return log_root / day / "errors.jsonl"


def append_quality_event(log_root: Path, event: dict[str, Any], date: str | None = None) -> Path:
    """Append one structured data-quality event to the date-partitioned error log."""

    path = quality_log_path(log_root, date)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **event,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def append_validation_results(log_root: Path, results: Sequence[ValidationResult]) -> None:
    """Append validation warnings and errors for a sequence of ValidationResult-like objects."""

    for result in results:
        for warning in result.warnings:
            append_quality_event(
                log_root,
                {
                    "event": "post_ingestion_validation_warning",
                    "dataset": result.dataset,
                    "severity": "warning",
                    "message": warning,
                },
            )
        for error in result.errors:
            append_quality_event(
                log_root,
                {
                    "event": "post_ingestion_validation_error",
                    "dataset": result.dataset,
                    "severity": "error",
                    "message": error,
                },
            )
