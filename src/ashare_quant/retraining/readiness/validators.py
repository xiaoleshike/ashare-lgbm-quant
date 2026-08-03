"""Shared read-only file and identity validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import file_sha256


class SourceTracker:
    """Track every byte source consumed by readiness validation."""

    def __init__(self) -> None:
        self.hashes: dict[str, str] = {}

    def json(self, path: Path, description: str) -> dict[str, Any]:
        if not path.is_file():
            raise DataValidationError(f"required {description} is missing: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DataValidationError(f"invalid {description}: {error}") from error
        if not isinstance(payload, dict):
            raise DataValidationError(f"{description} must contain an object")
        self.track(path)
        return payload

    def track(self, path: Path) -> None:
        if not path.is_file():
            raise DataValidationError(f"required readiness source is missing: {path}")
        self.hashes[str(path)] = file_sha256(path)


def require_string(payload: dict[str, Any], name: str, description: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise DataValidationError(f"{description} lacks {name}")
    return value
