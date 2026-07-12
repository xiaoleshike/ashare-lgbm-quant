"""Validation checks for point-in-time universe outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ashare_quant.universe.storage import UniverseStore

type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class UniverseValidationResult:
    """Validation result for stored or in-memory universe rows."""

    ok: bool
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


class UniverseValidator:
    """Run consistency checks required by the point-in-time universe layer."""

    def __init__(self, store: UniverseStore) -> None:
        self._store = store

    def validate(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> UniverseValidationResult:
        """Validate stored universe rows over an optional date range."""

        return validate_universe_frame(self._store.read(start_date, end_date))


def validate_universe_frame(frame: DataFrame) -> UniverseValidationResult:
    """Validate one universe frame."""

    errors: list[str] = []
    warnings: list[str] = []
    if frame.empty:
        warnings.append("universe is empty or not built")
        return UniverseValidationResult(ok=True, warnings=tuple(warnings))

    if frame.duplicated(subset=["trade_date", "ts_code"]).any():
        duplicate_count = int(frame.duplicated(subset=["trade_date", "ts_code"]).sum())
        errors.append(f"duplicate universe rows={duplicate_count}")

    if (pd.to_numeric(frame["list_days"], errors="coerce") < 0).any():
        errors.append("list_days contains negative values")

    suspended = frame["is_suspended"].astype(bool)
    if (suspended & frame["can_buy"].astype(bool)).any():
        errors.append("suspended stocks cannot be can_buy=true")
    if (suspended & frame["can_sell"].astype(bool)).any():
        errors.append("suspended stocks cannot be can_sell=true")

    if (frame["is_limit_up"].astype(bool) & frame["can_buy"].astype(bool)).any():
        errors.append("limit-up stocks must be can_buy=false under default execution")
    if (frame["is_limit_down"].astype(bool) & frame["can_sell"].astype(bool)).any():
        errors.append("limit-down stocks must be can_sell=false under default execution")

    model = frame["in_model_universe"].astype(bool)
    if (model & ~frame["in_base_universe"].astype(bool)).any():
        errors.append("in_model_universe implies in_base_universe")
    if (model & frame["is_st"].astype(bool)).any():
        errors.append("in_model_universe implies is_st=false")
    if (model & frame["is_suspended"].astype(bool)).any():
        errors.append("in_model_universe implies is_suspended=false")
    if (model & frame["is_low_liquidity"].astype(bool)).any():
        errors.append("in_model_universe implies is_low_liquidity=false")

    return UniverseValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))
