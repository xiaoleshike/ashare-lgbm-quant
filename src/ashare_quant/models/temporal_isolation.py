"""Authoritative horizon-safe temporal isolation for research folds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ashare_quant.data.exceptions import DataValidationError

type GapSetting = int | Literal["auto"]


@dataclass(frozen=True, slots=True)
class TemporalGapResolution:
    """Frozen purge and embargo resolution for one horizon scope."""

    gap_policy: Literal["AUTO", "EXPLICIT"]
    configured_purge: GapSetting
    configured_embargo: GapSetting
    resolved_purge: int
    resolved_embargo: int
    required_gap: int
    horizons: tuple[int, ...]
    label_semantics: str = "signal_close_t_entry_t_plus_1_exit_after_h_sessions"


def required_temporal_gap_sessions(horizon: int) -> int:
    """Return sessions needed for a next-open forward-return label to mature."""

    if horizon <= 0:
        raise DataValidationError("label horizon must be positive")
    return horizon + 1


def resolve_temporal_gaps(
    horizons: tuple[int, ...],
    *,
    purge: GapSetting,
    embargo: GapSetting,
) -> TemporalGapResolution:
    """Resolve AUTO gaps or reject explicit values below the strictest horizon."""

    if not horizons:
        raise DataValidationError("temporal gap resolution requires at least one horizon")
    required = max(required_temporal_gap_sessions(value) for value in horizons)
    resolved_purge = _resolve_one("purge", purge, required)
    resolved_embargo = _resolve_one("embargo", embargo, required)
    policy: Literal["AUTO", "EXPLICIT"] = (
        "AUTO" if purge == "auto" and embargo == "auto" else "EXPLICIT"
    )
    return TemporalGapResolution(
        gap_policy=policy,
        configured_purge=purge,
        configured_embargo=embargo,
        resolved_purge=resolved_purge,
        resolved_embargo=resolved_embargo,
        required_gap=required,
        horizons=tuple(sorted(set(horizons))),
    )


def _resolve_one(name: str, value: GapSetting, required: int) -> int:
    if value == "auto":
        return required
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataValidationError(f"{name}_sessions must be auto or a non-negative integer")
    if value < required:
        raise DataValidationError(
            f"unsafe explicit {name}_sessions: configured={value} required={required}"
        )
    return value
