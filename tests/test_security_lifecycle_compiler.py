from __future__ import annotations

import pandas as pd
import pytest

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_lifecycle_compiler import (
    LifecycleIntervalCompiler,
)

CALENDAR = ("20100104", "20100105", "20100106", "20100301")


def test_resume_closes_listing_suspension_on_previous_open_session() -> None:
    result = _compiler().compile(
        _events(_start("20100104"), _close("FORMAL_LISTING_RESUMPTION", "20100301")),
        base_events=(),
    )

    assert result.events[0]["effective_from"] == "20100104"
    assert result.events[0]["effective_to"] == "20100106"
    assert result.report["counts"]["resumption_closed"] == 1


def test_terminal_closes_listing_suspension_without_overlap() -> None:
    result = _compiler().compile(
        _events(_start("20100104"), _close("TERMINAL_DELISTING", "20100301")),
        base_events=(),
    )

    assert result.events[0]["effective_to"] == "20100106"
    assert result.report["counts"]["terminal_closed"] == 1


def test_pre_period_start_is_clipped_and_recorded_as_carry_in() -> None:
    compiler = LifecycleIntervalCompiler(
        open_trade_dates=("20091231", *CALENDAR),
        research_start="20100101",
        research_end="20100301",
    )
    result = compiler.compile(
        _events(_start("20091231"), _close("FORMAL_LISTING_RESUMPTION", "20100106")),
        base_events=(),
    )

    assert result.events[0]["effective_from"] == "20100101"
    assert result.events[0]["effective_to"] == "20100105"
    assert result.report["counts"]["carry_in"] == 1


def test_unclosed_start_fails_closed_when_closure_is_required() -> None:
    with pytest.raises(DataValidationError, match="LISTING_SUSPENSION_INTERVAL_UNCLOSED"):
        _compiler().compile(_events(_start("20100104")), base_events=(), require_closed=True)


def test_resume_without_start_fails_closed() -> None:
    with pytest.raises(DataValidationError, match="RESUME_WITHOUT_START"):
        _compiler().compile(
            _events(_close("FORMAL_LISTING_RESUMPTION", "20100301")),
            base_events=(),
            require_closed=True,
        )


def test_double_start_fails_closed() -> None:
    with pytest.raises(DataValidationError, match="DOUBLE_START"):
        _compiler().compile(
            _events(_start("20100104"), _start("20100105")),
            base_events=(),
            require_closed=True,
        )


def test_exact_base_interval_remains_single_runtime_event() -> None:
    base = (
        {
            "canonical_ts_code": "600001.SH",
            "event_type": "LISTING_SUSPENDED",
            "effective_from": "20100104",
            "effective_to": "20100106",
            "evidence_source": "v3",
            "evidence_reference": "v3-row",
        },
    )
    start = _start("20100104")
    start["effective_end"] = "20100106"
    result = _compiler().compile(_events(start), base_events=base, require_closed=True)

    assert result.events == ()
    assert result.report["counts"]["base_events_covering_official_starts"] == 1


def _compiler() -> LifecycleIntervalCompiler:
    return LifecycleIntervalCompiler(
        open_trade_dates=CALENDAR,
        research_start="20100101",
        research_end="20100301",
    )


def _events(*rows: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _start(date: str) -> dict[str, str]:
    return _row("FORMAL_LISTING_SUSPENSION_START", date, "start")


def _close(event_type: str, date: str) -> dict[str, str]:
    return _row(event_type, date, "close")


def _row(event_type: str, date: str, suffix: str) -> dict[str, str]:
    return {
        "canonical_ts_code": "600001.SH",
        "event_type": event_type,
        "effective_start": date,
        "effective_end": date,
        "row_identity": f"row-{suffix}-{date}",
        "package_id": "package-fixture",
        "package_hash": "a" * 64,
        "status": "VERIFIED",
    }
