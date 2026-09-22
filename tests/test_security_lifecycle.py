from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_lifecycle import SecurityLifecycleResolver


def test_governed_lifecycle_marks_only_explicit_effective_interval() -> None:
    resolver = SecurityLifecycleResolver.from_path(
        Path("config/security_identity/security_lifecycle_events.json")
    )

    assert not resolver.is_listing_suspended("300028.SZ", "20190510")
    assert resolver.is_listing_suspended("300028.SZ", "20190513")
    assert resolver.is_listing_suspended("300028.SZ", "20200617")
    assert not resolver.is_listing_suspended("300028.SZ", "20200618")
    assert not resolver.is_listing_suspended("000001.SZ", "20190513")
    assert resolver.policy_version == "security_lifecycle_events_v3"
    assert len(resolver.policy_hash) == 64


def test_governed_lifecycle_covers_authoritative_2019_listing_suspensions() -> None:
    resolver = SecurityLifecycleResolver.from_path(
        Path("config/security_identity/security_lifecycle_events.json")
    )

    assert not resolver.is_listing_suspended("300216.SZ", "20190510")
    assert resolver.is_listing_suspended("300216.SZ", "20190513")
    assert resolver.is_listing_suspended("300216.SZ", "20200804")
    assert not resolver.is_listing_suspended("300216.SZ", "20200805")
    assert resolver.is_listing_suspended("300104.SZ", "20190513")
    assert resolver.is_listing_suspended("000939.SZ", "20190513")
    assert resolver.is_listing_suspended("002604.SZ", "20190515")
    assert resolver.is_listing_suspended("002260.SZ", "20190515")
    assert resolver.is_listing_suspended("600074.SH", "20190524")
    assert not resolver.is_listing_suspended("000001.SZ", "20190513")


@pytest.mark.parametrize(
    ("ts_code", "start", "end", "resume"),
    [
        ("300362.SZ", "20200513", "20210718", "20210719"),
        ("600485.SH", "20200515", "20210531", "20210601"),
        ("002711.SZ", "20200515", "20210601", "20210602"),
        ("600677.SH", "20200529", "20210317", "20210318"),
        ("600701.SH", "20200529", "20210314", "20210315"),
        ("000760.SZ", "20200706", "20210609", "20210610"),
        ("300431.SZ", "20200708", "20200920", "20200921"),
        ("002359.SZ", "20200709", "20210609", "20210610"),
        ("002450.SZ", "20200710", "20210413", "20210414"),
        ("600614.SH", "20200717", "20210601", "20210602"),
        ("600634.SH", "20201209", "20210601", "20210602"),
    ],
)
def test_governed_lifecycle_covers_authoritative_2020_listing_suspensions(
    ts_code: str, start: str, end: str, resume: str
) -> None:
    resolver = SecurityLifecycleResolver.from_path(
        Path("config/security_identity/security_lifecycle_events.json")
    )

    assert resolver.is_listing_suspended(ts_code, start)
    assert resolver.is_listing_suspended(ts_code, end)
    assert not resolver.is_listing_suspended(ts_code, resume)


def test_lifecycle_overlay_never_infers_suspension_from_missing_price() -> None:
    resolver = SecurityLifecycleResolver.from_path(
        Path("config/security_identity/security_lifecycle_events.json")
    )
    frame = pd.DataFrame(
        {
            "trade_date": ["20190513", "20190513"],
            "ts_code": ["300028.SZ", "000001.SZ"],
            "is_suspended": [False, False],
            "close": [float("nan"), float("nan")],
        }
    )

    resolved = resolver.apply_suspension(frame)

    assert resolved["is_suspended"].tolist() == [True, False]


def test_lifecycle_policy_rejects_overlapping_intervals(tmp_path: Path) -> None:
    path = tmp_path / "lifecycle.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_name": "security_lifecycle_events",
                "policy_version": "fixture-v1",
                "events": [
                    _event("20200101", "20200110"),
                    _event("20200105", "20200112"),
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="SECURITY_LIFECYCLE_COLLISION"):
        SecurityLifecycleResolver.from_path(path)


def test_typed_ordinary_and_listing_evidence_remain_distinct(tmp_path: Path) -> None:
    path = tmp_path / "typed-events.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "artifact_name": "security_lifecycle_events",
                "policy_version": "typed-fixture-v1",
                "completeness": "partial",
                "events": [
                    {
                        **_event("20240102", "20240102"),
                        "event_type": "ORDINARY_SUSPENSION",
                    },
                    {
                        **_event("20240102", "20240105"),
                        "canonical_ts_code": "000002.SZ",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    resolver = SecurityLifecycleResolver.from_path(path)

    assert resolver.is_ordinary_suspended("000001.SZ", "20240102")
    assert not resolver.is_listing_suspended("000001.SZ", "20240102")
    assert resolver.is_listing_suspended("000002.SZ", "20240103")
    ordinary = resolver.suspension_keys(["20240102", "20240103"], event_type="ORDINARY_SUSPENSION")
    listing = resolver.suspension_keys(["20240102", "20240103"], event_type="LISTING_SUSPENDED")
    assert set(ordinary["ts_code"]) == {"000001.SZ"}
    assert set(listing["ts_code"]) == {"000002.SZ"}


def _event(start: str, end: str) -> dict[str, str]:
    return {
        "canonical_ts_code": "000001.SZ",
        "event_type": "LISTING_SUSPENDED",
        "effective_from": start,
        "effective_to": end,
        "evidence_source": "fixture",
        "evidence_reference": "fixture://authoritative-event",
    }
