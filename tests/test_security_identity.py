from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_identity import (
    SecurityIdentityResolver,
    scan_cross_source_identity,
)


def resolver() -> SecurityIdentityResolver:
    return SecurityIdentityResolver.from_path(
        Path("config/security_identity/bse_code_aliases.json")
    )


def test_explicit_mapping_is_deterministic_and_does_not_guess() -> None:
    identity = resolver()

    assert identity.canonicalize("839680.BJ", as_of_date="20250430") == "920680.BJ"
    assert identity.canonicalize("835305.BJ", as_of_date="20250430") == "920305.BJ"
    assert identity.canonicalize("920680.BJ", as_of_date="20250430") == "920680.BJ"
    assert identity.canonicalize("600000.SH", as_of_date="20250430") == "600000.SH"
    assert identity.canonicalize("000001.SZ", as_of_date="20250430") == "000001.SZ"
    assert identity.canonicalize("830000.BJ", as_of_date="20250430") == "830000.BJ"
    assert len(identity.mapping_hash) == 64
    assert identity.mapping_version == "bse_code_aliases_v2"
    assert identity.alias_count == 248


def test_effective_dated_mapping_is_point_in_time(tmp_path: Path) -> None:
    path = tmp_path / "mapping.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_name": "security_identity_mapping",
                "mapping_version": "test-v1",
                "aliases": [
                    {
                        "source_code": "830001.BJ",
                        "canonical_code": "920001.BJ",
                        "exchange": "BJ",
                        "effective_from": "20250101",
                        "effective_to": None,
                        "mapping_source": "fixture",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    identity = SecurityIdentityResolver.from_path(path)

    assert identity.canonicalize("830001.BJ", as_of_date="20241231") == "830001.BJ"
    assert identity.canonicalize("830001.BJ", as_of_date="20250101") == "920001.BJ"


def test_canonicalization_preserves_source_lineage_and_raw_frame() -> None:
    identity = resolver()
    raw = pd.DataFrame(
        {
            "ts_code": ["839680.BJ"],
            "trade_date": ["20250430"],
            "suspend_type": ["S"],
        }
    )

    canonical = identity.canonicalize_frame(raw, "suspend_d")

    assert raw.loc[0, "ts_code"] == "839680.BJ"
    assert canonical.loc[0, "source_ts_code"] == "839680.BJ"
    assert canonical.loc[0, "ts_code"] == "920680.BJ"


@pytest.mark.parametrize(
    ("dataset_name", "date_column"),
    [
        ("stock_basic", None),
        ("daily", "trade_date"),
        ("stk_limit", "trade_date"),
        ("namechange", "start_date"),
        ("adj_factor", "trade_date"),
        ("income", "ann_date"),
    ],
)
def test_shared_resolver_covers_processed_security_join_sources(
    dataset_name: str, date_column: str | None
) -> None:
    row: dict[str, object] = {"ts_code": "839680.BJ"}
    if date_column is not None:
        row[date_column] = "20250430"

    canonical = resolver().canonicalize_frame(pd.DataFrame([row]), dataset_name)

    assert canonical.loc[0, "ts_code"] == "920680.BJ"
    assert canonical.loc[0, "source_ts_code"] == "839680.BJ"


def test_equal_alias_collision_deduplicates_but_conflict_fails_closed() -> None:
    identity = resolver()
    equal = pd.DataFrame(
        {
            "ts_code": ["839680.BJ", "920680.BJ"],
            "trade_date": ["20250430", "20250430"],
            "suspend_type": ["S", "S"],
        }
    )
    conflict = equal.copy()
    conflict.loc[1, "suspend_type"] = "R"

    merged = identity.canonicalize_frame(equal, "suspend_d")

    assert len(merged) == 1
    assert merged.loc[0, "ts_code"] == "920680.BJ"
    with pytest.raises(DataValidationError, match="SECURITY_IDENTITY_COLLISION"):
        identity.canonicalize_frame(conflict, "suspend_d")


def test_same_source_multiple_suspend_events_are_not_identity_collision() -> None:
    events = pd.DataFrame(
        {
            "ts_code": ["000004.SZ", "000004.SZ"],
            "trade_date": ["20260623", "20260623"],
            "suspend_timing": [None, "11:07-11:17,11:22-13:02"],
            "suspend_type": ["R", "S"],
        }
    )

    canonical = resolver().canonicalize_frame(events, "suspend_d")

    assert len(canonical) == 2
    assert set(canonical["suspend_type"]) == {"R", "S"}

    scan = scan_cross_source_identity({"suspend_d": events}, resolver())

    assert scan.same_source_multi_event_keys == 1


def test_namechange_prefers_canonical_source_revision_for_same_event() -> None:
    changes = pd.DataFrame(
        {
            "ts_code": ["839680.BJ", "920680.BJ"],
            "name": ["*ST Test", "*ST Test"],
            "start_date": ["20250506", "20250506"],
            "end_date": [None, "20251210"],
            "ann_date": ["20250430", "20250430"],
            "change_reason": ["*ST", "*ST"],
        }
    )

    canonical = resolver().canonicalize_frame(changes, "namechange")

    assert len(canonical) == 1
    assert canonical.loc[0, "source_ts_code"] == "920680.BJ"
    assert canonical.loc[0, "end_date"] == "20251210"


def test_cross_source_scan_resolves_explicit_alias_and_reports_missing_reference() -> None:
    identity = resolver()
    suspend = pd.DataFrame(
        {
            "ts_code": ["839680.BJ"],
            "trade_date": ["20250430"],
            "suspend_type": ["S"],
        }
    )
    daily = pd.DataFrame(
        {
            "ts_code": ["920680.BJ"],
            "trade_date": ["20250430"],
            "close": [pd.NA],
        }
    )

    clean = scan_cross_source_identity(
        {"daily": daily, "suspend_d": suspend, "stk_limit": pd.DataFrame()}, identity
    )
    missing = scan_cross_source_identity(
        {"daily": pd.DataFrame(), "suspend_d": suspend, "stk_limit": pd.DataFrame()},
        identity,
    )

    assert clean.ok
    assert clean.observed_aliases == 1
    assert missing.unresolved_mismatches == ("suspend_d:20250430:920680.BJ",)
