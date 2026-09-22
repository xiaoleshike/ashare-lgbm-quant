from __future__ import annotations

import pandas as pd

from ashare_quant.data.security_identity_transition import (
    SecurityIdentityTransition,
    SecurityIdentityTransitionResolver,
)
from ashare_quant.data.security_lifecycle_transition_closure import resolve_transition_queue


def test_transition_closure_splits_parent_and_reconciles_every_session() -> None:
    queue = pd.DataFrame(
        [
            {
                "input_segment_id": "segment-a",
                "parent_interval_id": "parent-a",
                "canonical_ts_code": "000001.SZ",
                "exchange": "SZSE",
                "source_resolution": "PROVIDER_HAS_NO_SUSPEND_EVIDENCE",
                "repair_required": False,
                "segment_start": "20240102",
                "segment_end": "20240111",
                "session_count": 8,
            }
        ]
    )
    transition = SecurityIdentityTransition(
        predecessor_ts_code="000001.SZ",
        successor_ts_code="001001.SZ",
        predecessor_name="old",
        successor_name="new",
        transition_type="RESTRUCTURING_CODE_CHANGE",
        effective_date="20240105",
        continuity_type="SAME_LISTED_ENTITY",
        share_conversion_ratio=1.0,
        evidence_package_id="evidence-a",
        evidence_package_hash="a" * 64,
    )
    resolver = SecurityIdentityTransitionResolver(
        artifact_version="fixture-v1", artifact_hash="b" * 64, transitions=(transition,)
    )
    calendar = (
        "20240102",
        "20240103",
        "20240104",
        "20240105",
        "20240108",
        "20240109",
        "20240110",
        "20240111",
    )

    result = resolve_transition_queue(queue, calendar, resolver)

    assert len(result) == 2
    assert result["session_count"].sum() == 8
    unresolved = result[result["lifecycle_resolution"].eq("STILL_UNRESOLVED")].iloc[0]
    resolved = result[result["lifecycle_resolution"].eq("VERIFIED_SECURITY_CODE_TRANSITION")].iloc[
        0
    ]
    assert (
        unresolved["segment_start"],
        unresolved["segment_end"],
        unresolved["session_count"],
    ) == (
        "20240102",
        "20240104",
        3,
    )
    assert (resolved["segment_start"], resolved["segment_end"], resolved["session_count"]) == (
        "20240105",
        "20240111",
        5,
    )
    assert resolved["final_blocking"] == False  # noqa: E712
    assert resolved["execution_supported"] == False  # noqa: E712
