from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_lifecycle_audit import file_sha256
from ashare_quant.data.security_lifecycle_triage import (
    REQUIRED_ARTIFACTS,
    _duration_bucket,
    _triage_category,
    validate_security_lifecycle_triage_artifact,
)


@pytest.mark.parametrize(
    ("sessions", "expected"),
    [(1, "1"), (2, "2-5"), (6, "6-20"), (21, "21-60"), (61, "61-120"), (121, ">120")],
)
def test_duration_buckets_are_deterministic(sessions: int, expected: str) -> None:
    assert _duration_bucket(sessions) == expected


def test_triage_never_promotes_candidate_to_authoritative_classification() -> None:
    assert (
        _triage_category(
            session_count=200,
            same_day_blocking=False,
            overlaps_v3=False,
            aliases=[],
            delist_date=None,
            gap_end="20240131",
            next_quote="20240201",
            has_s=False,
            has_r=False,
        )
        == "LIKELY_FORMAL_LISTING_SUSPENSION"
    )
    assert (
        _triage_category(
            session_count=1,
            same_day_blocking=True,
            overlaps_v3=False,
            aliases=[],
            delist_date=None,
            gap_end="20240131",
            next_quote=None,
            has_s=True,
            has_r=True,
        )
        == "SAME_DAY_SR_REQUIRES_EVIDENCE"
    )


def test_triage_artifact_validation_is_root_to_leaf(tmp_path: Path) -> None:
    root = tmp_path / "security_lifecycle_triage_test"
    root.mkdir()
    for name in REQUIRED_ARTIFACTS:
        path = root / name
        if name.endswith(".parquet"):
            pd.DataFrame({"value": [1]}).to_parquet(path, index=False)
        else:
            path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_name": "security_lifecycle_triage",
        "triage_id": root.name,
        "counts": {},
        "artifact_hashes": {name: file_sha256(root / name) for name in REQUIRED_ARTIFACTS},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    validate_security_lifecycle_triage_artifact(root)
    (root / "summary.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(DataValidationError, match="HASH_MISMATCH"):
        validate_security_lifecycle_triage_artifact(root)
