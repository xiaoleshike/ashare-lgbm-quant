from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_identity import SecurityIdentityResolver
from ashare_quant.data.security_identity_transition import (
    SecurityIdentityTransition,
    SecurityIdentityTransitionResolver,
)
from ashare_quant.data.security_lifecycle import SecurityLifecycleResolver
from ashare_quant.data.security_lifecycle_audit import (
    CLASSIFICATION_CONTRACT_VERSION,
    LifecycleAuditPolicy,
    SecurityLifecycleScanner,
    normalize_ordinary_suspension_intervals,
    validate_pass_lifecycle_scan,
    validate_security_lifecycle_artifact,
)

DATES = ("20200102", "20200103", "20200106", "20200107")


def test_explicit_daily_s_snapshots_compress_without_persistent_state() -> None:
    events = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20200103", "suspend_type": "S"},
            {"ts_code": "000001.SZ", "trade_date": "20200106", "suspend_type": "S"},
            {"ts_code": "000001.SZ", "trade_date": "20200107", "suspend_type": "R"},
        ]
    )

    intervals, boundaries = normalize_ordinary_suspension_intervals(
        events, DATES, scan_start=DATES[0], scan_end=DATES[-1]
    )

    assert intervals[["effective_start", "effective_end"]].to_dict("records") == [
        {"effective_start": "20200103", "effective_end": "20200106"}
    ]
    assert boundaries["boundary_type"].tolist() == ["RESUME_DAY_CONSISTENCY"]
    assert boundaries["status"].tolist() == ["INFO"]
    assert not boundaries["blocking"].any()


def test_r_without_s_is_nonblocking_and_s_does_not_carry_forward() -> None:
    events = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20200103", "suspend_type": "R"},
            {"ts_code": "000002.SZ", "trade_date": "20200103", "suspend_type": "S"},
        ]
    )

    intervals, boundaries = normalize_ordinary_suspension_intervals(
        events, DATES, scan_start=DATES[0], scan_end=DATES[-1]
    )

    assert intervals[["effective_start", "effective_end"]].to_dict("records") == [
        {"effective_start": "20200103", "effective_end": "20200103"}
    ]
    assert boundaries["boundary_type"].tolist() == ["RESUME_DAY_CONSISTENCY"]
    assert boundaries["status"].tolist() == ["WARNING"]
    assert not boundaries["blocking"].any()


def test_future_r_or_terminal_does_not_change_prior_explicit_s_dates() -> None:
    events = pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20200103", "suspend_type": "S"}])

    intervals, boundaries = normalize_ordinary_suspension_intervals(
        events, DATES, scan_start=DATES[0], scan_end=DATES[-1]
    )

    assert intervals.loc[0, "effective_end"] == "20200103"
    assert intervals.loc[0, "end_source"] == "LAST_CONSECUTIVE_SUSPEND_D_S_SNAPSHOT"
    assert boundaries.empty

    with_resume = pd.concat(
        [
            events,
            pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20200107", "suspend_type": "R"}]),
        ],
        ignore_index=True,
    )
    resumed, _ = normalize_ordinary_suspension_intervals(
        with_resume, DATES, scan_start=DATES[0], scan_end=DATES[-1]
    )
    pd.testing.assert_frame_equal(intervals, resumed)


def test_intraday_s_r_is_not_full_session_suspension() -> None:
    events = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20200103",
                "suspend_type": "S",
                "suspend_timing": "09:30-09:40",
            },
            {
                "ts_code": "000001.SZ",
                "trade_date": "20200103",
                "suspend_type": "R",
                "suspend_timing": None,
            },
        ]
    )

    intervals, boundaries = normalize_ordinary_suspension_intervals(
        events,
        DATES,
        scan_start=DATES[0],
        scan_end=DATES[-1],
        valid_quote_keys=frozenset({("000001.SZ", "20200103")}),
    )

    assert intervals.empty
    assert boundaries["boundary_type"].tolist() == ["INTRADAY_SUSPEND_RESUME"]
    assert boundaries["status"].tolist() == ["INFO"]


def test_same_day_full_session_s_r_with_quote_is_nonblocking_ambiguity() -> None:
    events = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20200103", "suspend_type": "S"},
            {"ts_code": "000001.SZ", "trade_date": "20200103", "suspend_type": "R"},
        ]
    )

    intervals, boundaries = normalize_ordinary_suspension_intervals(
        events,
        DATES,
        scan_start=DATES[0],
        scan_end=DATES[-1],
        valid_quote_keys=frozenset({("000001.SZ", "20200103")}),
    )

    assert intervals.empty
    assert boundaries["boundary_type"].tolist() == ["SAME_DAY_SR_AMBIGUITY"]
    assert boundaries["status"].tolist() == ["WARNING"]
    assert not boundaries["blocking"].any()


def test_same_day_full_session_s_r_without_quote_is_blocking_ambiguity() -> None:
    events = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20200103", "suspend_type": "S"},
            {"ts_code": "000001.SZ", "trade_date": "20200103", "suspend_type": "R"},
        ]
    )

    intervals, boundaries = normalize_ordinary_suspension_intervals(
        events, DATES, scan_start=DATES[0], scan_end=DATES[-1]
    )

    assert intervals.empty
    assert boundaries["status"].tolist() == ["BOUNDARY_INCONSISTENCY"]
    assert boundaries["blocking"].all()


def test_same_day_s_r_with_quote_and_universe_suspended_remains_nonblocking(
    tmp_path: Path,
) -> None:
    scanner = _fixture_scanner(tmp_path, include_unresolved=False)
    suspend_path = scanner.raw_root / "suspend_d" / "data.parquet"
    suspend = pd.read_parquet(suspend_path)
    suspend = pd.concat(
        [
            suspend,
            pd.DataFrame(
                [
                    {
                        "ts_code": "000005.SZ",
                        "trade_date": "20200103",
                        "suspend_type": "S",
                        "suspend_timing": None,
                    },
                    {
                        "ts_code": "000005.SZ",
                        "trade_date": "20200103",
                        "suspend_type": "R",
                        "suspend_timing": None,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    suspend.to_parquet(suspend_path, index=False)
    universe_path = scanner.processed_root / "universe_daily" / "data.parquet"
    universe = pd.read_parquet(universe_path)
    universe.loc[
        universe["ts_code"].eq("000005.SZ") & universe["trade_date"].eq("20200103"),
        "is_suspended",
    ] = True
    universe.to_parquet(universe_path, index=False)

    result = scanner.scan(DATES[0], DATES[-1])
    boundaries = pd.read_parquet(result.output_dir / "boundary_checks.parquet")
    selected = boundaries[
        boundaries["canonical_ts_code"].eq("000005.SZ") & boundaries["trade_date"].eq("20200103")
    ]

    assert set(selected["boundary_type"]) == {
        "RESUME_DAY_CONSISTENCY",
        "SAME_DAY_SR_AMBIGUITY",
    }
    assert not selected["blocking"].any()


def test_duplicate_identical_suspend_rows_are_deduplicated() -> None:
    duplicate = {"ts_code": "000001.SZ", "trade_date": "20200103", "suspend_type": "S"}

    intervals, boundaries = normalize_ordinary_suspension_intervals(
        pd.DataFrame([duplicate, duplicate]),
        DATES,
        scan_start=DATES[0],
        scan_end=DATES[-1],
    )

    assert len(intervals) == 1
    assert intervals.loc[0, "effective_start"] == intervals.loc[0, "effective_end"]
    assert boundaries.empty


def test_full_scan_classifies_all_primary_states_and_remains_blocked(tmp_path: Path) -> None:
    scanner = _fixture_scanner(tmp_path, include_unresolved=True)

    result = scanner.scan(DATES[0], DATES[-1])
    manifest = validate_security_lifecycle_artifact(result.output_dir)
    classified = pd.read_parquet(result.output_dir / "classified_gaps.parquet")

    assert result.status == "BLOCKED"
    assert manifest["status"] == "BLOCKED"
    assert set(classified["classification"]) >= {
        "ORDINARY_SUSPENSION",
        "LISTING_SUSPENSION",
        "TERMINAL_DELISTING",
        "UNRESOLVED",
    }
    assert result.counts["unresolved"] == 1
    assert result.counts["policy_collisions"] == 0
    assert result.counts["boundary_inconsistencies"] == 0
    with pytest.raises(DataValidationError, match="AUDIT_REQUIRED"):
        validate_pass_lifecycle_scan(
            result.output_dir / "manifest.json",
            required_start=DATES[0],
            required_end=DATES[-1],
        )


def test_verified_transition_stops_old_code_expected_quote_population(tmp_path: Path) -> None:
    base = _fixture_scanner(tmp_path, include_unresolved=True)
    transition = SecurityIdentityTransition(
        predecessor_ts_code="000004.SZ",
        successor_ts_code="001004.SZ",
        predecessor_name="old",
        successor_name="new",
        transition_type="CODE_CHANGE_CONTINUITY",
        effective_date="20200106",
        continuity_type="SAME_LISTED_ENTITY",
        share_conversion_ratio=1.0,
        evidence_package_id="fixture-evidence",
        evidence_package_hash="a" * 64,
    )
    scanner = SecurityLifecycleScanner(
        raw_root=base.raw_root,
        processed_root=base.processed_root,
        reports_root=base.reports_root,
        identity_resolver=base.identity,
        lifecycle_evidence=base.evidence,
        lifecycle_policy=base.policy,
        identity_transitions=SecurityIdentityTransitionResolver(
            artifact_version="fixture-v1",
            artifact_hash="b" * 64,
            transitions=(transition,),
        ),
    )

    result = scanner.scan(DATES[0], DATES[-1])
    classified = pd.read_parquet(result.output_dir / "classified_gaps.parquet")

    assert not classified["canonical_ts_code"].eq("000004.SZ").any()
    assert result.counts["unresolved"] == 0


def test_pass_scan_is_idempotent_portable_and_gate_valid(tmp_path: Path) -> None:
    first = _fixture_scanner(tmp_path / "checkout_a", include_unresolved=False)
    first_result = first.scan(DATES[0], DATES[-1])
    second_result = first.scan(DATES[0], DATES[-1])
    second = _copy_fixture_scanner(tmp_path / "checkout_a", tmp_path / "checkout_b")
    relocated = second.scan(DATES[0], DATES[-1])

    assert first_result.status == "PASS"
    assert second_result.idempotent
    assert first_result.scan_id == relocated.scan_id
    manifest_path = first_result.output_dir / "manifest.json"
    validated = validate_pass_lifecycle_scan(
        manifest_path,
        required_start=DATES[0],
        required_end=DATES[-1],
        raw_root=first.raw_root,
        processed_root=first.processed_root,
        identity_resolver=first.identity,
        lifecycle_evidence=first.evidence,
        lifecycle_policy=first.policy,
        identity_transitions=first.identity_transitions,
    )
    assert validated["scan_id"] == first_result.scan_id


def test_supersession_is_append_only_and_publishes_comparison(tmp_path: Path) -> None:
    first = _fixture_scanner(tmp_path, include_unresolved=False)
    first_result = first.scan(DATES[0], DATES[-1])
    policy_payload = _policy()
    policy_payload["policy_version"] = "fixture-v2"
    policy_path = tmp_path / "policy-v2.json"
    _write_json(policy_path, policy_payload)
    second = SecurityLifecycleScanner(
        raw_root=first.raw_root,
        processed_root=first.processed_root,
        reports_root=first.reports_root,
        identity_resolver=first.identity,
        lifecycle_evidence=first.evidence,
        lifecycle_policy=LifecycleAuditPolicy.from_path(policy_path),
        identity_transitions=SecurityIdentityTransitionResolver.empty(),
        supersedes_scan_manifest=first_result.output_dir / "manifest.json",
        supersession_reason="SUSPEND_D_SEMANTICS_RECALIBRATION",
    )

    second_result = second.scan(DATES[0], DATES[-1])
    manifest = validate_security_lifecycle_artifact(second_result.output_dir)
    comparison = json.loads(
        (second_result.output_dir / "supersession_comparison.json").read_text(encoding="utf-8")
    )

    assert first_result.output_dir.exists()
    assert second_result.scan_id != first_result.scan_id
    assert manifest["supersedes_scan_id"] == first_result.scan_id
    assert comparison["old_scan_id"] == first_result.scan_id
    assert (second_result.output_dir / "old_vs_new_changes.parquet").is_file()


def test_scanner_terminal_boundary_matches_universe_contract(tmp_path: Path) -> None:
    result = _fixture_scanner(tmp_path, include_unresolved=False).scan(DATES[0], DATES[-1])
    classified = pd.read_parquet(result.output_dir / "classified_gaps.parquet")
    boundaries = pd.read_parquet(result.output_dir / "boundary_checks.parquet")

    terminal = classified[
        classified["canonical_ts_code"].eq("000003.SZ")
        & classified["classification"].eq("TERMINAL_DELISTING")
    ]
    assert terminal["trade_date"].tolist() == ["20200106"]
    assert not boundaries["boundary_type"].eq("TERMINAL_TRANSITION").any()


def test_whole_market_missing_date_is_blocking_raw_data_not_safe(tmp_path: Path) -> None:
    scanner = _fixture_scanner(tmp_path, include_unresolved=False, omit_daily_date="20200106")

    result = scanner.scan(DATES[0], DATES[-1])
    raw = pd.read_parquet(result.output_dir / "raw_data_gaps.parquet")
    classified = pd.read_parquet(result.output_dir / "classified_gaps.parquet")

    assert result.status == "BLOCKED"
    assert raw["trade_date"].tolist() == ["20200106"]
    assert raw["blocking"].all()
    assert "MISSING_RAW_DATA" in set(classified["classification"])


def test_isolated_missing_quote_is_unresolved_not_raw_data(tmp_path: Path) -> None:
    result = _fixture_scanner(tmp_path, include_unresolved=True).scan(DATES[0], DATES[-1])
    classified = pd.read_parquet(result.output_dir / "classified_gaps.parquet")
    selected = classified[classified["canonical_ts_code"] == "000004.SZ"]

    assert selected["classification"].tolist() == ["UNRESOLVED"]
    assert selected["blocking"].all()


def test_pre_listing_suspend_snapshot_does_not_require_universe_suspension(
    tmp_path: Path,
) -> None:
    scanner = _fixture_scanner(tmp_path, include_unresolved=False)
    stock_path = scanner.raw_root / "stock_basic" / "data.parquet"
    stocks = pd.read_parquet(stock_path)
    stocks.loc[stocks["ts_code"].eq("000005.SZ"), "list_date"] = "20200106"
    stocks.to_parquet(stock_path, index=False)
    suspend_path = scanner.raw_root / "suspend_d" / "data.parquet"
    suspend = pd.read_parquet(suspend_path)
    suspend = pd.concat(
        [
            suspend,
            pd.DataFrame(
                [
                    {
                        "ts_code": "000005.SZ",
                        "trade_date": "20200103",
                        "suspend_type": "S",
                        "suspend_timing": None,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    suspend.to_parquet(suspend_path, index=False)

    result = scanner.scan(DATES[0], DATES[-1])
    boundaries = pd.read_parquet(result.output_dir / "boundary_checks.parquet")
    selected = boundaries[
        boundaries["canonical_ts_code"].eq("000005.SZ")
        & boundaries["trade_date"].eq("20200103")
        & boundaries["boundary_type"].eq("ORDINARY_SUSPENSION_DAILY_STATE")
    ]

    assert selected.empty


def test_scan_artifact_tamper_and_insufficient_coverage_fail_closed(tmp_path: Path) -> None:
    result = _fixture_scanner(tmp_path, include_unresolved=False).scan(DATES[0], DATES[-1])
    manifest_path = result.output_dir / "manifest.json"

    with pytest.raises(DataValidationError, match="COVERAGE_INSUFFICIENT"):
        validate_pass_lifecycle_scan(
            manifest_path, required_start="20191231", required_end=DATES[-1]
        )
    summary = result.output_dir / "summary.json"
    summary.write_text("{}\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="ARTIFACT_HASH_MISMATCH"):
        validate_security_lifecycle_artifact(result.output_dir)


def test_pass_gate_rejects_current_source_drift(tmp_path: Path) -> None:
    scanner = _fixture_scanner(tmp_path, include_unresolved=False)
    result = scanner.scan(DATES[0], DATES[-1])
    daily_path = scanner.raw_root / "daily" / "data.parquet"
    daily = pd.read_parquet(daily_path)
    daily.loc[0, "close"] = 11.0
    daily.to_parquet(daily_path, index=False)

    with pytest.raises(DataValidationError, match="SOURCE_MISMATCH"):
        validate_pass_lifecycle_scan(
            result.output_dir / "manifest.json",
            required_start=DATES[0],
            required_end=DATES[-1],
            raw_root=scanner.raw_root,
            processed_root=scanner.processed_root,
        )


def test_root_status_cannot_override_blocked_child_evidence(tmp_path: Path) -> None:
    result = _fixture_scanner(tmp_path, include_unresolved=True).scan(DATES[0], DATES[-1])
    manifest_path = result.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "PASS"
    _write_json(manifest_path, manifest)

    with pytest.raises(DataValidationError, match="STATUS_MISMATCH"):
        validate_security_lifecycle_artifact(result.output_dir)


@pytest.mark.parametrize("mutation", ["missing_counts", "negative_count"])
def test_scan_gate_rejects_invalid_mandatory_counts(tmp_path: Path, mutation: str) -> None:
    result = _fixture_scanner(tmp_path, include_unresolved=False).scan(DATES[0], DATES[-1])
    manifest_path = result.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "missing_counts":
        manifest.pop("counts")
    else:
        manifest["counts"]["unresolved"] = -1
    _write_json(manifest_path, manifest)

    with pytest.raises(DataValidationError, match="COUNTS_INVALID"):
        validate_security_lifecycle_artifact(result.output_dir)


def test_pass_scan_requires_explicit_no_transition_contract(tmp_path: Path) -> None:
    scanner = _fixture_scanner(tmp_path, include_unresolved=False)
    result = scanner.scan(DATES[0], DATES[-1])

    with pytest.raises(DataValidationError, match="TRANSITION_CONTRACT_REQUIRED"):
        validate_pass_lifecycle_scan(
            result.output_dir / "manifest.json",
            required_start=DATES[0],
            required_end=DATES[-1],
        )


def test_explicit_alias_is_resolved_and_unknown_code_is_not_guessed(tmp_path: Path) -> None:
    scanner = _fixture_scanner(tmp_path, include_unresolved=False)
    mapping_path = tmp_path / "mapping.json"
    payload = _mapping()
    payload["aliases"] = [
        {
            "source_code": "830001.BJ",
            "canonical_code": "920001.BJ",
            "exchange": "BJ",
            "effective_from": None,
            "effective_to": None,
            "mapping_source": "fixture official mapping",
        }
    ]
    _write_json(mapping_path, payload)
    identity = SecurityIdentityResolver.from_path(mapping_path)

    assert identity.canonicalize("830001.BJ", as_of_date="20200102") == "920001.BJ"
    assert identity.canonicalize("830999.BJ", as_of_date="20200102") == "830999.BJ"
    assert scanner.identity.canonicalize("830001.BJ") == "830001.BJ"


def test_post_terminal_quote_is_boundary_inconsistency(tmp_path: Path) -> None:
    scanner = _fixture_scanner(tmp_path, include_unresolved=False)
    daily_path = tmp_path / "raw" / "daily" / "data.parquet"
    daily = pd.read_parquet(daily_path)
    daily = pd.concat(
        [
            daily,
            pd.DataFrame(
                [
                    {
                        "ts_code": "000003.SZ",
                        "trade_date": "20200107",
                        "open": 1.0,
                        "close": 1.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    daily.to_parquet(daily_path, index=False)

    result = scanner.scan(DATES[0], DATES[-1])
    boundaries = pd.read_parquet(result.output_dir / "boundary_checks.parquet")

    assert result.status == "BLOCKED"
    assert "POST_TERMINAL_QUOTE" in set(boundaries["boundary_type"])


def test_source_or_policy_change_changes_scan_identity(tmp_path: Path) -> None:
    first = _fixture_scanner(tmp_path / "first", include_unresolved=False)
    first_id = first.scan(DATES[0], DATES[-1]).scan_id
    policy_payload = _policy()
    policy_payload["policy_version"] = "fixture-v2"
    _write_json(tmp_path / "first" / "policy-v2.json", policy_payload)
    changed_policy = SecurityLifecycleScanner(
        raw_root=first.raw_root,
        processed_root=first.processed_root,
        reports_root=first.reports_root,
        identity_resolver=first.identity,
        lifecycle_evidence=first.evidence,
        lifecycle_policy=LifecycleAuditPolicy.from_path(tmp_path / "first" / "policy-v2.json"),
        identity_transitions=SecurityIdentityTransitionResolver.empty(),
    )

    assert changed_policy.scan(DATES[0], DATES[-1]).scan_id != first_id


def test_medium_event_set_compresses_by_interval_not_stock_day_loop() -> None:
    rows = []
    for index in range(200):
        code = f"{index:06d}.SZ"
        rows.extend(
            [
                {"ts_code": code, "trade_date": "20200103", "suspend_type": "S"},
                {"ts_code": code, "trade_date": "20200106", "suspend_type": "S"},
                {"ts_code": code, "trade_date": "20200107", "suspend_type": "R"},
            ]
        )

    intervals, boundaries = normalize_ordinary_suspension_intervals(
        pd.DataFrame(rows), DATES, scan_start=DATES[0], scan_end=DATES[-1]
    )

    assert len(rows) == 600
    assert len(intervals) == 200
    assert len(boundaries) == 200
    assert not boundaries["blocking"].any()


def test_v3_verified_events_are_loaded_without_rewrite() -> None:
    path = Path("config/security_identity/security_lifecycle_events.json")
    before = path.read_bytes()
    evidence = SecurityLifecycleResolver.from_path(path)

    assert evidence.policy_version == "security_lifecycle_events_v3"
    assert evidence.event_count == 19
    assert path.read_bytes() == before


def _fixture_scanner(
    root: Path,
    *,
    include_unresolved: bool,
    omit_daily_date: str | None = None,
) -> SecurityLifecycleScanner:
    raw = root / "raw"
    processed = root / "processed"
    reports = root / "reports"
    policy_path = root / "policy.json"
    evidence_path = root / "evidence.json"
    mapping_path = root / "mapping.json"
    _write_json(policy_path, _policy())
    _write_json(evidence_path, _evidence())
    _write_json(mapping_path, _mapping())
    stocks = [
        ("000001.SZ", None),
        ("000002.SZ", None),
        ("000003.SZ", "20200106"),
        ("000005.SZ", None),
    ]
    if include_unresolved:
        stocks.append(("000004.SZ", None))
    _write_parquet(
        raw / "trade_cal" / "data.parquet",
        pd.DataFrame(
            {
                "exchange": ["SSE"] * len(DATES),
                "cal_date": DATES,
                "is_open": [1] * len(DATES),
                "pretrade_date": [""] * len(DATES),
            }
        ),
    )
    _write_parquet(
        raw / "stock_basic" / "data.parquet",
        pd.DataFrame(
            [
                {"ts_code": code, "list_date": DATES[0], "delist_date": delist}
                for code, delist in stocks
            ]
        ),
    )
    daily_rows: list[dict[str, object]] = []
    quote_dates = {
        "000001.SZ": ("20200102", "20200107"),
        "000002.SZ": ("20200102", "20200107"),
        "000003.SZ": ("20200102", "20200103"),
        "000004.SZ": ("20200102", "20200103", "20200107"),
        "000005.SZ": DATES,
    }
    for code, dates in quote_dates.items():
        if code not in {item[0] for item in stocks}:
            continue
        for date in dates:
            if date != omit_daily_date:
                daily_rows.append(
                    {"ts_code": code, "trade_date": date, "open": 10.0, "close": 10.0}
                )
    _write_parquet(raw / "daily" / "data.parquet", pd.DataFrame(daily_rows))
    _write_parquet(
        raw / "suspend_d" / "data.parquet",
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20200103",
                    "suspend_type": "S",
                    "suspend_timing": None,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20200106",
                    "suspend_type": "S",
                    "suspend_timing": None,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20200107",
                    "suspend_type": "R",
                    "suspend_timing": None,
                },
            ]
        ),
    )
    universe_rows = []
    for code, delist in stocks:
        for date in DATES:
            listed = delist is None or date < delist
            suspended = code == "000001.SZ" and date in {"20200103", "20200106"}
            suspended |= code == "000002.SZ" and date in {"20200103", "20200106"}
            universe_rows.append(
                {
                    "ts_code": code,
                    "trade_date": date,
                    "is_listed": listed,
                    "is_suspended": suspended,
                }
            )
    _write_parquet(processed / "universe_daily" / "data.parquet", pd.DataFrame(universe_rows))
    return SecurityLifecycleScanner(
        raw_root=raw,
        processed_root=processed,
        reports_root=reports,
        identity_resolver=SecurityIdentityResolver.from_path(mapping_path),
        lifecycle_evidence=SecurityLifecycleResolver.from_path(evidence_path),
        lifecycle_policy=LifecycleAuditPolicy.from_path(policy_path),
        identity_transitions=SecurityIdentityTransitionResolver.empty(),
    )


def _copy_fixture_scanner(source: Path, target: Path) -> SecurityLifecycleScanner:
    shutil.copytree(source / "raw", target / "raw")
    shutil.copytree(source / "processed", target / "processed")
    for name in ("policy.json", "evidence.json", "mapping.json"):
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target / name)
    return SecurityLifecycleScanner(
        raw_root=target / "raw",
        processed_root=target / "processed",
        reports_root=target / "reports",
        identity_resolver=SecurityIdentityResolver.from_path(target / "mapping.json"),
        lifecycle_evidence=SecurityLifecycleResolver.from_path(target / "evidence.json"),
        lifecycle_policy=LifecycleAuditPolicy.from_path(target / "policy.json"),
        identity_transitions=SecurityIdentityTransitionResolver.empty(),
    )


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _policy() -> dict[str, object]:
    return {
        "schema_version": 2,
        "artifact_name": "security_lifecycle_policy",
        "policy_version": "fixture-v1",
        "classification_contract_version": CLASSIFICATION_CONTRACT_VERSION,
        "classification_precedence": [
            "LISTING_SUSPENSION",
            "ORDINARY_SUSPENSION",
            "TERMINAL_DELISTING",
            "SECURITY_ALIAS",
            "MISSING_RAW_DATA",
            "UNRESOLVED",
        ],
    }


def _evidence() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_name": "security_lifecycle_events",
        "policy_version": "fixture-evidence-v1",
        "events": [
            {
                "canonical_ts_code": "000002.SZ",
                "event_type": "LISTING_SUSPENDED",
                "effective_from": "20200103",
                "effective_to": "20200106",
                "evidence_source": "fixture exchange decision",
                "evidence_reference": "fixture://listing-suspension",
            }
        ],
    }


def _mapping() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_name": "security_identity_mapping",
        "mapping_version": "fixture-mapping-v1",
        "aliases": [],
    }
