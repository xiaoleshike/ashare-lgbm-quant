from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.backtest.engine import BacktestInputs, simulate_portfolio
from ashare_quant.config.settings import BacktestSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_identity import SecurityIdentityResolver
from ashare_quant.data.security_identity_transition import (
    SecurityIdentityTransition,
    SecurityIdentityTransitionResolver,
    SecuritySourceRepresentationRule,
    publish_security_identity_transitions,
    publish_source_reconciliation_artifact,
    publish_transition_evidence_package,
    validate_transition_evidence_package,
)


def test_simple_code_transition_ends_predecessor_quote_expectation(tmp_path: Path) -> None:
    resolver = _resolver((_transition("000001.SZ", "001001.SZ", "20240104", 1.0),))

    assert resolver.predecessor_expected("000001.SZ", trade_date="20240103") is True
    assert resolver.predecessor_expected("000001.SZ", trade_date="20240104") is False
    assert resolver.effective_code("000001.SZ", as_of_date="20240103") == "000001.SZ"
    assert resolver.effective_code("000001.SZ", as_of_date="20240104") == "001001.SZ"


def test_unknown_conversion_resolves_identity_but_blocks_execution() -> None:
    transition = _transition("000001.SZ", "001001.SZ", "20240104", None)
    inputs = BacktestInputs(
        signals=pd.DataFrame([{"trade_date": "20240102", "ts_code": "000001.SZ", "score": 1.0}]),
        prices=pd.DataFrame(
            [
                _price("20240102", "000001.SZ"),
                _price("20240103", "000001.SZ"),
                _price("20240104", "001001.SZ"),
                _price("20240105", "001001.SZ"),
            ]
        ),
        calendar=("20240102", "20240103", "20240104", "20240105"),
        benchmark=pd.DataFrame(
            {"trade_date": ["20240102", "20240103", "20240104", "20240105"], "close": [100.0] * 4}
        ),
        identity_transitions=(transition,),
        identity_transition_version="fixture-v1",
        identity_transition_hash="b" * 64,
    )

    with pytest.raises(DataValidationError, match="CORPORATE_ACTION_EXECUTION_UNSUPPORTED"):
        simulate_portfolio(
            inputs,
            top_n=1,
            settings=BacktestSettings.model_validate(
                {
                    "initial_cash": 1000.0,
                    "top_n": (1,),
                    "holding_period_days": 3,
                    "commission": 0.0,
                    "stamp_duty": 0.0,
                    "slippage": 0.0,
                    "sell_delay_max_days": 2,
                }
            ),
            purpose="executable_validation",
        )


def test_conflicting_successor_and_cycle_fail_closed() -> None:
    with pytest.raises(DataValidationError, match="multiple predecessors"):
        _resolver(
            (
                _transition("000001.SZ", "001001.SZ", "20240104", 1.0),
                _transition("000002.SZ", "001001.SZ", "20240105", 1.0),
            )
        )
    with pytest.raises(DataValidationError, match="cycle"):
        _resolver(
            (
                _transition("000001.SZ", "001001.SZ", "20240104", 1.0),
                _transition("001001.SZ", "000001.SZ", "20240105", 1.0),
            )
        )


def test_transition_graph_coexists_with_unrelated_bse_alias(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_name": "security_identity_mapping",
                "mapping_version": "fixture",
                "aliases": [
                    {
                        "source_code": "830001.BJ",
                        "canonical_code": "920001.BJ",
                        "exchange": "BJ",
                        "effective_from": None,
                        "effective_to": None,
                        "mapping_source": "fixture",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    resolver = _resolver((_transition("000001.SZ", "001001.SZ", "20240104", 1.0),))

    resolver.validate_alias_coexistence(SecurityIdentityResolver.from_path(mapping))


@pytest.mark.parametrize(
    ("effective_date", "ratio"),
    [
        ("20240230", 1.0),
        ("20240104", 0.0),
        ("20240104", -1.0),
        ("20240104", float("nan")),
        ("20240104", float("inf")),
        ("20240104", True),
    ],
)
def test_transition_rejects_invalid_dates_and_ratios(effective_date: str, ratio: float) -> None:
    with pytest.raises(DataValidationError, match="SECURITY_IDENTITY_TRANSITION_INVALID"):
        _resolver((_transition("000001.SZ", "001001.SZ", effective_date, ratio),))


def test_transition_chain_requires_monotonic_effective_dates() -> None:
    with pytest.raises(DataValidationError, match="out of order"):
        _resolver(
            (
                _transition("000001.SZ", "001001.SZ", "20240105", 1.0),
                _transition("001001.SZ", "002001.SZ", "20240104", 1.0),
            )
        )


def test_execution_code_closure_includes_effective_successor_chain() -> None:
    resolver = _resolver(
        (
            _transition("000001.SZ", "001001.SZ", "20240103", 1.0),
            _transition("001001.SZ", "002001.SZ", "20240105", 1.0),
        )
    )

    assert resolver.execution_code_closure({"000001.SZ"}, end_date="20240104") == (
        "000001.SZ",
        "001001.SZ",
    )
    assert resolver.execution_code_closure({"000001.SZ"}, end_date="20240105") == (
        "000001.SZ",
        "001001.SZ",
        "002001.SZ",
    )


def test_dataset_scoped_source_representation_preserves_exchange_identity() -> None:
    transition = _transition("000001.SZ", "001001.SZ", "20240104", 1.0)
    resolver = _resolver(
        (transition,),
        rules=(_source_rule("001001.SZ", "000001.SZ", "20200101", "20240103"),),
    )
    source = pd.DataFrame(
        [
            {
                "ts_code": "001001.SZ",
                "trade_date": "20230103",
                "suspend_type": "S",
                "suspend_timing": None,
            }
        ]
    )

    normalized = resolver.normalize_source_frame(source, dataset_name="suspend_d")

    assert normalized.loc[0, "ts_code"] == "000001.SZ"
    assert normalized.loc[0, "source_ts_code"] == "001001.SZ"
    assert normalized.loc[0, "source_ts_codes"] == "001001.SZ"
    assert resolver.effective_code("000001.SZ", as_of_date="20230103") == "000001.SZ"
    assert resolver.normalize_source_frame(source, dataset_name="daily").equals(source)


def test_identical_dual_code_source_rows_merge_but_conflicts_fail_closed() -> None:
    resolver = _resolver(
        (_transition("000001.SZ", "001001.SZ", "20240104", 1.0),),
        rules=(_source_rule("001001.SZ", "000001.SZ", "20200101", "20240103"),),
    )
    identical = pd.DataFrame(
        [
            {
                "ts_code": code,
                "trade_date": "20230103",
                "suspend_type": "S",
                "suspend_timing": None,
            }
            for code in ("000001.SZ", "001001.SZ")
        ]
    )

    normalized = resolver.normalize_source_frame(identical, dataset_name="suspend_d")

    assert len(normalized) == 1
    assert normalized.loc[0, "source_ts_codes"] == "000001.SZ,001001.SZ"
    conflicting = identical.copy()
    conflicting.loc[1, "suspend_timing"] = "09:30-10:00"
    with pytest.raises(DataValidationError, match="SOURCE_REPRESENTATION_CONFLICT"):
        resolver.normalize_source_frame(conflicting, dataset_name="suspend_d")


def test_source_representation_requires_verified_transition() -> None:
    with pytest.raises(DataValidationError, match="IDENTITY_UNVERIFIED"):
        _resolver(
            (),
            rules=(_source_rule("001001.SZ", "000001.SZ", "20200101", "20240103"),),
        )


def test_canonical_payload_rejects_non_finite_json() -> None:
    from ashare_quant.data.security_identity_transition import canonical_payload_hash

    with pytest.raises(DataValidationError, match="CANONICAL_JSON_INVALID"):
        canonical_payload_hash({"ratio": float("nan")})


def test_official_transition_package_and_catalog_are_hash_validated(tmp_path: Path) -> None:
    document = tmp_path / "official.pdf"
    document.write_bytes(b"official fixture")
    evidence = _evidence()
    package = publish_transition_evidence_package(
        evidence=evidence, document=document, reports_root=tmp_path / "reports"
    )
    catalog = publish_security_identity_transitions(
        evidence_packages=(package,),
        reports_root=tmp_path / "reports",
        transition_version="security_identity_transitions_v1",
    )

    resolver = SecurityIdentityTransitionResolver.from_path(catalog)
    assert resolver.transition_count == 1
    assert resolver.transition_for("000001.SZ", "20240104") is not None

    frozen = package / "documents" / "official_document.pdf"
    frozen.write_bytes(b"tampered")
    with pytest.raises(DataValidationError, match="HASH_MISMATCH"):
        validate_transition_evidence_package(package)
    with pytest.raises(DataValidationError, match="HASH_MISMATCH"):
        SecurityIdentityTransitionResolver.from_path(catalog)


def test_source_reconciliation_is_recursively_bound_to_transition_artifact(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    document = tmp_path / "official.pdf"
    document.write_bytes(b"official fixture")
    evidence_package = publish_transition_evidence_package(
        evidence=_evidence(), document=document, reports_root=reports
    )
    rows = pd.DataFrame(
        [
            {
                "parent_interval_id": "interval-1",
                "trade_date": "20230103",
                "source_dataset": "suspend_d",
                "source_ts_code": "001001.SZ",
                "resolved_identity": "000001.SZ",
                "effective_trade_code": "000001.SZ",
                "resolution_rule_id": "fixture-rule",
                "row_present": True,
                "valid_quote": False,
                "suspend_type": "S",
                "raw_suspend_timing": None,
                "source_partition": "year=2023/month=01",
                "source_content_hash": "c" * 64,
                "identity_evidence_hash": "a" * 64,
            }
        ]
    )
    sessions = pd.DataFrame(
        [
            {
                "parent_interval_id": "interval-1",
                "trade_date": "20230103",
                "resolution": "RESOLVED_BY_VERIFIED_SOURCE_CODE_RECONCILIATION",
            }
        ]
    )
    rule = {
        "dataset_name": "suspend_d",
        "source_ts_code": "001001.SZ",
        "effective_ts_code": "000001.SZ",
        "effective_from": "20200101",
        "effective_to": "20240103",
        "resolution_rule_id": "fixture-rule",
    }
    reconciliation = publish_source_reconciliation_artifact(
        source_reconciliation=rows,
        session_resolution=sessions,
        source_representation_rules=(rule,),
        input_identity={"snapshot_hash": "d" * 64},
        reports_root=reports,
    )
    catalog = publish_security_identity_transitions(
        evidence_packages=(evidence_package,),
        source_representation_artifacts=(reconciliation,),
        reports_root=reports,
        transition_version="fixture-v2",
    )

    resolver = SecurityIdentityTransitionResolver.from_path(catalog)

    assert resolver.source_representation_rule_count == 1
    (reconciliation / "session_resolution.parquet").write_bytes(b"tampered")
    with pytest.raises(DataValidationError, match="HASH_MISMATCH"):
        SecurityIdentityTransitionResolver.from_path(catalog)


def _transition(
    predecessor: str, successor: str, effective: str, ratio: float | None
) -> SecurityIdentityTransition:
    return SecurityIdentityTransition(
        predecessor_ts_code=predecessor,
        successor_ts_code=successor,
        predecessor_name="old",
        successor_name="new",
        transition_type="RESTRUCTURING_CODE_CHANGE",
        effective_date=effective,
        continuity_type="SAME_LISTED_ENTITY",
        share_conversion_ratio=ratio,
        evidence_package_id="evidence-fixture",
        evidence_package_hash="a" * 64,
    )


def _resolver(
    transitions: tuple[SecurityIdentityTransition, ...],
    *,
    rules: tuple[SecuritySourceRepresentationRule, ...] = (),
) -> SecurityIdentityTransitionResolver:
    return SecurityIdentityTransitionResolver(
        artifact_version="fixture-v1",
        artifact_hash="b" * 64,
        transitions=transitions,
        source_representation_rules=rules,
    )


def _source_rule(
    source: str, effective: str, start: str, end: str
) -> SecuritySourceRepresentationRule:
    return SecuritySourceRepresentationRule(
        dataset_name="suspend_d",
        source_ts_code=source,
        effective_ts_code=effective,
        effective_from=start,
        effective_to=end,
        resolution_rule_id="fixture-rule",
        evidence_package_id="source-reconciliation-fixture",
        evidence_package_hash="c" * 64,
    )


def _price(date: str, code: str) -> dict[str, object]:
    return {
        "trade_date": date,
        "ts_code": code,
        "open": 10.0,
        "close": 10.0,
        "can_buy": True,
        "can_sell": True,
        "is_suspended": False,
        "is_listed": True,
        "delist_date": None,
    }


def _evidence() -> dict[str, object]:
    return {
        "predecessor_ts_code": "000001.SZ",
        "successor_ts_code": "001001.SZ",
        "predecessor_name": "old",
        "successor_name": "new",
        "transition_type": "RESTRUCTURING_CODE_CHANGE",
        "effective_date": "20240104",
        "continuity_type": "SAME_LISTED_ENTITY",
        "share_conversion_ratio": 1.0,
        "official_source_type": "SZSE_OFFICIAL_DISCLOSURE",
        "official_url": "https://disc.static.szse.cn/official.pdf",
        "official_document_id": "fixture-document",
        "publication_date": "20240103",
        "reviewed_fact": "code changes at the effective date and holdings are unchanged",
        "status": "VERIFIED",
        "retrieved_at": "2026-09-22T00:00:00Z",
    }
