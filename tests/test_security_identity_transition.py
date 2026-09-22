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
    publish_security_identity_transitions,
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
) -> SecurityIdentityTransitionResolver:
    return SecurityIdentityTransitionResolver(
        artifact_version="fixture-v1", artifact_hash="b" * 64, transitions=transitions
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
