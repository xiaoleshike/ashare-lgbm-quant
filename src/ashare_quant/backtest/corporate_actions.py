"""Governed, fail-closed corporate-action execution policy."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_identity_transition import SecurityIdentityTransition

CORPORATE_ACTION_EXECUTION_POLICY_VERSION = "corporate_action_execution_policy_v1"
CORPORATE_ACTION_MARK_VALUE_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class CorporateActionExecutionEligibility:
    """Deterministic policy result for one validated identity transition."""

    supported: bool
    reason: str


@dataclass(frozen=True, slots=True)
class CorporateActionExecutionPolicy:
    """Shares-only execution semantics supported by accounting schema v3."""

    version: str = CORPORATE_ACTION_EXECUTION_POLICY_VERSION

    @property
    def policy_hash(self) -> str:
        return _payload_hash(self.payload())

    def payload(self) -> dict[str, object]:
        """Return the complete immutable semantic policy payload."""

        return {
            "version": self.version,
            "supported_semantics": ["SAME_LISTED_ENTITY_SHARE_CONVERSION"],
            "supported_transition_types": ["CODE_CHANGE_CONTINUITY"],
            "supported_continuity_types": ["SAME_LISTED_ENTITY"],
            "consideration": "shares_only_no_cash",
            "ratio": "authoritative_finite_positive_required",
            "topology": "one_predecessor_one_successor_unambiguous_chain",
            "fractional_shares": "preserve_exact_float_no_lot_rounding",
            "event_order": [
                "apply_effective_corporate_actions",
                "evaluate_scheduled_exits",
                "evaluate_new_entries",
                "end_of_day_valuation",
                "accounting_validation",
            ],
            "transaction_semantics": {
                "cash_delta": 0.0,
                "transaction_cost": 0.0,
                "turnover_contribution": 0.0,
                "synthetic_trade": False,
            },
            "mark_value_tolerance": CORPORATE_ACTION_MARK_VALUE_TOLERANCE,
            "unsupported_behavior": "CORPORATE_ACTION_EXECUTION_UNSUPPORTED",
        }

    def to_dict(self) -> dict[str, object]:
        """Return policy payload with its content-defined identity."""

        return {**self.payload(), "policy_hash": self.policy_hash}

    def classify(
        self, transition: SecurityIdentityTransition
    ) -> CorporateActionExecutionEligibility:
        """Classify engine support independently from identity evidence validity.

        ``CODE_CHANGE_CONTINUITY`` is the only v1 type whose contract guarantees a
        shares-only continuation with no cash or other consideration. Restructuring,
        merger, and successor-entity transitions remain unsupported even when a ratio
        happens to be present.
        """

        if transition.transition_type != "CODE_CHANGE_CONTINUITY":
            return CorporateActionExecutionEligibility(False, "UNSUPPORTED_TRANSITION_TYPE")
        if transition.continuity_type != "SAME_LISTED_ENTITY":
            return CorporateActionExecutionEligibility(False, "UNSUPPORTED_CONTINUITY_TYPE")
        ratio = transition.share_conversion_ratio
        if ratio is None:
            return CorporateActionExecutionEligibility(False, "EXECUTION_EVIDENCE_INSUFFICIENT")
        if isinstance(ratio, bool) or not math.isfinite(ratio) or ratio <= 0:
            return CorporateActionExecutionEligibility(False, "INVALID_SHARE_CONVERSION_RATIO")
        return CorporateActionExecutionEligibility(True, "SUPPORTED_SHARES_ONLY_CONTINUITY")


def default_corporate_action_execution_policy() -> CorporateActionExecutionPolicy:
    """Return the sole reviewed policy supported by accounting schema v3."""

    return CorporateActionExecutionPolicy()


def validate_corporate_action_execution_policy(
    payload: dict[str, Any], policy_hash: object
) -> None:
    """Require the sole reviewed policy supported by the current engine."""

    expected = default_corporate_action_execution_policy().to_dict()
    if payload != expected or policy_hash != expected["policy_hash"]:
        raise DataValidationError("CORPORATE_ACTION_EXECUTION_POLICY_UNSUPPORTED")


def _payload_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
