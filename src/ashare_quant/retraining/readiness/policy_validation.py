"""Promotion-policy drift checks for retraining execution."""

from __future__ import annotations

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.gate_rules import PromotionGatePolicy
from ashare_quant.retraining.readiness.schemas import GovernanceContext
from ashare_quant.retraining.schemas import TrainingRequest


class PromotionPolicyDriftError(DataValidationError):
    """The current promotion policy differs from frozen governance lineage."""


def validate_promotion_policy(
    current: PromotionGatePolicy,
    governance: GovernanceContext,
    request: TrainingRequest | None = None,
) -> None:
    """Require current, governance, and request promotion-policy identities to match."""

    if (
        governance.promotion_policy_hash != current.policy_hash
        or governance.promotion_policy_version != current.policy_version
    ):
        raise PromotionPolicyDriftError("promotion policy differs from governance snapshot")
    if governance.previous_promotion_policy_hash is not None and (
        governance.previous_promotion_policy_hash != current.policy_hash
        or governance.previous_promotion_policy_version != current.policy_version
    ):
        raise PromotionPolicyDriftError("promotion policy differs from prior Champion evidence")
    if request is not None and (
        request.promotion_policy_hash != current.policy_hash
        or request.promotion_policy_version != current.policy_version
    ):
        raise PromotionPolicyDriftError("promotion policy differs from retraining request")
