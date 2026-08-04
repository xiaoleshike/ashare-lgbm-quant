"""Read-only eligibility checks for future prospective shadow scoring."""

from __future__ import annotations

from collections.abc import Callable

from ashare_quant.models.inference import score_registered_model_range
from ashare_quant.retraining.validation.schemas import (
    CandidateValidationContext,
    ShadowEligibilityEvidence,
)


class RetrainingShadowEligibilityValidator:
    """Validate compatibility without producing any production prediction."""

    def __init__(self, *, adapter: Callable[..., object] = score_registered_model_range) -> None:
        self.adapter = adapter

    def evaluate(self, context: CandidateValidationContext) -> ShadowEligibilityEvidence:
        feature_ok = context.artifact.feature_hash == context.artifact.feature_list_hash
        universe_ok = context.artifact.universe_hash == context.dataset.universe_hash
        deployment_ok = (
            context.artifact.holding_period == context.artifact.horizon
            and context.artifact.execution_rule == "next_open"
        )
        adapter_ok = callable(self.adapter)
        reasons = tuple(
            reason
            for passed, reason in (
                (feature_ok, "feature hash is not inference-compatible"),
                (universe_ok, "universe identity differs"),
                (deployment_ok, "deployment execution contract differs"),
                (adapter_ok, "production inference adapter is unavailable"),
            )
            if not passed
        )
        return ShadowEligibilityEvidence(
            model_id=context.model.model_id,
            shadow_eligible=not reasons,
            feature_hash_compatible=feature_ok,
            universe_compatible=universe_ok,
            deployment_contract_compatible=deployment_ok,
            inference_adapter_available=adapter_ok,
            reasons=reasons,
        )
