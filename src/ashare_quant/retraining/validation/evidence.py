"""Validation evidence composition without promotion side effects."""

from __future__ import annotations

from ashare_quant.retraining.validation.schemas import (
    CandidateValidationContext,
    ExecutableValidationEvidence,
    OfflineValidationEvidence,
    ShadowEligibilityEvidence,
    ValidationEvidence,
)


def build_validation_evidence(
    *,
    run_id: str,
    context: CandidateValidationContext,
    offline: OfflineValidationEvidence,
    executable: ExecutableValidationEvidence,
    shadow: ShadowEligibilityEvidence,
    minimum_sessions: int,
) -> ValidationEvidence:
    """Summarize evidence completeness; never create a promotion request."""

    promotion_ready = bool(
        shadow.shadow_eligible
        and offline.evaluation_sessions >= minimum_sessions
        and executable.signal_dates == offline.evaluation_sessions
        and not executable.unresolved_holdings
    )
    return ValidationEvidence(
        run_id=run_id,
        model_id=context.model.model_id,
        candidate_registration_id=context.registration.candidate_registration_id,
        training_run_id=context.artifact.training_run_id,
        shadow_eligible=shadow.shadow_eligible,
        promotion_ready=promotion_ready,
    )
