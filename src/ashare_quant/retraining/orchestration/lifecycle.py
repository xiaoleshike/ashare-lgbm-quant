"""Explicit state machine for governed retrained Challenger lifecycles."""

from __future__ import annotations

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.retraining.orchestration.schemas import LifecycleState

_TRANSITIONS: dict[LifecycleState, frozenset[LifecycleState]] = {
    "REQUEST_ACCEPTED": frozenset({"READINESS_CHECKING", "CANCELLED"}),
    "READINESS_CHECKING": frozenset({"READINESS_FAILED", "READINESS_READY", "FAILED"}),
    "READINESS_FAILED": frozenset({"READINESS_CHECKING", "CANCELLED"}),
    "READINESS_READY": frozenset(
        {
            "READINESS_CHECKING",
            "TRAINING",
            "TRAINING_COOLDOWN_BLOCKED",
            "TRAINING_BUDGET_BLOCKED",
            "CANCELLED",
        }
    ),
    "TRAINING_COOLDOWN_BLOCKED": frozenset(
        {"READINESS_CHECKING", "TRAINING", "TRAINING_COOLDOWN_BLOCKED", "CANCELLED"}
    ),
    "TRAINING_BUDGET_BLOCKED": frozenset(
        {"READINESS_CHECKING", "TRAINING", "TRAINING_BUDGET_BLOCKED", "CANCELLED"}
    ),
    "TRAINING": frozenset({"TRAINING_FAILED", "TRAINING_COMPLETED", "FAILED"}),
    "TRAINING_FAILED": frozenset({"TRAINING", "CANCELLED"}),
    "TRAINING_COMPLETED": frozenset({"VALIDATING", "CANCELLED"}),
    "VALIDATING": frozenset({"VALIDATION_FAILED", "VALIDATION_COMPLETED", "FAILED"}),
    "VALIDATION_FAILED": frozenset({"VALIDATING", "CANCELLED"}),
    "VALIDATION_COMPLETED": frozenset({"SHADOW_ENROLLING", "CANCELLED"}),
    "SHADOW_ENROLLING": frozenset({"SHADOW_FAILED", "SHADOW_ENROLLED", "FAILED"}),
    "SHADOW_FAILED": frozenset({"SHADOW_ENROLLING", "CANCELLED"}),
    "SHADOW_ENROLLED": frozenset({"SHADOW_ENROLLED", "OBSERVATION_PENDING", "CANCELLED"}),
    "OBSERVATION_PENDING": frozenset(
        {"OBSERVATION_PENDING", "OBSERVATION_ACCUMULATING", "OBSERVATION_SUFFICIENT", "CANCELLED"}
    ),
    "OBSERVATION_ACCUMULATING": frozenset(
        {"OBSERVATION_ACCUMULATING", "OBSERVATION_SUFFICIENT", "CANCELLED"}
    ),
    "OBSERVATION_SUFFICIENT": frozenset(
        {"OBSERVATION_SUFFICIENT", "POLICY_REVIEW_REQUIRED", "EVIDENCE_READY", "CANCELLED"}
    ),
    "POLICY_REVIEW_REQUIRED": frozenset({"POLICY_REVIEW_REQUIRED", "EVIDENCE_READY", "CANCELLED"}),
    "EVIDENCE_READY": frozenset({"EVIDENCE_READY", "POLICY_REVIEW_REQUIRED"}),
    "FAILED": frozenset(),
    "CANCELLED": frozenset(),
}


def require_transition(current: LifecycleState, target: LifecycleState) -> None:
    """Reject skipped, governance-bypassing, or terminal transitions."""

    if target not in _TRANSITIONS[current]:
        raise DataValidationError(f"forbidden lifecycle transition: {current} -> {target}")
