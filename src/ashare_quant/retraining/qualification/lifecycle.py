"""Explicit qualification state machine with no promotion transitions."""

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.retraining.qualification.schemas import QualificationState

_TRANSITIONS: dict[QualificationState, frozenset[QualificationState]] = {
    "CREATED": frozenset({"PREFLIGHT_CHECKING", "CANCELLED"}),
    "PREFLIGHT_CHECKING": frozenset({"PREFLIGHT_READY", "PREFLIGHT_BLOCKED", "FAILED"}),
    "PREFLIGHT_BLOCKED": frozenset({"PREFLIGHT_CHECKING", "CANCELLED"}),
    "PREFLIGHT_READY": frozenset({"DRY_RUN_CHECKING", "CANCELLED"}),
    "DRY_RUN_CHECKING": frozenset({"DRY_RUN_READY", "DRY_RUN_BLOCKED", "FAILED"}),
    "DRY_RUN_BLOCKED": frozenset({"DRY_RUN_CHECKING", "CANCELLED"}),
    "DRY_RUN_READY": frozenset({"READINESS_CHECKING", "CANCELLED"}),
    "READINESS_CHECKING": frozenset({"READINESS_READY", "READINESS_FAILED", "FAILED"}),
    "READINESS_FAILED": frozenset({"READINESS_CHECKING", "CANCELLED"}),
    "READINESS_READY": frozenset({"TRAINING_PENDING_APPROVAL", "CANCELLED"}),
    "TRAINING_PENDING_APPROVAL": frozenset({"TRAINING_PENDING_APPROVAL", "TRAINING", "CANCELLED"}),
    "TRAINING": frozenset({"TRAINING_COMPLETED", "TRAINING_FAILED", "FAILED"}),
    "TRAINING_FAILED": frozenset({"TRAINING", "CANCELLED"}),
    "TRAINING_COMPLETED": frozenset({"VALIDATION_PENDING_APPROVAL", "CANCELLED"}),
    "VALIDATION_PENDING_APPROVAL": frozenset({"VALIDATING", "CANCELLED"}),
    "VALIDATING": frozenset({"VALIDATION_COMPLETED", "VALIDATION_FAILED", "FAILED"}),
    "VALIDATION_FAILED": frozenset({"VALIDATING", "CANCELLED"}),
    "VALIDATION_COMPLETED": frozenset({"SHADOW_PENDING_APPROVAL", "CANCELLED"}),
    "SHADOW_PENDING_APPROVAL": frozenset(
        {"SHADOW_PENDING_APPROVAL", "SHADOW_ENROLLING", "CANCELLED"}
    ),
    "SHADOW_ENROLLING": frozenset({"SHADOW_ENROLLED", "SHADOW_FAILED", "FAILED"}),
    "SHADOW_FAILED": frozenset({"SHADOW_ENROLLING", "CANCELLED"}),
    "SHADOW_ENROLLED": frozenset({"OBSERVATION_CHECKING", "CANCELLED"}),
    "OBSERVATION_CHECKING": frozenset(
        {"OBSERVATION_PENDING", "OBSERVATION_ACCUMULATING", "FAILED"}
    ),
    "OBSERVATION_PENDING": frozenset({"QUALIFIED", "OBSERVATION_CHECKING", "CANCELLED"}),
    "OBSERVATION_ACCUMULATING": frozenset({"QUALIFIED", "OBSERVATION_CHECKING", "CANCELLED"}),
    "QUALIFIED": frozenset(),
    "FAILED": frozenset(),
    "CANCELLED": frozenset(),
}


def require_qualification_transition(
    current: QualificationState, target: QualificationState
) -> None:
    if target not in _TRANSITIONS[current]:
        raise DataValidationError(f"forbidden qualification transition: {current} -> {target}")
