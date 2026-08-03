"""Horizon-isolated retraining trigger evaluation."""

from __future__ import annotations

from typing import Any, cast

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.retraining.configuration import RetrainingPolicy
from ashare_quant.retraining.rules import evaluate_trigger_reasons
from ashare_quant.retraining.schemas import (
    ModelRole,
    RetrainingDecision,
    RetrainingSources,
)

_TRAINABLE_ROLES = {
    "champion",
    "challenger_h5",
    "challenger_h10",
    "challenger_h20",
    "challenger_h60",
}
_HORIZONS = {5, 10, 20, 60}
_ROLE_HORIZON = {
    "challenger_h5": 5,
    "challenger_h10": 10,
    "challenger_h20": 20,
    "challenger_h60": 60,
}


def _validate_target_identity(model_id: str, role: str, horizon: int) -> None:
    if not model_id or role not in _TRAINABLE_ROLES or horizon not in _HORIZONS:
        raise DataValidationError(
            f"unsupported retraining target identity: model={model_id} role={role} "
            f"horizon={horizon}"
        )
    expected_horizon = _ROLE_HORIZON.get(role)
    if expected_horizon is not None and horizon != expected_horizon:
        raise DataValidationError(
            f"retraining model role/horizon mismatch: role={role} horizon={horizon}"
        )


def evaluate_sources(
    sources: RetrainingSources,
    policy: RetrainingPolicy,
) -> tuple[tuple[RetrainingDecision, dict[str, Any]], ...]:
    """Evaluate each trainable model/horizon independently."""

    results: list[tuple[RetrainingDecision, dict[str, Any]]] = []
    for raw_row in sources.performance_metrics.to_dict("records"):
        row: dict[str, Any] = {str(key): value for key, value in raw_row.items()}
        model_id = str(row.get("model_id") or "")
        role = str(row.get("model_role") or "")
        horizon = int(row.get("horizon", -1))
        if role == "multi_horizon_ensemble":
            continue
        _validate_target_identity(model_id, role, horizon)
        sessions = int(row.get("sessions", 0))
        required = policy.required_sessions(horizon)
        if not policy.enabled:
            status = "DISABLED"
            reasons: tuple[str, ...] = ()
        elif sessions < required:
            status = "INSUFFICIENT_OBSERVATIONS"
            reasons = ()
        else:
            reasons = evaluate_trigger_reasons(row=row, sources=sources, policy=policy)
            status = "TRIGGERED" if reasons else "NO_ACTION_REQUIRED"
        decision = RetrainingDecision(
            model_id=model_id,
            model_role=role,
            horizon=horizon,
            status=cast(Any, status),
            reasons=reasons,
            observation_sessions=sessions,
            required_sessions=required,
        )
        results.append((decision, row))
    return tuple(results)


def select_manual_target(
    sources: RetrainingSources,
    model_id: str,
    policy: RetrainingPolicy,
) -> tuple[RetrainingDecision, dict[str, Any]]:
    """Resolve one unambiguous model identity for an operator-created request."""

    matches = [
        {str(key): value for key, value in row.items()}
        for row in sources.performance_metrics.to_dict("records")
        if str(row.get("model_id")) == model_id and str(row.get("model_role")) in _TRAINABLE_ROLES
    ]
    if len(matches) != 1:
        raise DataValidationError(
            f"manual retraining model must identify exactly one monitored horizon: {model_id}"
        )
    row = matches[0]
    role = str(row["model_role"])
    horizon = int(row["horizon"])
    _validate_target_identity(model_id, role, horizon)
    sessions = int(row.get("sessions", 0))
    required = policy.required_sessions(horizon)
    if sessions < required:
        raise DataValidationError(
            f"manual retraining request lacks mature observations: sessions={sessions} "
            f"required={required}"
        )
    return (
        RetrainingDecision(
            model_id=model_id,
            model_role=cast(ModelRole, role),
            horizon=horizon,
            status="TRIGGERED",
            reasons=("manual_request",),
            observation_sessions=sessions,
            required_sessions=required,
        ),
        row,
    )
