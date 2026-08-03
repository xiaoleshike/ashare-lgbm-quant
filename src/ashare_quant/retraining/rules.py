"""Pure trigger rules over already-computed monitoring metrics."""

from __future__ import annotations

import math
from typing import Any

from ashare_quant.retraining.configuration import RetrainingPolicy
from ashare_quant.retraining.schemas import RetrainingSources


def evaluate_trigger_reasons(
    *,
    row: dict[str, Any],
    sources: RetrainingSources,
    policy: RetrainingPolicy,
) -> tuple[str, ...]:
    """Return deterministic reasons without recomputing any research metric."""

    reasons: list[str] = []
    model_id = str(row["model_id"])
    if policy.triggers.alpha_decay.enabled:
        value = _finite(row.get("alpha_decay_ratio"))
        if value is not None and value < policy.triggers.alpha_decay.threshold:
            reasons.append("alpha_decay")
    if policy.triggers.ic_decline.enabled:
        column = f"rolling_{policy.triggers.ic_decline.rolling_window}_ic_mean"
        value = _finite(row.get(column))
        if value is not None and value < policy.triggers.ic_decline.threshold:
            reasons.append("ic_decline")
    if policy.triggers.feature_drift.enabled and model_id == str(sources.health["model_id"]):
        drift = sources.health.get("drift_reference")
        metrics = drift.get("metrics") if isinstance(drift, dict) else None
        value = _finite(metrics.get("maximum_feature_psi")) if isinstance(metrics, dict) else None
        if value is not None and value >= policy.triggers.feature_drift.psi_threshold:
            reasons.append("feature_drift")
    if policy.triggers.critical_alert.enabled and _has_critical_alert(
        alerts=sources.alerts,
        model_id=model_id,
        champion_model_id=str(sources.health["model_id"]),
    ):
        reasons.append("critical_alert")
    return tuple(reasons)


def _has_critical_alert(
    *,
    alerts: dict[str, Any],
    model_id: str,
    champion_model_id: str,
) -> bool:
    rows = alerts.get("alerts")
    if not isinstance(rows, list):
        return False
    for item in rows:
        if not isinstance(item, dict):
            continue
        if item.get("severity") != "CRITICAL" or item.get("status") not in {"NEW", "ACTIVE"}:
            continue
        alert_model = item.get("model_id")
        if alert_model == model_id or (alert_model is None and model_id == champion_model_id):
            return True
    return False


def _finite(value: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    result = float(value)
    return result if math.isfinite(result) else None
