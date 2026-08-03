"""Deterministic operator-facing retraining trigger output."""

from __future__ import annotations

from ashare_quant.retraining.schemas import RetrainingEvaluationResult


def render_evaluation(result: RetrainingEvaluationResult) -> str:
    """Render concise text without implying training or promotion occurred."""

    lines = [f"Retraining evaluation: {result.as_of}"]
    for item in result.decisions:
        reasons = ",".join(item.reasons) if item.reasons else "none"
        lines.append(
            f"{item.model_id} h{item.horizon}: {item.status} reasons={reasons} "
            f"sessions={item.observation_sessions}/{item.required_sessions} "
            f"request_id={item.request_id}"
        )
    return "\n".join(lines)
