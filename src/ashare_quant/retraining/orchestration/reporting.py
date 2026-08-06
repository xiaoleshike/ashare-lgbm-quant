"""Deterministic operator report for retrained Challenger lifecycles."""

from __future__ import annotations

from ashare_quant.retraining.orchestration.schemas import LifecycleSnapshot


def render_lifecycle_report(snapshot: LifecycleSnapshot) -> str:
    summary = snapshot.summary
    lines = [
        "# Retrained Challenger Lifecycle",
        "",
        f"- Lifecycle run: {summary.lifecycle_run_id}",
        f"- Request: {summary.request_id}",
        f"- Parent model: {summary.parent_model_id}",
        f"- Candidate model: {summary.model_id or 'not trained'}",
        f"- Horizon: {summary.horizon}",
        f"- Current state: {summary.current_state}",
        f"- Observation: {summary.observation_status}",
        f"- Mature sessions: {summary.mature_sessions}/{summary.required_sessions}",
        f"- Promotion evidence: {summary.promotion_evidence_status}",
        f"- Successful Shadow runs: {len(summary.successful_shadow_run_ids)}",
        f"- Observation cutoff: {summary.observation_cutoff or 'none'}",
        f"- Frozen Promotion Policy: {summary.frozen_promotion_policy_hash or 'legacy'}",
        f"- Evaluated Promotion Policy: {summary.evaluated_promotion_policy_hash or 'none'}",
        f"- Policy drift: {summary.policy_drift}",
        "",
        "EVIDENCE_READY means exact immutable evidence can be prepared under the recorded "
        "evaluated Promotion Policy. It is not Gate PASS, approval, Champion replacement, "
        "or deployment.",
        "",
        "## Events",
        "",
    ]
    lines.extend(f"- {event.sequence}. {event.state}: {event.message}" for event in snapshot.events)
    lines.append("")
    return "\n".join(lines)
