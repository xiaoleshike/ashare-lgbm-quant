"""Deterministic Markdown rendering for execution readiness."""

from __future__ import annotations

from ashare_quant.retraining.readiness.schemas import RetrainingReadinessReport


def render_readiness(report: RetrainingReadinessReport) -> str:
    lines = [
        "# Retraining Execution Readiness",
        "",
        f"- As of: {report.as_of}",
        f"- Status: {report.status}",
        f"- Request: {report.request_id or 'unresolved'}",
        f"- Production run: {report.production_run_id or 'unresolved'}",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- {item.name}: {item.status} - {item.message}" for item in report.check_details)
    return "\n".join(lines) + "\n"
