"""Deterministic alert JSON and Markdown reporting."""

from __future__ import annotations

from typing import Any

from ashare_quant.monitoring.alerts.schemas import Alert


def build_alert_payload(
    as_of: str,
    alerts: tuple[Alert, ...],
    warnings: tuple[str, ...],
) -> dict[str, Any]:
    """Build the machine-readable alert publication."""

    return {
        "schema_version": 1,
        "artifact_name": "monitoring_alerts",
        "as_of": as_of,
        "alerts": [alert.to_dict() for alert in alerts],
        "warnings": list(warnings),
    }


def render_alert_report(payload: dict[str, Any]) -> str:
    """Render alerts in deterministic severity and identity order."""

    alerts = sorted(
        payload["alerts"],
        key=lambda item: (-_severity_rank(str(item["severity"])), str(item["alert_id"])),
    )
    lines = [
        "# Monitoring Alerts",
        "",
        f"- As of: {payload['as_of']}",
        f"- Lifecycle events: {len(alerts)}",
        "",
        "| Severity | Status | Type | Model | Portfolio | Metric | Value | Threshold |",
        "|---|---|---|---|---|---|---:|---:|",
    ]
    for alert in alerts:
        lines.append(
            f"| {alert['severity']} | {alert['status']} | {alert['alert_type']} | "
            f"{alert['model_id'] or ''} | {alert['portfolio_id'] or ''} | "
            f"{alert['metric_name']} | {float(alert['metric_value']):.6f} | "
            f"{float(alert['threshold']):.6f} |"
        )
    if payload["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    lines.append("")
    return "\n".join(lines)


def _severity_rank(value: str) -> int:
    return {"INFO": 1, "WARNING": 2, "CRITICAL": 3}[value]
