"""Machine- and human-readable performance-monitor reporting."""

from __future__ import annotations

from numbers import Real
from typing import Any

import pandas as pd


def build_performance_summary(
    *,
    as_of: str,
    metrics: pd.DataFrame,
    details: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Build a deterministic summary with no execution-time fields."""

    normalized = metrics.astype(object).where(metrics.notna(), None)
    return {
        "schema_version": 1,
        "artifact_name": "performance_monitor_summary",
        "as_of": as_of,
        "models": normalized.to_dict("records"),
        "details": details,
        "warnings": warnings,
        "scope": {
            "observation_artifacts_only": True,
            "labels_read": False,
            "prices_read": False,
            "features_read": False,
            "inference_called": False,
            "backtest_called": False,
            "paper_trading_called": False,
            "registry_modified": False,
        },
    }


def render_performance_report(summary: dict[str, Any]) -> str:
    """Render stable Markdown explicitly describing observation statistics."""

    lines = [
        "# Prospective Model Performance Monitor",
        "",
        f"- As of: {summary['as_of']}",
        f"- Model-horizon groups: {len(summary['models'])}",
        "",
        "These are maturity-gated observation statistics, not portfolio or backtest returns.",
        "",
        "| Model | Role | Origin | Horizon | Rank IC | ICIR | Top10 Excess | Alpha Decay |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["models"]:
        lines.append(
            f"| {row['model_id']} | {row['model_role']} | {row['model_origin']} | "
            f"{row['horizon']} | "
            f"{_format(row['rank_ic'])} | {_format(row['icir'])} | "
            f"{_format(row['top10_average_excess_ret'])} | "
            f"{_format(row['alpha_decay_ratio'])} |"
        )
    if summary["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    lines.append("")
    return "\n".join(lines)


def _format(value: object) -> str:
    return f"{float(value):.6f}" if isinstance(value, Real) else "N/A"
