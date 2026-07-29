"""Deterministic human-readable monitoring report rendering."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ashare_quant.monitoring.schemas import HealthMetrics

type DataFrame = pd.DataFrame


def build_monitor_summary(
    health: HealthMetrics,
    performance: dict[str, Any],
    portfolios: DataFrame,
) -> dict[str, Any]:
    """Build the compact machine-readable monitoring summary."""

    return {
        "schema_version": 1,
        "artifact_name": "production_monitor_summary",
        "as_of": health.as_of,
        "model_id": health.model_id,
        "health": health.to_dict(),
        "performance": performance,
        "portfolio_count": len(portfolios),
        "portfolios": portfolios.to_dict("records"),
        "scope": {
            "labels_read": False,
            "model_rescored": False,
            "trading_state_modified": False,
            "orders_generated": False,
        },
    }


def render_monitor_report(summary: dict[str, Any]) -> str:
    """Render stable Markdown without changing source artifacts."""

    health = summary["health"]
    lines = [
        "# Production Model Monitor",
        "",
        "## Health",
        "",
        f"- Date: {summary['as_of']}",
        f"- Model: {summary['model_id']}",
        f"- Universe: {health['universe_size']}",
        f"- Model universe: {health['model_universe_size']}",
        f"- Predictions: {health['prediction_count']}",
        f"- Candidates: {health['candidate_count']}",
        f"- Feature coverage: {health['feature_coverage']:.4f}",
        f"- Score mean/std: {health['score_mean']:.6f} / {health['score_std']:.6f}",
        f"- Score P90-P10 spread: {health['score_spread']:.6f}",
        f"- Duplicate score ratio: {health['duplicate_score_ratio']:.4f}",
        "",
        "## Paper Portfolios",
        "",
        "| Portfolio | NAV | Daily Return | Drawdown | Turnover | Positions | Cash Ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["portfolios"]:
        lines.append(
            f"| {row['portfolio_id']} | {row['nav']:.6f} | "
            f"{row['daily_return']:.6f} | {row['drawdown']:.6f} | "
            f"{row['turnover']:.6f} | {row['position_count']} | "
            f"{row['cash_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Prospective Performance",
            "",
            f"- Model-horizon groups: {len(summary['performance']['models'])}",
            f"- Warnings: {len(summary['performance']['warnings'])}",
            "",
            "## Scope",
            "",
            "This report is read-only. It does not load labels, rescore the model, "
            "generate orders, or modify paper-trading state.",
            "",
        ]
    )
    return "\n".join(lines)
