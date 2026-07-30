"""Deterministic Markdown rendering from validated structured output."""

from __future__ import annotations

from ashare_quant.research.agent.schemas import ResearchAgentSummary

_SECTIONS = (
    ("Market And Model Overview", "market_model_overview"),
    ("Champion Performance", "champion_performance"),
    ("Challenger Comparison", "challenger_comparison"),
    ("Alert Interpretation", "alert_interpretation"),
    ("Top Candidate Explanations", "candidate_explanations"),
    ("Paper Trading Status", "paper_trading_status"),
    ("Risk Summary", "risk_summary"),
    ("Data Limitations", "data_limitations"),
)


def render_daily_research(
    as_of: str,
    summary: ResearchAgentSummary,
    generation_mode: str,
) -> str:
    """Render the same validated JSON identically on every run."""

    lines = [
        "# Daily Quantitative Research",
        "",
        f"- Date: {as_of}",
        f"- Generation mode: {generation_mode}",
        "",
        "This report summarizes existing quantitative research artifacts only.",
    ]
    for title, field in _SECTIONS:
        lines.extend(["", f"## {title}", ""])
        conclusions = getattr(summary, field)
        lines.extend(f"- {conclusion.text}" for conclusion in conclusions)
    lines.append("")
    return "\n".join(lines)
