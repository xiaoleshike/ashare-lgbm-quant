"""Deterministic JSON and Markdown rendering for local model explanations."""

from __future__ import annotations

from typing import Any

from ashare_quant.research.explainability.schemas import StockExplanation


def build_payload(
    *,
    as_of: str,
    model_id: str,
    feature_hash: str,
    feature_count: int,
    method: str,
    history_sessions: int,
    explanations: tuple[StockExplanation, ...],
) -> dict[str, Any]:
    """Build the structured explanation artifact."""

    return {
        "schema_version": 1,
        "artifact_name": "daily_model_explanations",
        "as_of": as_of,
        "model_id": model_id,
        "feature_hash": feature_hash,
        "feature_count": feature_count,
        "method": method,
        "history_sessions": history_sessions,
        "candidate_count": len(explanations),
        "interpretation": (
            "Feature contributions explain the model raw ranking score only. They do not "
            "change ranking and are not causal claims or trading recommendations."
        ),
        "stocks": [explanation.to_dict() for explanation in explanations],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render one stable human-readable explanation report."""

    lines = [
        "# Daily Model Explainability Report",
        "",
        f"- Date: {payload['as_of']}",
        f"- Model ID: `{payload['model_id']}`",
        f"- Method: `{payload['method']}`",
        f"- Candidates: {payload['candidate_count']}",
        f"- Prior same-model sessions: {payload['history_sessions']}",
        "",
        "> Feature contributions explain the unchanged model ranking score only. "
        "They are not causal conclusions, trading advice, or buy/sell instructions.",
    ]
    stocks = payload["stocks"]
    assert isinstance(stocks, list)
    for stock in stocks:
        assert isinstance(stock, dict)
        lines.extend(
            [
                "",
                f"## {stock['ts_code']}",
                "",
                f"- Model rank: {stock['model_rank']}",
                f"- Candidate rank: {stock['candidate_rank']}",
                f"- Prediction score: {float(stock['prediction_score']):.10f}",
                f"- Same-day score percentile: {float(stock['score_percentile']):.4%}",
                f"- Historical score percentile: {_percent(stock['historical_score_percentile'])}",
                f"- Signal strength: `{stock['signal_strength']}`",
                f"- History confidence: `{stock['confidence']}`",
                "",
                "### Positive Contributions",
                "",
            ]
        )
        lines.extend(_contribution_table(stock["positive_contributions"]))
        lines.extend(["", "### Negative Contributions", ""])
        lines.extend(_contribution_table(stock["negative_contributions"]))
    return "\n".join(lines) + "\n"


def _contribution_table(value: object) -> list[str]:
    rows = value if isinstance(value, list) else []
    lines = [
        "| Feature | Value | SHAP | Description |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        feature_value = row.get("value")
        displayed_value = "NA" if feature_value is None else f"{_numeric(feature_value):.8g}"
        lines.append(
            f"| `{row['feature']}` | {displayed_value} | {float(row['shap']):+.10f} | "
            f"{row['description']} |"
        )
    if not rows:
        lines.append("| - | - | - | No contribution in this direction |")
    return lines


def _percent(value: object) -> str:
    return "unavailable" if value is None else f"{_numeric(value):.4%}"


def _numeric(value: object) -> float:
    if isinstance(value, (int, float, str)):
        return float(value)
    raise ValueError(f"expected numeric explanation value, received {type(value).__name__}")
