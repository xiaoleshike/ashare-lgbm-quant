"""Deterministic normalization and fact-catalog construction."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import canonical_payload_hash
from ashare_quant.research.agent.schemas import (
    CollectedArtifacts,
    ResearchContext,
    ResearchFact,
)

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]


def build_research_context(
    collected: CollectedArtifacts,
    *,
    top_candidates: int,
) -> ResearchContext:
    """Normalize structured reports and exclude all untrusted Markdown text."""

    payloads = collected.payloads
    production_summary = payloads["production_summary"]
    research_summary = payloads["research_summary"]
    health = payloads["health"]
    performance = _records(payloads["performance_metrics"], ("model_id", "horizon"))
    alerts = tuple(
        sorted(
            (_json_object(row) for row in payloads["alerts"]["alerts"]),
            key=lambda row: str(row["alert_id"]),
        )
    )
    decision_rows = sorted(
        payloads["decision"]["stocks"],
        key=lambda row: (int(row["candidate_rank"]), str(row["ts_code"])),
    )[:top_candidates]
    candidates = tuple(_candidate(row) for row in decision_rows)
    paper_rows = tuple(
        sorted(
            (_json_object(row) for row in payloads["paper_summary"]["portfolios"]),
            key=lambda row: str(row["portfolio_id"]),
        )
    )
    production = {
        "as_of": collected.as_of,
        "run_id": production_summary["run_id"],
        "model_id": production_summary["model_id"],
        "candidate_count": production_summary["candidate_count"],
        "prediction_count": research_summary["prediction_count"],
    }
    model_health = _json_object(health)
    data_availability = {
        "performance_available": bool(performance),
        "performance_groups": len(performance),
        "alerts_available": bool(alerts),
        "paper_portfolios": len(paper_rows),
        "candidate_details": len(candidates),
        "markdown_admitted_to_context": False,
        "monitor_warnings": list(
            payloads["monitor_summary"].get("performance", {}).get("warnings", [])
        ),
        "alert_warnings": list(payloads["alerts"].get("warnings", [])),
    }
    facts: list[ResearchFact] = []
    _fact(facts, "production", "overview", production)
    _fact(facts, "health", "model_health", model_health)
    _fact(
        facts,
        "availability",
        "data_availability",
        data_availability,
    )
    for row in performance:
        _fact(
            facts,
            "performance",
            f"{row.get('model_id')}:{row.get('horizon')}",
            row,
        )
    for row in alerts:
        _fact(facts, "alert", str(row["alert_id"]), row)
    for row in candidates:
        _fact(facts, "candidate", f"{row['rank']}:{row['ts_code']}", row)
    for row in paper_rows:
        _fact(facts, "paper_portfolio", str(row["portfolio_id"]), row)
    return ResearchContext(
        as_of=collected.as_of,
        production=production,
        model_health=model_health,
        performance=performance,
        alerts=alerts,
        candidates=candidates,
        paper_portfolios=paper_rows,
        data_availability=data_availability,
        fact_catalog=tuple(sorted(facts, key=lambda fact: fact.fact_id)),
    )


def context_hash(context: ResearchContext) -> str:
    """Hash the complete normalized context."""

    return canonical_payload_hash(context.model_dump(mode="json"))


def _fact(facts: list[ResearchFact], category: str, key: str, value: object) -> None:
    normalized = _json_value(value)
    fact_id = canonical_payload_hash({"category": category, "key": key, "value": normalized})
    facts.append(ResearchFact(fact_id=fact_id, category=category, key=key, value=normalized))


def _candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": int(row["candidate_rank"]),
        "ts_code": str(row["ts_code"]),
        "prediction_score": float(row["prediction_score"]),
        "signal_strength": str(row["signal_strength"]),
        "confidence": str(row["confidence"]),
        "positive_contributions": _json_value(row.get("positive_contributions", [])[:3]),
        "negative_contributions": _json_value(row.get("negative_contributions", [])[:3]),
        "risk_observations": _json_value(row.get("risk_observations", [])),
    }


def _records(frame: object, sort_columns: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return ()
    ordered = frame.sort_values(list(sort_columns), kind="mergesort")
    normalized = ordered.astype(object).where(ordered.notna(), None)
    records = cast(list[dict[str, Any]], normalized.to_dict("records"))
    return tuple(_json_object(record) for record in records)


def _json_object(value: object) -> dict[str, JsonValue]:
    normalized = _json_value(value)
    if not isinstance(normalized, dict):
        raise DataValidationError("research context expected a JSON object")
    return normalized


def _json_value(value: object) -> JsonValue:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return None if pd.isna(number) else number
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, str):
        return value
    if hasattr(value, "item"):
        return _json_value(value.item())
    return str(value)
