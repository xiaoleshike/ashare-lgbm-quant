"""Source and generated-output validation for the research agent."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.research.agent.schemas import (
    CollectedArtifacts,
    ResearchAgentSummary,
    ResearchContext,
    ResearchFact,
)

_CHINESE_FORBIDDEN = ("买入", "卖出", "加仓", "减仓", "仓位", "目标价", "止盈", "止损")
_ENGLISH_FORBIDDEN = (
    "buy",
    "sell",
    "position",
    "allocation",
    "target price",
    "stop loss",
    "take profit",
)
_STOCK_PATTERN = re.compile(r"\b\d{6}\.(?:SH|SZ|BJ)\b")
_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?")
_RANK_PATTERNS = (
    re.compile(r"(?P<code>\d{6}\.(?:SH|SZ|BJ)).{0,30}?\brank\s+(?P<rank>\d+)", re.I),
    re.compile(r"(?P<code>\d{6}\.(?:SH|SZ|BJ)).{0,30}?排名[:：]?\s*(?P<rank>\d+)"),
)


def validate_collected_artifacts(collected: CollectedArtifacts) -> None:
    """Validate schema, cross-artifact identity, and recorded file hashes."""

    payloads = collected.payloads
    as_of = collected.as_of
    production = _identity(payloads["production_summary"], "production_daily_summary", as_of)
    production_manifest = _identity(
        payloads["production_manifest"], "production_predictions", as_of
    )
    candidates_manifest = _identity(payloads["candidates_manifest"], "production_candidates", as_of)
    decision = _identity(payloads["decision"], "daily_investment_decision_support", as_of)
    explanations = _identity(payloads["explanations"], "daily_model_explanations", as_of)
    research = _identity(payloads["research_summary"], "daily_quantitative_research_report", as_of)
    monitor_manifest = _identity(payloads["monitor_manifest"], "production_monitor_manifest", as_of)
    monitor = _identity(payloads["monitor_summary"], "production_monitor_summary", as_of)
    alerts = _identity(payloads["alerts"], "monitoring_alerts", as_of)
    alerts_manifest = _identity(payloads["alerts_manifest"], "alert_engine", as_of)
    performance_manifest_value = payloads["performance_manifest"]
    performance_manifest = (
        _identity(performance_manifest_value, "performance_monitor", as_of)
        if performance_manifest_value is not None
        else None
    )
    paper = _identity(payloads["paper_summary"], "paper_trading_daily_report", as_of)

    model_id = _required_string(production, "model_id", "production summary")
    for name, payload in (
        ("production manifest", production_manifest),
        ("candidates manifest", candidates_manifest),
        ("decision", decision),
        ("explanations", explanations),
        ("research summary", research),
        ("monitor manifest", monitor_manifest),
        ("monitor summary", monitor),
    ):
        if payload.get("model_id") != model_id:
            raise DataValidationError(f"{name} model_id differs from production summary")
    candidate_count = int(production.get("candidate_count", -1))
    if int(candidates_manifest.get("candidate_count", -2)) != candidate_count:
        raise DataValidationError("candidates manifest count differs from production summary")
    if candidates_manifest.get("feature_hash") != decision.get(
        "feature_hash"
    ) or candidates_manifest.get("feature_hash") != explanations.get("feature_hash"):
        raise DataValidationError("candidate, decision, and explanation feature hashes differ")
    health = payloads["health"]
    if health.get("as_of") != as_of or health.get("model_id") != model_id:
        raise DataValidationError("health identity differs from production summary")
    _validate_candidates(production, decision, explanations)
    _validate_monitor_hashes(collected)
    _validate_alerts(alerts, alerts_manifest)
    _validate_performance(payloads["performance_metrics"], performance_manifest)
    _validate_portfolios(payloads["portfolio_metrics"])
    constraints = paper.get("constraints")
    if not isinstance(constraints, dict) or constraints.get("real_orders_generated") is not False:
        raise DataValidationError("paper summary does not assert no real orders")


def parse_and_validate_summary(
    raw: str,
    context: ResearchContext,
    *,
    allow_advisory_language: bool = True,
) -> ResearchAgentSummary:
    """Parse strict JSON and enforce factual grounding and configured language policy."""

    if raw.lstrip().startswith("```"):
        raise DataValidationError("LLM response must be JSON only, without code fences")
    try:
        value = json.loads(raw)
        summary = ResearchAgentSummary.model_validate(value)
    except (json.JSONDecodeError, ValidationError) as error:
        raise DataValidationError(f"invalid structured LLM response: {error}") from error
    validate_summary(
        summary,
        context,
        allow_advisory_language=allow_advisory_language,
    )
    return summary


def validate_summary(
    summary: ResearchAgentSummary,
    context: ResearchContext,
    *,
    allow_advisory_language: bool = True,
) -> None:
    """Require fact grounding, unchanged candidate identities, and policy compliance."""

    facts = {fact.fact_id: fact for fact in context.fact_catalog}
    candidate_ranks = {
        str(row["ts_code"]): int(row["rank"])
        for row in context.candidates
        if "ts_code" in row and "rank" in row
    }
    cited: set[str] = set()
    for conclusion in _conclusions(summary):
        unknown = sorted(set(conclusion.fact_ids) - set(facts))
        if unknown:
            raise DataValidationError(f"research conclusion cites unknown fact_id: {unknown}")
        cited.update(conclusion.fact_ids)
        if not allow_advisory_language:
            _reject_forbidden_language(conclusion.text)
        _validate_stocks_and_ranks(conclusion.text, candidate_ranks)
        _validate_numeric_grounding(
            conclusion.text,
            [facts[fact_id] for fact_id in conclusion.fact_ids],
            tuple(candidate_ranks),
        )
    if set(summary.source_fact_ids) != cited:
        raise DataValidationError("source_fact_ids must equal the facts cited by conclusions")
    if len(summary.source_fact_ids) != len(set(summary.source_fact_ids)):
        raise DataValidationError("source_fact_ids contains duplicates")


def _validate_monitor_hashes(collected: CollectedArtifacts) -> None:
    payloads = collected.payloads
    hashes = collected.source_hashes
    expected = payloads["monitor_manifest"].get("monitor_metric_file_hashes")
    if not isinstance(expected, dict):
        raise DataValidationError("monitor manifest lacks metric file hashes")
    for source_name, manifest_name in (
        ("health", "health"),
        ("portfolio_metrics", "portfolio_metrics"),
    ):
        if hashes[source_name] != expected.get(manifest_name):
            raise DataValidationError(f"monitor source hash mismatch: {source_name}")
    performance_names = ("performance_metrics", "performance_manifest")
    for source_name in performance_names:
        expected_hash = expected.get(source_name)
        actual_hash = hashes.get(source_name)
        if expected_hash is not None and actual_hash is None:
            raise DataValidationError(f"monitor manifest references missing source: {source_name}")
        if actual_hash is not None and actual_hash != expected_hash:
            raise DataValidationError(f"monitor source hash mismatch: {source_name}")
    if hashes["alerts"] != payloads["alerts_manifest"].get("alerts_file_sha256"):
        raise DataValidationError("alerts hash differs from alert manifest")
    performance_manifest = payloads["performance_manifest"]
    if performance_manifest is not None and (
        hashes["performance_metrics"] != performance_manifest.get("metrics_file_sha256")
    ):
        raise DataValidationError("performance metrics hash differs from manifest")


def _validate_candidates(
    production: dict[str, Any],
    decision: dict[str, Any],
    explanations: dict[str, Any],
) -> None:
    candidate_count = int(production.get("candidate_count", -1))
    if (
        int(decision.get("candidate_count", -2)) != candidate_count
        or int(explanations.get("candidate_count", -3)) != candidate_count
    ):
        raise DataValidationError("candidate counts differ across research artifacts")
    decision_rows = _records(decision, "stocks", "decision")
    explanation_rows = _records(explanations, "stocks", "explanations")
    decision_identity = [
        (str(row.get("ts_code")), int(row.get("candidate_rank", -1))) for row in decision_rows
    ]
    explanation_identity = [
        (str(row.get("ts_code")), int(row.get("candidate_rank", -1))) for row in explanation_rows
    ]
    if len(set(decision_identity)) != len(decision_identity):
        raise DataValidationError("decision artifact contains duplicate candidate identities")
    if decision_identity != explanation_identity:
        raise DataValidationError("decision and explanation candidate rankings differ")
    if decision_identity != sorted(decision_identity, key=lambda item: (item[1], item[0])):
        raise DataValidationError("candidate rows are not deterministically ranked")


def _validate_alerts(alerts: dict[str, Any], manifest: dict[str, Any]) -> None:
    rows = _records(alerts, "alerts", "alerts")
    if len(rows) != int(manifest.get("alert_count", -1)):
        raise DataValidationError("alert count differs from alert manifest")
    identifiers = [str(row.get("alert_id", "")) for row in rows]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise DataValidationError("alerts contain invalid or duplicate identities")


def _validate_performance(frame: object, manifest: dict[str, Any] | None) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError("performance metrics must be tabular")
    required = {"model_id", "model_role", "horizon", "rank_ic", "alpha_decay_ratio"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"performance metrics lack columns: {missing}")
    if manifest is None:
        if not frame.empty:
            raise DataValidationError("performance metrics exist without a manifest")
        return
    if manifest.get("status") != "success":
        raise DataValidationError("performance manifest is not successful")
    expected = manifest.get("row_counts")
    if not isinstance(expected, dict) or int(expected.get("model_horizon_metrics", -1)) != len(
        frame
    ):
        raise DataValidationError("performance metrics row count differs from manifest")


def _validate_portfolios(frame: object) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError("portfolio metrics must be tabular")
    required = {
        "portfolio_id",
        "nav",
        "drawdown",
        "turnover",
        "position_count",
        "cash_ratio",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"portfolio metrics lack columns: {missing}")
    if not frame.empty and frame["portfolio_id"].astype(str).duplicated().any():
        raise DataValidationError("portfolio metrics contain duplicate portfolio_id")


def _identity(payload: object, artifact_name: str, as_of: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise DataValidationError(f"{artifact_name} must be an object")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_name") != artifact_name
        or str(payload.get("as_of") or payload.get("observation_as_of")) != as_of
    ):
        raise DataValidationError(f"invalid {artifact_name} identity")
    return payload


def _records(payload: dict[str, Any], key: str, description: str) -> list[dict[str, Any]]:
    rows = payload.get(key)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise DataValidationError(f"{description} lacks valid {key}")
    return rows


def _required_string(payload: dict[str, Any], key: str, description: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise DataValidationError(f"{description} lacks {key}")
    return value


def _conclusions(summary: ResearchAgentSummary) -> Iterable[Any]:
    for field in (
        summary.market_model_overview,
        summary.champion_performance,
        summary.challenger_comparison,
        summary.alert_interpretation,
        summary.candidate_explanations,
        summary.paper_trading_status,
        summary.risk_summary,
        summary.data_limitations,
    ):
        yield from field


def _reject_forbidden_language(text: str) -> None:
    lowered = text.casefold()
    found = [term for term in _CHINESE_FORBIDDEN if term in text]
    found.extend(
        term
        for term in _ENGLISH_FORBIDDEN
        if re.search(rf"\b{re.escape(term)}\b", lowered, flags=re.IGNORECASE)
    )
    if found:
        raise DataValidationError(f"research output contains prohibited language: {sorted(found)}")


def _validate_stocks_and_ranks(text: str, candidate_ranks: dict[str, int]) -> None:
    mentioned = set(_STOCK_PATTERN.findall(text))
    invented = sorted(mentioned - set(candidate_ranks))
    if invented:
        raise DataValidationError(f"research output invents candidate stocks: {invented}")
    for pattern in _RANK_PATTERNS:
        for match in pattern.finditer(text):
            code, rank = match.group("code"), int(match.group("rank"))
            if candidate_ranks.get(code) != rank:
                raise DataValidationError(f"research output changes candidate ranking: {code}")


def _validate_numeric_grounding(
    text: str,
    facts: list[ResearchFact],
    stock_codes: tuple[str, ...],
) -> None:
    scrubbed = text
    for code in stock_codes:
        scrubbed = scrubbed.replace(code, "")
    allowed = _numeric_tokens([fact.value for fact in facts])
    unsupported = sorted(set(_NUMBER_PATTERN.findall(scrubbed)) - allowed)
    if unsupported:
        raise DataValidationError(f"research output contains unsupported metrics: {unsupported}")


def _numeric_tokens(values: list[Any]) -> set[str]:
    tokens: set[str] = set()
    pending: list[Any] = list(values)
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
        elif isinstance(value, bool) or value is None:
            continue
        elif isinstance(value, int | np.integer):
            tokens.add(str(int(value)))
        elif isinstance(value, float | np.floating) and np.isfinite(float(value)):
            tokens.add(format(float(value), ".17g"))
            tokens.add(str(float(value)))
        elif isinstance(value, str):
            tokens.update(_NUMBER_PATTERN.findall(value))
    return tokens
