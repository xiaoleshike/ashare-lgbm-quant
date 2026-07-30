"""Deterministic grounded report generation without an LLM."""

from __future__ import annotations

from ashare_quant.research.agent.schemas import (
    ResearchAgentSummary,
    ResearchConclusion,
    ResearchContext,
    ResearchFact,
)


def deterministic_fallback(context: ResearchContext) -> ResearchAgentSummary:
    """Generate all required sections directly from citable facts."""

    by_category: dict[str, list[ResearchFact]] = {}
    for fact in context.fact_catalog:
        by_category.setdefault(fact.category, []).append(fact)
    production = by_category["production"][0]
    health = by_category["health"][0]
    availability = by_category["availability"][0]
    production_value = production.value
    health_value = health.value
    overview = (
        _conclusion(
            "The production model "
            f"{production_value['model_id']} produced "
            f"{production_value['candidate_count']} candidates from "
            f"{production_value['prediction_count']} predictions.",
            production,
        ),
        _conclusion(
            "Model health reports score standard deviation "
            f"{health_value['score_std']} and feature coverage "
            f"{health_value['feature_coverage']}.",
            health,
        ),
    )
    performance_facts = by_category.get("performance", [])
    champion_facts = [
        fact
        for fact in performance_facts
        if fact.value.get("model_id") == production_value["model_id"]
    ]
    champion = tuple(
        _conclusion(
            "Champion performance for horizon "
            f"{fact.value['horizon']} reports Rank IC "
            f"{fact.value.get('rank_ic')} and alpha decay ratio "
            f"{fact.value.get('alpha_decay_ratio')}.",
            fact,
        )
        for fact in champion_facts
    ) or (
        _conclusion(
            "No mature Champion performance observations are available.",
            availability,
        ),
    )
    challenger_facts = [
        fact
        for fact in performance_facts
        if fact.value.get("model_id") != production_value["model_id"]
    ]
    challengers = tuple(
        _conclusion(
            "Observed model "
            f"{fact.value['model_id']} at horizon {fact.value['horizon']} "
            f"reports Rank IC {fact.value.get('rank_ic')}.",
            fact,
        )
        for fact in challenger_facts
    ) or (
        _conclusion(
            "No mature Challenger comparison observations are available.",
            availability,
        ),
    )
    alert_facts = by_category.get("alert", [])
    alert_section = tuple(
        _conclusion(
            f"Alert {fact.value['alert_type']} is {fact.value['severity']} "
            f"for {fact.value['metric_name']} at {fact.value['metric_value']} "
            f"with threshold {fact.value['threshold']}.",
            fact,
        )
        for fact in alert_facts
    ) or (_conclusion("No active monitoring alerts are reported.", availability),)
    candidate_facts = by_category.get("candidate", [])
    candidates = tuple(
        _conclusion(
            f"Candidate {fact.value['ts_code']} rank {fact.value['rank']} has signal "
            f"strength {fact.value['signal_strength']} and confidence "
            f"{fact.value['confidence']}.",
            fact,
        )
        for fact in candidate_facts
    ) or (_conclusion("No candidate details are available.", availability),)
    portfolio_facts = by_category.get("paper_portfolio", [])
    paper = tuple(
        _conclusion(
            f"Paper portfolio {fact.value['portfolio_id']} reports NAV "
            f"{fact.value['nav']}, drawdown {fact.value['drawdown']}, and holdings "
            f"count {fact.value['position_count']}.",
            fact,
        )
        for fact in portfolio_facts
    ) or (_conclusion("No paper portfolio summary is available.", availability),)
    risk = tuple(
        _conclusion(
            f"Risk monitoring records {fact.value['severity']} "
            f"{fact.value['alert_type']} evidence.",
            fact,
        )
        for fact in alert_facts
    ) or (_conclusion("No alert-based risk evidence is currently reported.", availability),)
    limitations = (
        _conclusion(
            "Performance availability is "
            f"{availability.value['performance_available']}; untrusted Markdown was not "
            "admitted to the research context.",
            availability,
        ),
    )
    sections = (
        overview + champion + challengers + alert_section + candidates + paper + risk + limitations
    )
    cited = tuple(sorted({fact_id for item in sections for fact_id in item.fact_ids}))
    return ResearchAgentSummary(
        market_model_overview=overview,
        champion_performance=champion,
        challenger_comparison=challengers,
        alert_interpretation=alert_section,
        candidate_explanations=candidates,
        paper_trading_status=paper,
        risk_summary=risk,
        data_limitations=limitations,
        source_fact_ids=cited,
    )


def _conclusion(text: str, fact: ResearchFact) -> ResearchConclusion:
    return ResearchConclusion(text=text, fact_ids=(fact.fact_id,))
