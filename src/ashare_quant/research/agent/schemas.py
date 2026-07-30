"""Typed contracts for the isolated LLM research agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ResearchFact(BaseModel):
    """One deterministic, citable fact supplied to the research agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: str
    category: str
    key: str
    value: Any


class ResearchConclusion(BaseModel):
    """One grounded narrative conclusion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    fact_ids: tuple[str, ...] = Field(min_length=1)


class ResearchAgentSummary(BaseModel):
    """Strict JSON-only response contract shared by LLM and fallback."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    market_model_overview: tuple[ResearchConclusion, ...] = Field(min_length=1)
    champion_performance: tuple[ResearchConclusion, ...] = Field(min_length=1)
    challenger_comparison: tuple[ResearchConclusion, ...] = Field(min_length=1)
    alert_interpretation: tuple[ResearchConclusion, ...] = Field(min_length=1)
    candidate_explanations: tuple[ResearchConclusion, ...] = Field(min_length=1)
    paper_trading_status: tuple[ResearchConclusion, ...] = Field(min_length=1)
    risk_summary: tuple[ResearchConclusion, ...] = Field(min_length=1)
    data_limitations: tuple[ResearchConclusion, ...] = Field(min_length=1)
    source_fact_ids: tuple[str, ...]


class ResearchContext(BaseModel):
    """Normalized immutable context; Markdown content is deliberately absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of: str
    production: dict[str, Any]
    model_health: dict[str, Any]
    performance: tuple[dict[str, Any], ...]
    alerts: tuple[dict[str, Any], ...]
    candidates: tuple[dict[str, Any], ...]
    paper_portfolios: tuple[dict[str, Any], ...]
    data_availability: dict[str, Any]
    fact_catalog: tuple[ResearchFact, ...]


@dataclass(frozen=True, slots=True)
class CollectedArtifacts:
    """Validated source payloads and physical lineage."""

    as_of: str
    payloads: dict[str, Any]
    source_hashes: dict[str, str]
    source_paths: dict[str, str]


@dataclass(frozen=True, slots=True)
class ResearchAgentResult:
    """Published research-agent artifact identity."""

    as_of: str
    output_dir: Path
    generation_mode: str
    run_id: str
    idempotent: bool = False


@dataclass(frozen=True, slots=True)
class ResearchAgentValidationResult:
    """Read-only validation/status result."""

    as_of: str
    valid: bool
    exists: bool
    generation_mode: str | None = None
    error: str | None = None
