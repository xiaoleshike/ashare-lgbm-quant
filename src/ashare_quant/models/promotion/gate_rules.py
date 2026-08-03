"""Versioned policy and evidence-level rules for promotion eligibility."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ashare_quant.models.promotion.gate_schemas import GateCheck
from ashare_quant.models.shadow.storage import canonical_payload_hash


class PromotionGatePolicy(BaseModel):
    """Conservative policy used only to decide eligibility for human review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    minimum_mature_sessions: int = Field(default=20, gt=0)
    minimum_paper_sessions: int = Field(default=20, gt=0)
    high_turnover_threshold: float = Field(default=1.0, gt=0)
    expected_execution_rule: str = "next_open"

    @property
    def policy_hash(self) -> str:
        """Return the deterministic policy identity."""

        return canonical_payload_hash(self.model_dump(mode="json"))


def performance_checks(
    *,
    manifest: dict[str, object],
    metrics: dict[str, object],
    candidate_model_id: str,
    evidence_hash: str,
    policy: PromotionGatePolicy,
) -> tuple[GateCheck, ...]:
    """Evaluate prospective maturity and observation coverage."""

    checks: list[GateCheck] = []
    access_policy = manifest.get("access_policy")
    if access_policy != "prospective_production":
        checks.append(
            _check(
                "prospective_observation_source",
                "FAIL",
                f"performance access_policy is {access_policy!r}, not prospective_production",
                evidence_hash,
            )
        )
        return tuple(checks)
    checks.append(
        _check(
            "prospective_observation_source",
            "PASS",
            "performance evidence is prospective_production",
            evidence_hash,
        )
    )
    available_rows = _integer(manifest.get("available_rows", metrics.get("available_rows")))
    checks.append(
        _check(
            "prospective_observation_rows",
            "PASS" if available_rows > 0 else "FAIL",
            f"prospective available_rows={available_rows}",
            evidence_hash,
        )
    )
    model_ids = manifest.get("model_ids")
    model_present = isinstance(model_ids, list) and candidate_model_id in model_ids
    checks.append(
        _check(
            "prospective_candidate_lineage",
            "PASS" if model_present else "FAIL",
            "candidate is present in prospective observation lineage"
            if model_present
            else "candidate is absent from prospective observation lineage",
            evidence_hash,
        )
    )
    mature_sessions = _mature_sessions(metrics, candidate_model_id)
    checks.append(
        _check(
            "minimum_mature_sessions",
            "PASS" if mature_sessions >= policy.minimum_mature_sessions else "FAIL",
            f"mature_sessions={mature_sessions}, required={policy.minimum_mature_sessions}",
            evidence_hash,
        )
    )
    return tuple(checks)


def review_checks(
    *,
    challenger_manifest: dict[str, object],
    executable_manifest: dict[str, object],
    monitoring_summary: dict[str, object],
    evidence_hash: str,
    policy: PromotionGatePolicy,
) -> tuple[GateCheck, ...]:
    """Evaluate conditions that require human interpretation rather than hard rejection."""

    checks: list[GateCheck] = []
    gate = challenger_manifest.get("promotion_gate")
    raw_criteria = gate.get("criteria") if isinstance(gate, dict) else None
    criteria = raw_criteria if isinstance(raw_criteria, list) else []
    criterion_map = {
        str(item.get("name")): bool(item.get("passed"))
        for item in criteria
        if isinstance(item, dict)
    }
    ic_improved = criterion_map.get("minimum_rank_ic_delta") is True
    top_n_degraded = criterion_map.get("minimum_top10_return_delta") is False
    checks.append(
        _check(
            "ic_top_n_consistency",
            "WARNING" if ic_improved and top_n_degraded else "PASS",
            "Rank IC improved while Top-N return degraded"
            if ic_improved and top_n_degraded
            else "no IC/Top-N divergence is declared by evaluation evidence",
            evidence_hash,
        )
    )
    regime_unstable = (
        any("regime" in name and not passed for name, passed in criterion_map.items())
        or challenger_manifest.get("regime_stable") is False
    )
    checks.append(
        _check(
            "regime_stability",
            "WARNING" if regime_unstable else "PASS",
            "evaluation evidence indicates regime instability"
            if regime_unstable
            else "evaluation evidence does not declare regime instability",
            evidence_hash,
        )
    )
    turnover = executable_manifest.get("average_turnover")
    high_turnover = isinstance(turnover, (int, float)) and float(turnover) > (
        policy.high_turnover_threshold
    )
    checks.append(
        _check(
            "turnover_review",
            "WARNING" if high_turnover else "PASS",
            f"average_turnover={turnover} exceeds {policy.high_turnover_threshold}"
            if high_turnover
            else "no excessive turnover is declared by immutable evidence",
            evidence_hash,
        )
    )
    paper_sessions = _paper_sessions(monitoring_summary)
    short_paper = paper_sessions is not None and paper_sessions < policy.minimum_paper_sessions
    checks.append(
        _check(
            "paper_trading_history",
            "WARNING" if short_paper else "PASS",
            f"paper_sessions={paper_sessions}, required={policy.minimum_paper_sessions}"
            if short_paper
            else "paper-trading evidence does not declare a short observation period",
            evidence_hash,
        )
    )
    return tuple(checks)


def _mature_sessions(metrics: dict[str, object], model_id: str) -> int:
    explicit = metrics.get("mature_sessions")
    if isinstance(explicit, int) and not isinstance(explicit, bool):
        return explicit
    daily = metrics.get("daily")
    if not isinstance(daily, list):
        return 0
    dates = {
        str(item.get("signal_date") or item.get("trade_date") or item.get("date"))
        for item in daily
        if isinstance(item, dict) and item.get("model_id") == model_id
    }
    return len(dates - {"None"})


def _paper_sessions(summary: dict[str, object]) -> int | None:
    value = summary.get("paper_trading_sessions")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _integer(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _check(name: str, status: str, message: str, evidence_hash: str) -> GateCheck:
    from typing import cast

    from ashare_quant.models.promotion.gate_schemas import GateCheckStatus

    return GateCheck(
        name=name,
        status=cast(GateCheckStatus, status),
        message=message,
        evidence_hash=evidence_hash,
    )
