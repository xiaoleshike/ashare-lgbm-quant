"""Exposure-scoped, performance-blind economic evidence review. No execution policy."""

from __future__ import annotations

import json
import math
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ashare_quant.backtest.continuous_source import payload_hash
from ashare_quant.data.economic_event_evidence import position_coverage, terminal_coverage
from ashare_quant.data.economic_event_sources import EconomicExposureSource, records
from ashare_quant.data.economic_events import AMOUNTS, DATES, amount, date_value, require
from ashare_quant.data.exceptions import DataValidationError

CONTRACT = "economic_event_review_v1"
DECISIONS = "economic_event_review_decisions_v1"
RULES = {
    "matching": "canonical_code_and_exact_ex_date_only",
    "context": "event_date_in_actual_holding_or_factor_observation_interval_no_tolerance",
    "grouping": "nullable_material_terms_and_report_period_v1",
    "conflict": "same_code_and_ex_date_or_report_period_disagreeing_material_terms",
    "share_check": "diagnostic_only_abs_1e-10_rel_1e-8_no_missing_component_inference",
    "priority": "P0:capital>=100000;P1:capital>=10000;P2:any_entry_gross>1;else:DUST_ONLY",
    "material": "entry_gross > 1; descriptive only",
    "ownership": "source_end_of_day_record_date_holdings",
    "execution_authorized": False,
}
MATERIAL = ["record_date", "ex_date", "pay_date", "div_listdate", *AMOUNTS]
DATE_FIELDS = ["end_date", *DATES]
GROUP_COLUMNS = [
    "candidate_group_id",
    "canonical_ts_code",
    "source_codes",
    "source_row_hashes",
    "end_date",
    *MATERIAL,
    "announcement_dates",
    "implementation_announcements",
    "provider_row_count",
    "revision_classification",
    "materiality_category",
    "issues",
    "share_consistency_diagnostic",
    "conflict_group_ids",
    "div_proc",
]
QUEUE_COLUMNS = [
    "review_event_id",
    "economic_boundary_id",
    "scope",
    "canonical_ts_code",
    "source_codes",
    "observed_adjustment_date",
    "affected_position_count",
    "affected_top_n",
    "capital_exposed",
    "capital_material_exposed",
    "material_position_count",
    "candidate_count",
    "exact_ex_date_candidate_count",
    "candidate_group_ids",
    "exact_candidate_group_ids",
    "materiality_category",
    "div_proc",
    *MATERIAL,
    "factor_before",
    "factor_after",
    "factor_ratio",
    "factor_diagnostic",
    "match_level",
    "triage_status",
    "review_priority",
]


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_event_id: str
    candidate_group_id: str | None
    decision: Literal[
        "APPROVE_AS_EVIDENCE",
        "REJECT_NOT_SAME_EVENT",
        "BLOCK_CONFLICT",
        "REQUIRE_OFFICIAL_DOCUMENT",
    ]
    reviewed_by: str = Field(min_length=1)
    review_reason: str = Field(min_length=1)
    review_timestamp: str | None = None
    source_row_hashes: list[str]


def decision_body(body: dict[str, Any], context: dict[str, Any]) -> list[dict[str, Any]]:
    require(
        body.get("schema") == DECISIONS and body.get("source") == context, "REVIEW_SOURCE_MISMATCH"
    )
    result = []
    keys = set()
    for raw in body["decisions"]:
        row = ReviewDecision.model_validate(raw).model_dump(exclude={"review_timestamp"})
        require(
            bool(row["reviewed_by"].strip()) and bool(row["review_reason"].strip()), "EMPTY_REVIEW"
        )
        key = (row["review_event_id"], row["candidate_group_id"])
        require(key not in keys, "DUPLICATE_REVIEW_DECISION")
        keys.add(key)
        require(
            len(set(row["source_row_hashes"])) == len(row["source_row_hashes"]),
            "DUPLICATE_ROW_REFERENCE",
        )
        row["source_row_hashes"] = sorted(row["source_row_hashes"])
        result.append(row)
    return sorted(result, key=payload_hash)


def _parse(item: dict[str, Any]) -> dict[str, Any]:
    body = json.loads(item["raw_row_json"])
    result: dict[str, Any] = {"canonical_ts_code": item["ts_code"]}
    issues = []
    for field in [*DATE_FIELDS, *AMOUNTS]:
        try:
            result[field] = (
                date_value(body.get(field)) if field in DATE_FIELDS else amount(body.get(field))
            )
        except DataValidationError:
            result[field] = None
            issues.append("INVALID_" + field)
    if any(result[k] is None for k in AMOUNTS):
        issues.append("MISSING_ECONOMIC_COMPONENTS")
    if not result["record_date"] or not result["ex_date"]:
        issues.append("MISSING_RECORD_OR_EX_DATE")
    if result["record_date"] and result["ex_date"] and result["record_date"] >= result["ex_date"]:
        issues.append("DATE_ORDER_CONFLICT")
    for k in ("pay_date", "div_listdate"):
        if result[k] and result["ex_date"] and result[k] < result["ex_date"]:
            issues.append("DATE_ORDER_CONFLICT")
    cash, taxed = result["cash_div_tax"], result["cash_div"]
    if cash is not None and taxed is not None and taxed > cash + 1e-10:
        issues.append("CASH_FIELD_CONFLICT")
    cash_nonzero = any((result[k] or 0) > 0 for k in ("cash_div", "cash_div_tax"))
    shares = any((result[k] or 0) > 0 for k in ("stk_div", "stk_bo_rate", "stk_co_rate"))
    category = (
        "INCOMPLETE_ECONOMICS"
        if any(result[k] is None for k in AMOUNTS)
        else "CASH_AND_SHARES"
        if cash_nonzero and shares
        else "CASH_ONLY"
        if cash_nonzero
        else "SHARES_ONLY"
        if shares
        else "ZERO_ECONOMICS"
    )
    share_check = "NOT_EVALUATED"
    if all(result[k] is not None for k in ("stk_div", "stk_bo_rate", "stk_co_rate")):
        share_check = (
            "CONSISTENT"
            if math.isclose(
                result["stk_div"],
                result["stk_bo_rate"] + result["stk_co_rate"],
                abs_tol=1e-10,
                rel_tol=1e-8,
            )
            else "REVIEW_REQUIRED"
        )
    result.update(
        materiality_category=category,
        issues=json.dumps(sorted(set(issues))),
        share_consistency_diagnostic=share_check,
    )
    return result


def revision_groups(raw: pd.DataFrame) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for item in records(raw):
        if json.loads(item["raw_row_json"]).get("div_proc") != "实施":
            continue
        parsed = _parse(item)
        # Invalid observations cannot collapse into a valid null through normalization.
        identity = {k: parsed[k] for k in ["canonical_ts_code", "end_date", *MATERIAL, "issues"]}
        if "INVALID_" in parsed["issues"]:
            identity["invalid_raw_hash"] = item["provider_raw_row_hash"]
        grouped.setdefault(payload_hash(identity), []).append((item, parsed))
    groups = []
    for digest, members in sorted(grouped.items()):
        parsed = members[0][1]
        hashes = sorted({m[0]["provider_raw_row_hash"] for m in members})
        groups.append(
            {
                "candidate_group_id": "economic_candidate_" + digest[:24],
                **{
                    k: parsed[k]
                    for k in [
                        "canonical_ts_code",
                        "end_date",
                        *MATERIAL,
                        "materiality_category",
                        "issues",
                        "share_consistency_diagnostic",
                    ]
                },
                "source_codes": json.dumps(sorted({m[0]["source_ts_code"] for m in members})),
                "source_row_hashes": json.dumps(hashes),
                "announcement_dates": json.dumps(
                    sorted({m[1]["ann_date"] for m in members if m[1]["ann_date"]})
                ),
                "implementation_announcements": json.dumps(
                    sorted({m[1]["imp_ann_date"] for m in members if m[1]["imp_ann_date"]})
                ),
                "provider_row_count": len(members),
                "div_proc": "IMPLEMENTED",
                "revision_classification": "SEMANTICALLY_IDENTICAL_PROVIDER_REVISIONS"
                if len(hashes) > 1
                else "SINGLE_IMPLEMENTED_ROW",
                "conflict_group_ids": "[]",
            }
        )
    for group in groups:
        conflicts = [
            other["candidate_group_id"]
            for other in groups
            if other["candidate_group_id"] != group["candidate_group_id"]
            and other["canonical_ts_code"] == group["canonical_ts_code"]
            and any(group[k] is not None and group[k] == other[k] for k in ("ex_date", "end_date"))
            and any(group[k] != other[k] for k in MATERIAL)
        ]
        group["conflict_group_ids"] = json.dumps(sorted(conflicts))
        if conflicts:
            group["revision_classification"] = "CONFLICTING_IMPLEMENTED_PROVIDER_ROWS"
    return groups


def _related(group: dict[str, Any], exposures: pd.DataFrame) -> bool:
    dates = [group[k] for k in ("ex_date", "record_date", "pay_date", "div_listdate") if group[k]]
    if not dates:
        return True  # Undated terms for the same security cannot be proven irrelevant.
    return any(
        e.entry_date <= d <= e.resolution_date for e in exposures.itertuples() for d in dates
    )


def build_triage(
    source: EconomicExposureSource, raw: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    groups = revision_groups(raw)
    adjustments = source.exposures[source.exposures.exposure_type.eq("ADJ_FACTOR_CHANGE")]
    queue = []
    linked: set[str] = set()
    for boundary, exposure in adjustments.groupby("economic_boundary_id", sort=True):
        first = exposure.iloc[0]
        code, dt = first.ts_code, first.candidate_event_date
        same = [g for g in groups if g["canonical_ts_code"] == code]
        exact = [g for g in same if g["ex_date"] == dt]
        context = [g for g in same if _related(g, exposure)]
        candidates = {g["candidate_group_id"]: g for g in [*exact, *context]}
        for g in list(candidates.values()):
            for sibling in groups:
                if sibling["candidate_group_id"] in json.loads(g["conflict_group_ids"]):
                    candidates[sibling["candidate_group_id"]] = sibling
        linked.update(candidates)
        queue.append(
            _queue_row(
                str(boundary),
                "AUDIT_ADJUSTMENT",
                code,
                dt,
                exposure,
                list(candidates.values()),
                exact,
                float(first.adj_factor_before),
                float(first.adj_factor_after),
            )
        )
    # Reverse coverage must not disappear merely because the audit saw no factor movement.
    observed = {(q["canonical_ts_code"], q["observed_adjustment_date"]) for q in queue}
    supplemental: dict[tuple[str, str], list[dict[str, Any]]] = {}
    holdings = set(
        zip(
            source.holdings.top_n,
            source.holdings.position_id,
            source.holdings.trade_date,
            strict=True,
        )
    )
    for g in groups:
        ex = g["ex_date"]
        if not ex or ex > source.identity["governed_execution_cutoff"]:
            continue
        key = (g["canonical_ts_code"], ex)
        if key in observed:
            continue
        positions = source.positions[source.positions.ts_code.eq(g["canonical_ts_code"])]
        relevant = positions.apply(
            lambda p, ex=ex, g=g: (
                p.entry_date <= ex <= p.resolution_date
                or (p.top_n, p.position_id, g["record_date"]) in holdings
            ),
            axis=1,
        )
        if not positions.empty and relevant.any() and g["materiality_category"] != "ZERO_ECONOMICS":
            supplemental.setdefault(key, []).append(g)
    for (code, dt), matches in sorted(supplemental.items()):
        positions = source.positions[source.positions.ts_code.eq(code)]
        selected = positions[
            positions.apply(
                lambda p, dt=dt, matches=matches: (
                    p.entry_date <= dt <= p.resolution_date
                    or any((p.top_n, p.position_id, g["record_date"]) in holdings for g in matches)
                ),
                axis=1,
            )
        ]
        linked.update(g["candidate_group_id"] for g in matches)
        queue.append(
            _queue_row(
                payload_hash([code, dt, "REVERSE_COVERAGE"]),
                "SUPPLEMENTAL_ENTITLEMENT",
                code,
                dt,
                selected,
                matches,
                matches,
                None,
                None,
            )
        )
    undated: dict[str, list[dict[str, Any]]] = {}
    for g in groups:
        if g["ex_date"] is None:
            positions = source.positions[source.positions.ts_code.eq(g["canonical_ts_code"])]
            if not positions.empty and _related(g, positions):
                undated.setdefault(g["canonical_ts_code"], []).append(g)
    for code, matches in sorted(undated.items()):
        linked.update(g["candidate_group_id"] for g in matches)
        positions = source.positions[source.positions.ts_code.eq(code)]
        queue.append(
            _queue_row(
                payload_hash([code, "UNDATED_ENTITLEMENT_REVIEW"]),
                "UNDATED_ENTITLEMENT_REVIEW",
                code,
                None,
                positions,
                matches,
                [],
                None,
                None,
            )
        )
    relevant_groups = [g for g in groups if g["candidate_group_id"] in linked]
    hashes = {h for g in relevant_groups for h in json.loads(g["source_row_hashes"])}
    relevant_raw = raw[raw.provider_raw_row_hash.isin(hashes)].sort_values(
        ["provider_raw_row_hash", "request_id"]
    )
    implemented = sum(json.loads(r).get("div_proc") == "实施" for r in raw.raw_row_json)
    tables = {
        "relevant_provider_rows.parquet": relevant_raw.reset_index(drop=True),
        "provider_revision_groups.parquet": pd.DataFrame(relevant_groups, columns=GROUP_COLUMNS),
        "economic_event_review_queue.parquet": pd.DataFrame(queue, columns=QUEUE_COLUMNS)
        .sort_values("review_event_id")
        .reset_index(drop=True),
    }
    summary: dict[str, Any] = {
        "provider_raw_rows": len(raw),
        "implemented_candidate_rows": implemented,
        "relevant_candidate_rows": len(relevant_raw),
        "irrelevant_historical_rows": implemented - len(relevant_raw),
        "unique_relevant_groups": len(relevant_groups),
        "adjustment_boundaries_total": int(adjustments.economic_boundary_id.nunique()),
        "supplemental_entitlement_boundaries": len(supplemental),
        "undated_entitlement_security_queues": len(undated),
        "review_queue_items": len(queue),
    }
    return tables, summary


def _queue_row(
    boundary: str,
    scope: str,
    code: str,
    dt: str | None,
    exposure: pd.DataFrame,
    candidates: list[dict[str, Any]],
    exact: list[dict[str, Any]],
    before: float | None,
    after: float | None,
) -> dict[str, Any]:
    conflict = len(exact) > 1 or any(json.loads(g["conflict_group_ids"]) for g in exact)
    chosen = exact[0] if len(exact) == 1 else None
    status = (
        "MULTIPLE_CONFLICTING_EXACT_EVENTS"
        if conflict
        else "ONLY_NEARBY_EVENT"
        if not exact and candidates
        else "NO_EXACT_EVENT"
        if not exact
        else "INCOMPLETE_PROVIDER_EVENT"
        if chosen and json.loads(chosen["issues"])
        else "UNSUPPORTED_EVENT_TYPE"
        if chosen and chosen["materiality_category"] == "ZERO_ECONOMICS"
        else "IDENTICAL_REVISIONS_EXACT_EVENT"
        if chosen
        and chosen["revision_classification"] == "SEMANTICALLY_IDENTICAL_PROVIDER_REVISIONS"
        else "UNIQUE_EXACT_PROVIDER_EVENT"
    )
    positions = exposure.drop_duplicates(["top_n", "position_id"])
    capital = float(positions.entry_gross.sum())
    material = positions[positions.entry_gross.gt(1)]
    return {
        "review_event_id": "economic_review_" + payload_hash([boundary, scope])[:24],
        "economic_boundary_id": boundary,
        "scope": scope,
        "canonical_ts_code": code,
        "source_codes": json.dumps(
            sorted({c for g in candidates for c in json.loads(g["source_codes"])})
        ),
        "observed_adjustment_date": dt,
        "affected_position_count": len(positions),
        "affected_top_n": json.dumps(sorted(int(n) for n in positions.top_n.unique())),
        "capital_exposed": capital,
        "capital_material_exposed": float(material.entry_gross.sum()),
        "material_position_count": len(material),
        "candidate_count": len(candidates),
        "exact_ex_date_candidate_count": len(exact),
        "candidate_group_ids": json.dumps(sorted(g["candidate_group_id"] for g in candidates)),
        "exact_candidate_group_ids": json.dumps(sorted(g["candidate_group_id"] for g in exact)),
        **{
            k: chosen[k] if chosen else None
            for k in ["materiality_category", "div_proc", *MATERIAL]
        },
        "factor_before": before,
        "factor_after": after,
        "factor_ratio": after / before if before and after else None,
        "factor_diagnostic": "FACTOR_DIRECTION_CONSISTENT"
        if before
        and after
        and after > before
        and chosen
        and chosen["materiality_category"] in {"CASH_ONLY", "SHARES_ONLY", "CASH_AND_SHARES"}
        else "FACTOR_DIRECTION_UNEXPLAINED"
        if before
        else "NOT_EVALUATED",
        "match_level": "EXACT_EX_DATE"
        if exact
        else "DATE_NEARBY_REVIEW_REQUIRED"
        if candidates
        else "NO_DATE_MATCH",
        "triage_status": status,
        "review_priority": "P0_MATERIAL"
        if capital >= 100000
        else "P1_MEDIUM"
        if capital >= 10000
        else "P2_LOW"
        if len(material)
        else "DUST_ONLY",
    }


def apply_decisions(
    source: EconomicExposureSource,
    tables: dict[str, pd.DataFrame],
    decisions: list[dict[str, Any]],
    terminal: list[dict[str, Any]],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    queue = {
        r["review_event_id"]: r for r in records(tables["economic_event_review_queue.parquet"])
    }
    groups = {
        r["candidate_group_id"]: r for r in records(tables["provider_revision_groups.parquet"])
    }
    approved: dict[str, dict[str, Any]] = {}
    approved_events = set()
    for d in decisions:
        require(d["review_event_id"] in queue, "UNKNOWN_REVIEW_EVENT")
        q = queue[d["review_event_id"]]
        gid = d["candidate_group_id"]
        if gid is None:
            require(
                not d["source_row_hashes"]
                and d["decision"] in {"REQUIRE_OFFICIAL_DOCUMENT", "BLOCK_CONFLICT"},
                "REVIEW_CANDIDATE_REQUIRED",
            )
            continue
        require(gid in json.loads(q["candidate_group_ids"]), "WRONG_REVIEW_CANDIDATE")
        g = groups[gid]
        require(
            sorted(d["source_row_hashes"]) == json.loads(g["source_row_hashes"]),
            "REVIEW_ROW_HASH_MISMATCH",
        )
        if d["decision"] != "APPROVE_AS_EVIDENCE":
            continue
        require(
            q["triage_status"] in {"UNIQUE_EXACT_PROVIDER_EVENT", "IDENTICAL_REVISIONS_EXACT_EVENT"}
            and gid in json.loads(q["exact_candidate_group_ids"])
            and g["ex_date"] <= source.identity["governed_execution_cutoff"],
            "APPROVAL_NOT_ELIGIBLE",
        )
        approved_events.add(d["review_event_id"])
        row = {
            **g,
            "review_decision_hash": payload_hash(d),
            "evidence_status": "REVIEWED_PROVIDER_EVIDENCE",
            "cash_tax_semantics": "UNRESOLVED_EXECUTION_POLICY",
            "execution_authorized": False,
        }
        approved[gid] = row
    recon = []
    for event, q in queue.items():
        choices = [d for d in decisions if d["review_event_id"] == event]
        require(
            not (
                event in approved_events
                and any(
                    d["decision"] in {"BLOCK_CONFLICT", "REQUIRE_OFFICIAL_DOCUMENT"}
                    for d in choices
                )
            ),
            "CONTRADICTORY_REVIEW_DECISIONS",
        )
        recon.append(
            {
                **q,
                "review_status": "APPROVED_AS_EVIDENCE"
                if event in approved_events
                else "UNRESOLVED",
                "review_decisions": json.dumps(choices, sort_keys=True),
            }
        )
    reviewed = pd.DataFrame(
        list(approved.values()),
        columns=[
            *GROUP_COLUMNS,
            "review_decision_hash",
            "evidence_status",
            "cash_tax_semantics",
            "execution_authorized",
        ],
    )
    # Adapt approved evidence to the existing EOD ownership service; no cash calculation.
    normalized = []
    for g in approved.values():
        normalized.append(
            {
                "event_id": g["candidate_group_id"],
                "ts_code": g["canonical_ts_code"],
                **{v: g[k] for k, v in AMOUNTS.items()},
                "record_date": g["record_date"],
                "ex_date": g["ex_date"],
                "component_type": g["materiality_category"],
                "within_execution_support": g["ex_date"]
                <= source.identity["governed_execution_cutoff"],
                "evidence_status": "REVIEWED_PROVIDER_EVIDENCE",
            }
        )
    coverage = position_coverage(source, pd.DataFrame(normalized))
    for field, group_field in (("pay_date", "pay_date"), ("div_listdate", "div_listdate")):
        coverage[field] = coverage.event_id.map(
            {gid: g[group_field] for gid, g in approved.items()}
        )
    terminal_table = terminal_coverage(source, terminal)
    reconciled = pd.DataFrame(recon, columns=[*QUEUE_COLUMNS, "review_status", "review_decisions"])
    unresolved_rows = [
        {
            "finding_type": q.scope,
            "finding_id": q.review_event_id,
            "ts_code": q.canonical_ts_code,
            "reason": q.triage_status,
        }
        for q in reconciled[reconciled.review_status.eq("UNRESOLVED")].itertuples()
    ]
    unresolved_rows.extend(
        {
            "finding_type": "TERMINAL",
            "finding_id": t.economic_boundary_id,
            "ts_code": t.ts_code,
            "reason": t.coverage,
        }
        for t in terminal_table[~terminal_table.coverage.eq("EVIDENCE_COMPLETE")].itertuples()
    )
    unresolved_rows.extend(
        {
            "finding_type": "ENTITLEMENT",
            "finding_id": payload_hash([p.event_id, p.top_n, p.position_id]),
            "ts_code": p.ts_code,
            "reason": p.eligibility,
        }
        for p in coverage[
            ~coverage.eligibility.isin(["HELD_THROUGH_RECORD_DATE", "NOT_HELD_ON_RECORD_DATE"])
        ].itertuples()
    )
    unresolved_rows.extend(
        {
            "finding_type": "SHARE_COMPONENT_DIAGNOSTIC",
            "finding_id": gid,
            "ts_code": g["canonical_ts_code"],
            "reason": "SHARE_FIELDS_REQUIRE_AUTHORITATIVE_REVIEW",
        }
        for gid, g in approved.items()
        if g["share_consistency_diagnostic"] == "REVIEW_REQUIRED"
    )
    unresolved = pd.DataFrame(
        unresolved_rows, columns=["finding_type", "finding_id", "ts_code", "reason"]
    )
    audit = reconciled[reconciled.scope.eq("AUDIT_ADJUSTMENT")]
    total_capital = float(audit.capital_exposed.sum())
    reviewed_capital = float(
        audit.loc[audit.review_status.eq("APPROVED_AS_EVIDENCE"), "capital_exposed"].sum()
    )
    summary: dict[str, Any] = {
        "adjustment_exact_provider_match": int(audit.exact_ex_date_candidate_count.gt(0).sum()),
        "adjustment_no_match": int(audit.exact_ex_date_candidate_count.eq(0).sum()),
        "adjustment_conflict": int(
            audit.triage_status.eq("MULTIPLE_CONFLICTING_EXACT_EVENTS").sum()
        ),
        "adjustment_reviewed": int(audit.review_status.eq("APPROVED_AS_EVIDENCE").sum()),
        "adjustment_unresolved": int(audit.review_status.eq("UNRESOLVED").sum()),
        "reviewed_catalog_events": len(reviewed),
        "capital_exposure_reviewed_fraction": reviewed_capital / total_capital
        if total_capital
        else None,
        "terminal_total": len(terminal_table),
        "terminal_reviewed_complete": int(terminal_table.coverage.eq("EVIDENCE_COMPLETE").sum()),
        "terminal_unknown": int(terminal_table.coverage.eq("ECONOMIC_RECOVERY_UNKNOWN").sum()),
        "execution_authorized": False,
        "economic_event_evidence_status": "BLOCKED"
        if (not terminal_table.coverage.eq("EVIDENCE_COMPLETE").all() or len(unresolved))
        else "COMPLETE",
    }
    material = audit[audit.material_position_count.gt(0)]
    summary["material_only"] = {
        "boundaries": len(material),
        "reviewed": int(material.review_status.eq("APPROVED_AS_EVIDENCE").sum()),
        "unresolved": int(material.review_status.eq("UNRESOLVED").sum()),
        "capital_exposure_reviewed_fraction": float(
            material.loc[
                material.review_status.eq("APPROVED_AS_EVIDENCE"), "capital_material_exposed"
            ].sum()
        )
        / float(material.capital_material_exposed.sum())
        if material.capital_material_exposed.sum()
        else None,
    }
    summary["by_top_n"] = {
        str(n): {
            "boundaries": sum(n in json.loads(q["affected_top_n"]) for q in recon),
            "reviewed": sum(
                q["review_status"] == "APPROVED_AS_EVIDENCE"
                and n in json.loads(q["affected_top_n"])
                for q in recon
            ),
        }
        for n in sorted(int(x) for x in source.exposures.top_n.unique())
    }
    return {
        "reviewed_corporate_actions.parquet": reviewed,
        "adjustment_reconciliation.parquet": reconciled,
        "position_entitlement_review.parquet": coverage,
        "terminal_coverage.parquet": terminal_table,
        "unresolved_events.parquet": unresolved,
    }, summary
