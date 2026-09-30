"""Immutable economic evidence compilation and root-to-leaf business validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ashare_quant.backtest.continuous_source import file_hash, payload_hash, read_json
from ashare_quant.data.economic_event_sources import (
    EconomicExposureSource,
    compatible_implementation,
    content_hash,
    implementation_hash,
    load_exposure_source,
    publish,
    records,
    safe_file,
    validate_envelope,
    validate_provider_source,
)
from ashare_quant.data.economic_events import (
    AMOUNTS,
    CONTRACT,
    NORMALIZATION,
    TERMINAL_CONTRACT,
    TerminalEconomicRecord,
    normalize_dividends,
    require,
)

RECON_COLUMNS = [
    "exposure_id",
    "economic_boundary_id",
    "position_id",
    "top_n",
    "ts_code",
    "candidate_event_date",
    "event_ids",
    "classification",
    "reason",
]
COVERAGE_COLUMNS = [
    "event_id",
    "position_id",
    "top_n",
    "ts_code",
    "record_date",
    "ex_date",
    "entry_date",
    "resolution_date",
    "eligibility",
    "observed_adjustment",
    "evidence_status",
    "entry_gross",
]
TERMINAL_COLUMNS = [
    "economic_boundary_id",
    "ts_code",
    "terminal_date",
    "exposure_count",
    "evidence_record_ids",
    "coverage",
    "economic_resolution_type",
]


def publish_terminal_evidence(review: Path, documents_root: Path, output_root: Path) -> Path:
    """Append a reviewed package. Original documents are copied and hash-checked."""
    body = read_json(review)
    require(body.get("schema") == TERMINAL_CONTRACT, "TERMINAL_SCHEMA_INVALID")
    parsed = [TerminalEconomicRecord.model_validate(r) for r in body["records"]]
    documents: dict[str, bytes] = {}
    rows = []
    for record in parsed:
        path = safe_file(documents_root, record.document_file)
        require(file_hash(path) == record.document_sha256, "TERMINAL_DOCUMENT_HASH_MISMATCH")
        name = "documents/" + record.document_sha256
        documents[name] = path.read_bytes()
        row = record.model_dump()
        row["document_file"] = name
        rows.append(row)
    rows.sort(key=payload_hash)
    require(
        bool(rows) and len({payload_hash(r) for r in rows}) == len(rows),
        "TERMINAL_RECORD_SET_INVALID",
    )
    logical = {
        "contract": TERMINAL_CONTRACT,
        "records_hash": payload_hash(rows),
        "evidence_use": "POST_HOC_ACCOUNTING_EVIDENCE",
    }
    return publish(
        output_root,
        "terminal_economic_evidence",
        logical,
        {},
        {},
        {"records.json": {"schema": TERMINAL_CONTRACT, "records": rows}},
        validate_terminal_evidence,
        documents,
    )


def validate_terminal_evidence(path: Path) -> list[dict[str, Any]]:
    m = validate_envelope(path, "terminal_economic_evidence")
    data = read_json(path / "records.json")
    require(data.get("schema") == TERMINAL_CONTRACT, "TERMINAL_SCHEMA_INVALID")
    rows = [TerminalEconomicRecord.model_validate(r).model_dump() for r in data["records"]]
    require(
        m["logical_identity"]
        == {
            "contract": TERMINAL_CONTRACT,
            "records_hash": payload_hash(rows),
            "evidence_use": "POST_HOC_ACCOUNTING_EVIDENCE",
        },
        "TERMINAL_IDENTITY_MISMATCH",
    )
    require(bool(rows) and rows == sorted(rows, key=payload_hash), "TERMINAL_RECORD_SET_INVALID")
    for row in rows:
        require(
            file_hash(safe_file(path, row["document_file"])) == row["document_sha256"],
            "TERMINAL_DOCUMENT_HASH_MISMATCH",
        )
    return rows


def reconcile_adjustments(source: EconomicExposureSource, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for e in source.exposures[source.exposures.exposure_type.eq("ADJ_FACTOR_CHANGE")].itertuples():
        same = events[events.ts_code.eq(e.ts_code)]
        candidates = same[same.ex_date.eq(e.candidate_event_date)]
        state, reason = "NO_EVENT_FOUND", "NO_REVIEWED_EX_DATE_EVENT"
        if not candidates.empty:
            states = set(candidates.evidence_status)
            if "MULTIPLE_CONFLICTING_EVENTS" in states:
                state, reason = "MULTIPLE_CONFLICTING_EVENTS", "IMPLEMENTED_ROWS_DISAGREE"
            elif "UNSUPPORTED_OTHER_EVENT" in set(candidates.component_type):
                state, reason = "FACTOR_CHANGE_WITH_UNSUPPORTED_EVENT", "UNSUPPORTED_ECONOMICS"
            elif states == {"REVIEWED_PROVIDER_EVIDENCE"}:
                state = (
                    "MATCHED_COMPOSITE_EVENT"
                    if "COMPOSITE_DISTRIBUTION" in set(candidates.component_type)
                    else "MATCHED_SINGLE_EVENT"
                )
                reason = "EXACT_CODE_EX_DATE_REVIEWED"
            else:
                reason = ",".join(sorted(states))
        elif same.ex_date.isna().any() or (
            e.previous_observation
            and any(
                str(e.previous_observation) < str(dt) < str(e.candidate_event_date)
                for dt in same.ex_date.dropna()
            )
        ):
            state, reason = "EVENT_DATE_CONFLICT", "MISSING_EX_DATE_OR_FACTOR_OBSERVATION_GAP"
        rows.append(
            {
                "exposure_id": e.exposure_id,
                "economic_boundary_id": e.economic_boundary_id,
                "position_id": e.position_id,
                "top_n": e.top_n,
                "ts_code": e.ts_code,
                "candidate_event_date": e.candidate_event_date,
                "event_ids": json.dumps(sorted(candidates.event_id)),
                "classification": state,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=RECON_COLUMNS)


def position_coverage(source: EconomicExposureSource, events: pd.DataFrame) -> pd.DataFrame:
    """Record-date EOD ownership, including sold-before-payment entitlements."""
    keys = set(
        zip(
            source.holdings.top_n,
            source.holdings.position_id,
            source.holdings.trade_date,
            strict=True,
        )
    )
    sessions = set(source.calendar)
    adjustment_keys = set(
        zip(
            source.exposures.top_n,
            source.exposures.position_id,
            source.exposures.candidate_event_date,
            strict=True,
        )
    )
    rows = []
    for event in records(events):
        ex, record = event["ex_date"], event["record_date"]
        if not event["within_execution_support"]:
            continue
        nonzero = any((event[field] or 0) > 0 for field in AMOUNTS.values())
        if not nonzero and event["component_type"] != "UNSUPPORTED_OTHER_EVENT":
            continue
        for p in source.positions[source.positions.ts_code.eq(event["ts_code"])].itertuples():
            held_record = (p.top_n, p.position_id, record) in keys
            if not (p.entry_date <= ex <= p.resolution_date or held_record):
                continue
            if not record:
                eligibility = "AMBIGUOUS"
            elif record < source.calendar[0] or record > source.calendar[-1]:
                eligibility = "INSUFFICIENT_POSITION_HISTORY"
            elif record not in sessions:
                eligibility = "RECORD_DATE_NOT_IN_EXECUTION_CALENDAR"
            elif held_record:
                eligibility = "HELD_THROUGH_RECORD_DATE"
            elif p.entry_date <= record < p.resolution_date:
                eligibility = "INSUFFICIENT_POSITION_HISTORY"
            else:
                eligibility = "NOT_HELD_ON_RECORD_DATE"
            rows.append(
                {
                    "event_id": event["event_id"],
                    "position_id": p.position_id,
                    "top_n": p.top_n,
                    "ts_code": p.ts_code,
                    "record_date": record,
                    "ex_date": ex,
                    "entry_date": p.entry_date,
                    "resolution_date": p.resolution_date,
                    "eligibility": eligibility,
                    "observed_adjustment": (p.top_n, p.position_id, ex) in adjustment_keys,
                    "evidence_status": event["evidence_status"],
                    "entry_gross": float(cast(Any, p.entry_gross)),
                }
            )
    return (
        pd.DataFrame(rows, columns=COVERAGE_COLUMNS)
        .sort_values(["event_id", "top_n", "position_id"])
        .reset_index(drop=True)
    )


def terminal_coverage(
    source: EconomicExposureSource, terminal: list[dict[str, Any]]
) -> pd.DataFrame:
    rows = []
    exposures = source.exposures[source.exposures.exposure_type.eq("TERMINAL_EVENT")]
    for boundary, group in exposures.groupby("economic_boundary_id", sort=True):
        code, dt = group.iloc[0].ts_code, group.iloc[0].candidate_event_date
        matches = [r for r in terminal if r["ts_code"] == code and r["terminal_date"] == dt]
        economics = [
            "economic_resolution_type",
            "cash_per_share",
            "successor_security",
            "share_conversion_ratio",
            "receivable_terms",
            "effective_date",
            "settlement_date",
        ]
        signatures = {payload_hash({k: r[k] for k in economics}) for r in matches}
        state, treatment = "ECONOMIC_RECOVERY_UNKNOWN", "ECONOMIC_RECOVERY_UNKNOWN"
        if len(signatures) > 1:
            state, treatment = "CONFLICTING_EVIDENCE", "CONFLICTING_EVIDENCE"
        elif matches:
            treatment = matches[0]["economic_resolution_type"]
            if treatment != "ECONOMIC_RECOVERY_UNKNOWN":
                state = (
                    "EVIDENCE_COMPLETE"
                    if TerminalEconomicRecord.model_validate(matches[0]).complete()
                    else "EVIDENCE_PARTIAL"
                )
        rows.append(
            {
                "economic_boundary_id": boundary,
                "ts_code": code,
                "terminal_date": dt,
                "exposure_count": len(group),
                "evidence_record_ids": json.dumps(sorted(payload_hash(r) for r in matches)),
                "coverage": state,
                "economic_resolution_type": treatment,
            }
        )
    return pd.DataFrame(rows, columns=TERMINAL_COLUMNS)


def derive(
    source: EconomicExposureSource,
    raw: pd.DataFrame,
    source_id: str,
    requests: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    terminal: list[dict[str, Any]],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    events = normalize_dividends(
        raw, source_id, reviews, source.identity["governed_execution_cutoff"]
    )
    catalog = events[
        events.evidence_status.eq("REVIEWED_PROVIDER_EVIDENCE")
        & events.within_execution_support
        & ~events.component_type.eq("UNSUPPORTED_OTHER_EVENT")
    ].reset_index(drop=True)
    reconciliation = reconcile_adjustments(source, events)
    coverage = position_coverage(source, events)
    terminals = terminal_coverage(source, terminal)
    unmatched = reconciliation[
        ~reconciliation.classification.isin(["MATCHED_SINGLE_EVENT", "MATCHED_COMPOSITE_EVENT"])
    ]
    bad_positions = coverage[
        ~coverage.eligibility.isin(["HELD_THROUGH_RECORD_DATE", "NOT_HELD_ON_RECORD_DATE"])
        | ~coverage.evidence_status.eq("REVIEWED_PROVIDER_EVIDENCE")
    ]
    unresolved = [
        dict(
            finding_type="ADJUSTMENT",
            finding_id=r.exposure_id,
            ts_code=r.ts_code,
            reason=r.classification,
        )
        for r in unmatched.itertuples()
    ]
    unresolved.extend(
        dict(
            finding_type="TERMINAL",
            finding_id=r.economic_boundary_id,
            ts_code=r.ts_code,
            reason=r.coverage,
        )
        for r in terminals.itertuples()
        if r.coverage != "EVIDENCE_COMPLETE"
    )
    unresolved.extend(
        dict(
            finding_type="ENTITLEMENT",
            finding_id=payload_hash([r.event_id, r.position_id, r.top_n]),
            ts_code=r.ts_code,
            reason=f"{r.eligibility}:{r.evidence_status}",
        )
        for r in bad_positions.itertuples()
    )
    unresolved.extend(
        dict(
            finding_type="NORMALIZATION",
            finding_id=r.event_id,
            ts_code=r.ts_code,
            reason="EX_DATE_UNKNOWN_CANNOT_EXCLUDE_EXPOSURE",
        )
        for r in events[events.ex_date.isna()].itertuples()
    )
    incomplete = [r for r in requests if r["status"] != "RESPONSE_CAPTURED"]
    unresolved.extend(
        dict(
            finding_type="PROVIDER",
            finding_id=r["request_id"],
            ts_code=r["ts_code"],
            reason=r["status"],
        )
        for r in incomplete
    )
    conflicts = events.evidence_status.eq("MULTIPLE_CONFLICTING_EVENTS").any()
    blocked = bool(
        incomplete
        or conflicts
        or terminals.coverage.isin(
            ["ECONOMIC_RECOVERY_UNKNOWN", "CONFLICTING_EVIDENCE", "EVIDENCE_PARTIAL"]
        ).any()
    )
    status = "BLOCKED" if blocked else "PARTIAL" if unresolved else "COMPLETE"
    tables = {
        "exposure_inventory.parquet": source.exposures,
        "raw_dividend_evidence.parquet": raw,
        "dividend_candidates.parquet": events,
        "normalized_corporate_actions.parquet": catalog,
        "adjustment_event_reconciliation.parquet": reconciliation,
        "position_entitlement_coverage.parquet": coverage,
        "terminal_event_inventory.parquet": source.exposures[
            source.exposures.exposure_type.eq("TERMINAL_EVENT")
        ].reset_index(drop=True),
        "terminal_coverage.parquet": terminals,
        "unresolved_events.parquet": pd.DataFrame(
            unresolved, columns=["finding_type", "finding_id", "ts_code", "reason"]
        ),
    }
    exposed_positions = source.exposures.drop_duplicates(["top_n", "position_id"])
    summary: dict[str, Any] = {
        "economic_event_evidence_status": status,
        "unique_adj_factor_change_events": reconciliation.economic_boundary_id.nunique(),
        "matched_dividend_events": reconciliation[
            reconciliation.classification.str.startswith("MATCHED_")
        ].economic_boundary_id.nunique(),
        "unmatched_changes": unmatched.economic_boundary_id.nunique(),
        "ambiguous_events": int(
            events.evidence_status.isin(
                ["AMBIGUOUS_PROVIDER_EVENT", "MULTIPLE_CONFLICTING_EVENTS"]
            ).sum()
        ),
        "implemented_candidate_rows": len(events),
        "reviewed_catalog_events": len(catalog),
        "cash_dividend_events": int(catalog.cash_div_before_tax_per_share.fillna(0).gt(0).sum()),
        "share_distribution_events": int(
            catalog.total_stock_distribution_rate.fillna(0).gt(0).sum()
        ),
        "composite_events": int(catalog.component_type.eq("COMPOSITE_DISTRIBUTION").sum()),
        "terminal_unique_securities": terminals.ts_code.nunique(),
        "terminal_events_complete": int(terminals.coverage.eq("EVIDENCE_COMPLETE").sum()),
        "terminal_events_unknown": int(terminals.coverage.eq("ECONOMIC_RECOVERY_UNKNOWN").sum()),
        "positions_affected": len(exposed_positions),
        "capital_exposed": float(exposed_positions.entry_gross.sum()),
        "record_date_eligible_without_factor_change": int(
            (
                coverage.eligibility.eq("HELD_THROUGH_RECORD_DATE") & ~coverage.observed_adjustment
            ).sum()
        ),
        "by_top_n": {
            str(n): {
                "raw_position_count": len(g),
                "capital_material_position_count": int(g.entry_gross.gt(1).sum()),
                "capital_exposed": float(g.entry_gross.sum()),
            }
            for n, g in exposed_positions.groupby("top_n")
        },
        "capital_material_threshold": "entry_gross > 1; descriptive audit threshold only",
        "dust_execution_policy_status": "DEFERRED_SEPARATE_EXECUTION_SEMANTICS",
        "evidence_use": "POST_HOC_ACCOUNTING_EVIDENCE",
        "execution_authorized": False,
        "normalization_contract": NORMALIZATION,
        "record_date_semantics": "END_OF_DAY_HOLDINGS",
    }
    return tables, summary


def compile_economic_evidence(
    source: EconomicExposureSource,
    provider_source: Path,
    output_root: Path,
    *,
    dividend_review: Path | None = None,
    terminal_packages: tuple[Path, ...] = (),
) -> Path:
    pm = validate_provider_source(provider_source, source)
    raw = pd.read_parquet(provider_source / "raw_dividend_evidence.parquet")
    requests = read_json(provider_source / "responses.json")["requests"]
    reviews = [] if dividend_review is None else read_json(dividend_review)["reviews"]
    terminal, dependencies = [], []
    for p in terminal_packages:
        terminal.extend(validate_terminal_evidence(p))
        dependencies.append(
            {"path": str(p.resolve()), "manifest_hash": file_hash(p / "manifest.json")}
        )
    terminal.sort(key=payload_hash)
    tables, summary = derive(source, raw, pm["artifact_id"], requests, reviews, terminal)
    logical = {
        "contract": CONTRACT,
        "normalization_contract": NORMALIZATION,
        "source": source.identity,
        "provider_source_id": pm["artifact_id"],
        "provider_manifest_hash": file_hash(provider_source / "manifest.json"),
        "provider_content_hash": pm["logical_identity"]["requests_hash"],
        "review_hash": payload_hash(reviews),
        "terminal_hash": payload_hash(terminal),
        "terminal_package_hashes": sorted(d["manifest_hash"] for d in dependencies),
        "implementation_hash": implementation_hash(),
        "evidence_use": "POST_HOC_ACCOUNTING_EVIDENCE",
    }
    locators = {
        "continuous_run": str(source.run.resolve()),
        "audit_root": str(source.audit.resolve()),
        "provider_source": str(provider_source.resolve()),
        "terminal_packages": dependencies,
    }
    objects = {
        "dividend_review.json": {"reviews": reviews},
        "provider_source_inventory.json": {
            "source_id": pm["artifact_id"],
            "manifest_hash": logical["provider_manifest_hash"],
        },
        "terminal_economic_evidence.json": {"schema": TERMINAL_CONTRACT, "records": terminal},
        "summary.json": summary,
    }
    return publish(
        output_root,
        "economic_event_evidence",
        logical,
        locators,
        tables,
        objects,
        lambda p: _validate_compilation(p, source),
    )


def validate_economic_event_evidence_artifact(path: Path) -> dict[str, Any]:
    """Revalidate original run and audit, then independently recompile every derived child."""
    m = validate_envelope(path, "economic_event_evidence")
    loc, logical = m["locators"], m["logical_identity"]
    source = load_exposure_source(
        Path(loc["continuous_run"]),
        Path(loc["audit_root"]),
        continuous_hash=logical["source"]["continuous_manifest_hash"],
        audit_hash=logical["source"]["audit_manifest_hash"],
    )
    return _validate_compilation(path, source)


def _validate_compilation(path: Path, source: EconomicExposureSource) -> dict[str, Any]:
    m = validate_envelope(path, "economic_event_evidence")
    loc, logical = m["locators"], m["logical_identity"]
    require(
        logical["source"] == source.identity
        and logical["contract"] == CONTRACT
        and logical["normalization_contract"] == NORMALIZATION
        and logical.get("evidence_use") == "POST_HOC_ACCOUNTING_EVIDENCE"
        and compatible_implementation(logical["implementation_hash"]),
        "COMPILE_CONTRACT_MISMATCH",
    )
    provider = Path(loc["provider_source"])
    pm = validate_provider_source(provider, source)
    require(
        file_hash(provider / "manifest.json") == logical["provider_manifest_hash"]
        and pm["artifact_id"] == logical["provider_source_id"]
        and pm["logical_identity"]["requests_hash"] == logical["provider_content_hash"],
        "PROVIDER_DEPENDENCY_MISMATCH",
    )
    reviews = read_json(path / "dividend_review.json")["reviews"]
    require(payload_hash(reviews) == logical["review_hash"], "REVIEW_HASH_MISMATCH")
    terminal = []
    for dep in loc["terminal_packages"]:
        p = Path(dep["path"])
        require(
            file_hash(p / "manifest.json") == dep["manifest_hash"], "TERMINAL_DEPENDENCY_MISMATCH"
        )
        terminal.extend(validate_terminal_evidence(p))
    terminal.sort(key=payload_hash)
    require(
        sorted(d["manifest_hash"] for d in loc["terminal_packages"])
        == logical["terminal_package_hashes"]
        and payload_hash(terminal) == logical["terminal_hash"],
        "TERMINAL_HASH_MISMATCH",
    )
    require(
        read_json(path / "terminal_economic_evidence.json")
        == {"schema": TERMINAL_CONTRACT, "records": terminal},
        "TERMINAL_RECONSTRUCTION_MISMATCH",
    )
    require(
        read_json(path / "provider_source_inventory.json")
        == {"source_id": pm["artifact_id"], "manifest_hash": logical["provider_manifest_hash"]},
        "PROVIDER_INVENTORY_MISMATCH",
    )
    tables, summary = derive(
        source,
        pd.read_parquet(provider / "raw_dividend_evidence.parquet"),
        pm["artifact_id"],
        read_json(provider / "responses.json")["requests"],
        reviews,
        terminal,
    )
    require(read_json(path / "summary.json") == summary, "COMPLETENESS_MISMATCH")
    for name, expected in tables.items():
        require(
            content_hash(pd.read_parquet(path / name)) == content_hash(expected),
            "BUSINESS_RECONSTRUCTION_MISMATCH",
        )
    expected_files = set(tables) | {
        "dividend_review.json",
        "provider_source_inventory.json",
        "terminal_economic_evidence.json",
        "summary.json",
    }
    require(set(m["artifact_hashes"]) == expected_files, "FILE_SET_MISMATCH")
    return m
