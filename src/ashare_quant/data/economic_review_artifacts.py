"""Append-only review artifacts re-derived from recursively validated D1 inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.backtest.continuous_source import file_hash, payload_hash, read_json
from ashare_quant.data.economic_event_evidence import (
    validate_economic_event_evidence_artifact,
    validate_terminal_evidence,
)
from ashare_quant.data.economic_event_review import (
    CONTRACT,
    DECISIONS,
    RULES,
    apply_decisions,
    build_triage,
    decision_body,
)
from ashare_quant.data.economic_event_sources import (
    EconomicExposureSource,
    content_hash,
    load_exposure_source,
    publish,
    records,
    typed_table,
    validate_envelope,
    validate_provider_source,
)
from ashare_quant.data.economic_events import AMOUNTS, require
from ashare_quant.data.terminal_announcement_sources import (
    CANDIDATE_COLUMNS,
    validate_announcement_source,
)


@dataclass(frozen=True)
class ReviewInputs:
    initial: Path
    provider: Path
    source: EconomicExposureSource
    context: dict[str, Any]


def review_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Extend D1 nullable serialization without changing its content-bound implementation."""
    typed = typed_table(frame)
    floating = {
        *AMOUNTS,
        "capital_exposed",
        "capital_material_exposed",
        "factor_before",
        "factor_after",
        "factor_ratio",
    }
    integer = {
        "provider_row_count",
        "affected_position_count",
        "material_position_count",
        "candidate_count",
        "exact_ex_date_candidate_count",
    }
    for name in frame:
        if name in floating:
            typed[name] = frame[name].astype("Float64")
        elif name in integer:
            typed[name] = frame[name].astype("Int64")
        elif name == "execution_authorized":
            typed[name] = frame[name].astype("boolean")
    return typed


def implementation_hash() -> str:
    return payload_hash(
        {
            name: file_hash(Path(__file__).with_name(name))
            for name in (
                "economic_event_review.py",
                "economic_review_artifacts.py",
                "terminal_announcement_sources.py",
            )
        }
    )


def load_review_inputs(
    initial: Path, provider: Path, initial_hash: str, provider_hash: str
) -> ReviewInputs:
    require(file_hash(initial / "manifest.json") == initial_hash, "INITIAL_EVIDENCE_HASH_MISMATCH")
    require(file_hash(provider / "manifest.json") == provider_hash, "PROVIDER_HASH_MISMATCH")
    m = validate_economic_event_evidence_artifact(initial)
    require(
        m["logical_identity"]["provider_manifest_hash"] == provider_hash
        and m["logical_identity"]["provider_source_id"] == provider.name,
        "WRONG_D1_PROVIDER",
    )
    loc = m["locators"]
    identity = m["logical_identity"]["source"]
    source = load_exposure_source(
        Path(loc["continuous_run"]),
        Path(loc["audit_root"]),
        continuous_hash=identity["continuous_manifest_hash"],
        audit_hash=identity["audit_manifest_hash"],
    )
    validate_provider_source(provider, source)
    context = {
        "initial_evidence_id": m["artifact_id"],
        "initial_manifest_hash": initial_hash,
        "provider_source_id": provider.name,
        "provider_manifest_hash": provider_hash,
        "source": source.identity,
        "contract": CONTRACT,
        "rules": RULES,
        "implementation_hash": implementation_hash(),
    }
    return ReviewInputs(initial, provider, source, context)


def _template(
    tables: dict[str, pd.DataFrame], context: dict[str, Any], reviewed: set[str]
) -> dict[str, Any]:
    groups = {
        g["candidate_group_id"]: g for g in records(tables["provider_revision_groups.parquet"])
    }
    items = []
    for q in records(tables["economic_event_review_queue.parquet"]):
        if q["review_event_id"] in reviewed:
            continue
        for gid in json.loads(q["candidate_group_ids"]) or [None]:
            items.append(
                {
                    "review_event_id": q["review_event_id"],
                    "candidate_group_id": gid,
                    "decision": None,
                    "reviewed_by": "",
                    "review_reason": "",
                    "review_timestamp": None,
                    "source_row_hashes": json.loads(groups[gid]["source_row_hashes"])
                    if gid
                    else [],
                }
            )
    return {"schema": DECISIONS, "source": context, "decisions": items}


def _decisions(path: Path, context: dict[str, Any]) -> list[dict[str, Any]]:
    m = validate_envelope(path, "economic_review_decisions")
    require(
        set(m["artifact_hashes"]) == {"decisions.json", "review_metadata.json"}, "DECISION_FILE_SET"
    )
    rows = decision_body(read_json(path / "decisions.json"), context)
    require(
        m["logical_identity"]
        == {"schema": DECISIONS, "source": context, "decisions_hash": payload_hash(rows)},
        "DECISION_IDENTITY_MISMATCH",
    )
    return rows


def publish_decisions(
    path: Path, inputs: ReviewInputs, root: Path, tables: dict[str, pd.DataFrame]
) -> Path:
    body = read_json(path)
    rows = decision_body(body, inputs.context)
    require(bool(rows), "EMPTY_DECISION_SET")
    apply_decisions(inputs.source, tables, rows, [])  # Validate links before publication.
    logical = {"schema": DECISIONS, "source": inputs.context, "decisions_hash": payload_hash(rows)}
    metadata = [
        {
            "review_event_id": r["review_event_id"],
            "candidate_group_id": r["candidate_group_id"],
            "review_timestamp": r.get("review_timestamp"),
        }
        for r in body["decisions"]
    ]
    return publish(
        root,
        "economic_review_decisions",
        logical,
        {},
        {},
        {
            "decisions.json": {"schema": DECISIONS, "source": inputs.context, "decisions": rows},
            "review_metadata.json": metadata,
        },
        lambda p: _decisions(p, inputs.context),
    )


def derive_review(
    inputs: ReviewInputs,
    decisions: list[dict[str, Any]],
    terminal: list[dict[str, Any]],
    announcements: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    raw = pd.read_parquet(inputs.provider / "raw_dividend_evidence.parquet")
    tables, reduction = build_triage(inputs.source, raw)
    result, summary = apply_decisions(inputs.source, tables, decisions, terminal)
    tables.update(result)
    tables["terminal_announcement_candidates.parquet"] = announcements
    terminal_queue = []
    matches_path = inputs.source.audit / "terminal_official_index_matches.csv"
    matches = (
        pd.read_csv(matches_path, dtype=str).fillna("") if matches_path.exists() else pd.DataFrame()
    )
    for row in records(result["terminal_coverage.parquet"]):
        candidates = announcements[announcements.terminal_event_id.eq(row["economic_boundary_id"])]
        matching = (
            matches[matches.canonical_ts_code.eq(row["ts_code"])]
            if "canonical_ts_code" in matches
            else matches.iloc[:0]
        )
        terminal_queue.append(
            {
                "terminal_event_id": row["economic_boundary_id"],
                "ts_code": row["ts_code"],
                "terminal_date": row["terminal_date"],
                "candidate_documents": records(candidates),
                "existing_lifecycle_evidence": records(matching),
                "lifecycle_is_not_recovery_evidence": True,
                "coverage": row["coverage"],
                "economic_resolution_type": row["economic_resolution_type"],
            }
        )
    summary = {
        **reduction,
        **summary,
        "terminal_documents_found": int(announcements.metadata_hash.nunique()),
        "terminal_boundaries_with_documents": int(announcements.terminal_event_id.nunique()),
        "factor_movement_use": "CONSISTENCY_DIAGNOSTIC_ONLY",
        "capital_measure": "sum entry gross per exposed boundary; not unique invested capital",
        "cash_tax_semantics": "UNRESOLVED_EXECUTION_POLICY",
    }
    requests = read_json(inputs.provider / "responses.json")["requests"]
    statuses = {r["status"] for r in requests}
    summary["provider_request_statuses"] = sorted(statuses)
    summary["provider_capture_complete"] = all(s == "RESPONSE_CAPTURED" for s in statuses)
    if not summary["provider_capture_complete"]:
        summary["economic_event_evidence_status"] = "BLOCKED"
        incomplete = [
            {
                "finding_type": "PROVIDER",
                "finding_id": r["request_id"],
                "ts_code": r["ts_code"],
                "reason": r["status"],
            }
            for r in requests
            if r["status"] != "RESPONSE_CAPTURED"
        ]
        tables["unresolved_events.parquet"] = pd.concat(
            [tables["unresolved_events.parquet"], pd.DataFrame(incomplete)], ignore_index=True
        )
    reviewed = set(
        result["adjustment_reconciliation.parquet"].loc[
            result["adjustment_reconciliation.parquet"].review_status.eq("APPROVED_AS_EVIDENCE"),
            "review_event_id",
        ]
    )
    objects = {
        "summary.json": summary,
        "source_summary.json": read_json(inputs.initial / "summary.json"),
        "terminal_economic_review_queue.json": {"events": terminal_queue},
        "economic_event_review_template.json": _template(tables, inputs.context, reviewed),
    }
    return tables, objects


def _dependencies(
    inputs: ReviewInputs, terminal_paths: tuple[Path, ...], announcement: Path | None
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    terminal = []
    for p in terminal_paths:
        terminal.extend(validate_terminal_evidence(p))
    terminal.sort(key=payload_hash)
    announcements = (
        validate_announcement_source(announcement, inputs.source)
        if announcement
        else pd.DataFrame(columns=CANDIDATE_COLUMNS)
    )
    return terminal, announcements


def _all_terminal_packages(inputs: ReviewInputs, packages: tuple[Path, ...]) -> tuple[Path, ...]:
    inherited = read_json(inputs.initial / "manifest.json")["locators"]["terminal_packages"]
    unique = {file_hash(p / "manifest.json"): p for p in packages}
    for dep in inherited:
        p = Path(dep["path"])
        require(
            file_hash(p / "manifest.json") == dep["manifest_hash"], "TERMINAL_DEPENDENCY_MISMATCH"
        )
        unique[dep["manifest_hash"]] = p
    return tuple(unique[k] for k in sorted(unique))


def compile_review(
    inputs: ReviewInputs,
    output: Path,
    *,
    decisions: Path | None = None,
    terminal_packages: tuple[Path, ...] = (),
    announcement_source: Path | None = None,
) -> Path:
    terminal_packages = _all_terminal_packages(inputs, terminal_packages)
    initial_tables, _ = build_triage(
        inputs.source, pd.read_parquet(inputs.provider / "raw_dividend_evidence.parquet")
    )
    decision_artifact = (
        publish_decisions(decisions, inputs, output, initial_tables) if decisions else None
    )
    reviewed = _decisions(decision_artifact, inputs.context) if decision_artifact else []
    terminal, announcements = _dependencies(inputs, terminal_packages, announcement_source)
    tables, objects = derive_review(inputs, reviewed, terminal, announcements)
    logical = {
        "context": inputs.context,
        "review_decision_hash": payload_hash(reviewed),
        "review_decision_id": decision_artifact.name if decision_artifact else None,
        "terminal_package_hashes": sorted(
            file_hash(p / "manifest.json") for p in terminal_packages
        ),
        "terminal_discovery_hash": file_hash(announcement_source / "manifest.json")
        if announcement_source
        else None,
        "execution_authorized": False,
    }
    locators = {
        "initial_evidence": str(inputs.initial.resolve()),
        "provider_source": str(inputs.provider.resolve()),
        "decisions": str(decision_artifact.resolve()) if decision_artifact else None,
        "terminal_packages": [str(p.resolve()) for p in terminal_packages],
        "announcement_source": str(announcement_source.resolve()) if announcement_source else None,
    }
    return publish(
        output,
        "economic_event_review",
        logical,
        locators,
        {},
        objects,
        lambda p: _validate_review(p, inputs),
        {
            **{name: review_table(frame).to_parquet(index=False) for name, frame in tables.items()},
            "economic_event_review_queue.csv": tables["economic_event_review_queue.parquet"]
            .to_csv(index=False, lineterminator="\n")
            .encode(),
        },
    )


def validate_reviewed_economic_evidence_artifact(path: Path) -> dict[str, Any]:
    m = validate_envelope(path, "economic_event_review")
    context, loc = m["logical_identity"]["context"], m["locators"]
    inputs = load_review_inputs(
        Path(loc["initial_evidence"]),
        Path(loc["provider_source"]),
        context["initial_manifest_hash"],
        context["provider_manifest_hash"],
    )
    return _validate_review(path, inputs)


def _validate_review(path: Path, inputs: ReviewInputs) -> dict[str, Any]:
    m = validate_envelope(path, "economic_event_review")
    loc, logical = m["locators"], m["logical_identity"]
    decisions = _decisions(Path(loc["decisions"]), inputs.context) if loc["decisions"] else []
    packages = tuple(Path(p) for p in loc["terminal_packages"])
    require(
        sorted(file_hash(p / "manifest.json") for p in packages)
        == sorted(file_hash(p / "manifest.json") for p in _all_terminal_packages(inputs, packages)),
        "INHERITED_TERMINAL_EVIDENCE_MISSING",
    )
    announcement = Path(loc["announcement_source"]) if loc["announcement_source"] else None
    terminal, candidates = _dependencies(inputs, packages, announcement)
    require(
        logical
        == {
            "context": inputs.context,
            "review_decision_hash": payload_hash(decisions),
            "review_decision_id": Path(loc["decisions"]).name if loc["decisions"] else None,
            "terminal_package_hashes": sorted(file_hash(p / "manifest.json") for p in packages),
            "terminal_discovery_hash": file_hash(announcement / "manifest.json")
            if announcement
            else None,
            "execution_authorized": False,
        },
        "REVIEW_IDENTITY_MISMATCH",
    )
    tables, objects = derive_review(inputs, decisions, terminal, candidates)
    require(
        set(m["artifact_hashes"])
        == set(tables) | set(objects) | {"economic_event_review_queue.csv"},
        "REVIEW_FILE_SET",
    )
    for name, frame in tables.items():
        require(
            content_hash(pd.read_parquet(path / name)) == content_hash(review_table(frame)),
            "REVIEW_BUSINESS_MISMATCH",
        )
    for name, value in objects.items():
        require(read_json(path / name) == value, "REVIEW_BUSINESS_MISMATCH")
    require(
        (path / "economic_event_review_queue.csv").read_bytes()
        == tables["economic_event_review_queue.parquet"]
        .to_csv(index=False, lineterminator="\n")
        .encode(),
        "REVIEW_CSV_MISMATCH",
    )
    return m
