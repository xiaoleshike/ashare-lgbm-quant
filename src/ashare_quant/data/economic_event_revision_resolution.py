"""Event-level official reconciliation of conflicting provider observations.

This is qualification evidence only. It never chooses a provider row or authorizes
portfolio accounting. A reviewed decision binds the complete conflict closure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import Field

from ashare_quant.backtest.continuous_source import file_hash, payload_hash, read_json
from ashare_quant.data.economic_event_sources import (
    content_hash,
    publish,
    records,
    safe_file,
    validate_envelope,
)
from ashare_quant.data.economic_event_supplements import (
    STATUS_CONTRACT_VERSION,
    Citation,
    Claim,
    _cite,
    document_pages,
    validate_supplements,
)
from ashare_quant.data.economic_events import AMOUNTS, StrictRecord, date_value, require

CONTRACT = "economic_event_revision_resolution_v1"
KIND = "economic_event_revision_resolution"
MODE = "OFFICIAL_FIELD_RECONCILIATION_NO_PROVIDER_SUPERSESSION"
AMOUNT_FIELDS = frozenset(AMOUNTS)
UNRESOLVED_FIELDS = AMOUNT_FIELDS | {"pay_date"}
REVIEWED_COLUMNS = [
    "resolution_hash",
    "review_event_id",
    "canonical_ts_code",
    "candidate_group_ids",
    "source_row_hashes",
    "supplement_hashes",
    "provider_supersession",
    "record_date",
    "ex_date",
    "authoritative_economic_terms",
    "review_decision_hash",
    "evidence_status",
    "qualification_only",
    "execution_authorized",
    "cash_tax_semantics",
]


class Proposal(StrictRecord):
    contract: Literal["economic_event_revision_resolution_v1"]
    parent_artifact_id: str
    parent_manifest_hash: str
    review_event_id: str
    canonical_ts_code: str
    end_date: str
    record_date: str
    ex_date: str
    imp_ann_date: str
    div_proc: Literal["实施"]
    resolution_mode: Literal["OFFICIAL_FIELD_RECONCILIATION_NO_PROVIDER_SUPERSESSION"]
    provider_supersession: Literal["UNKNOWN"]
    candidate_group_ids: list[str]
    source_row_hashes: list[str]
    supplement_hashes: list[str]
    request_id: str
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_id: str
    identity_citations: list[Citation]
    authoritative_terms: list[Claim]
    unresolved_fields: list[str]
    qualification_only: Literal[True]
    execution_authorized: Literal[False]


class Decision(StrictRecord):
    resolution_hash: str
    decision: Literal[
        "APPROVE_OFFICIAL_RECONCILIATION_AS_EVIDENCE",
        "REQUIRE_OFFICIAL_DOCUMENT",
        "REJECT_NOT_SAME_EVENT",
    ]
    reviewed_by: str = Field(min_length=1)
    review_reason: str = Field(min_length=1)
    review_timestamp: str | None = None


def _exact_set(values: list[str], expected: set[str], code: str) -> list[str]:
    require(len(values) == len(set(values)) and set(values) == expected, code)
    return sorted(values)


def _closure(
    proposal: Proposal,
    queue: dict[str, dict[str, Any]],
    groups: dict[str, dict[str, Any]],
    supplements: list[dict[str, Any]],
    raw: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[str], list[str], list[str], list[dict[str, Any]]]:
    require(proposal.review_event_id in queue, "REVISION_EVENT_NOT_FOUND")
    event = queue[proposal.review_event_id]
    exact = set(json.loads(event["exact_candidate_group_ids"]))
    require(
        event["scope"] == "AUDIT_ADJUSTMENT" and len(exact) >= 2, "REVISION_EXACT_CONFLICT_REQUIRED"
    )
    require(exact <= set(groups), "REVISION_GROUP_MISSING")
    closure = set(exact)
    pending = list(exact)
    while pending:
        for other in json.loads(groups[pending.pop()]["conflict_group_ids"]):
            require(other in groups, "REVISION_GROUP_MISSING")
            if other not in closure:
                closure.add(other)
                pending.append(other)
    require(
        any(json.loads(groups[g]["conflict_group_ids"]) for g in exact) and closure == exact,
        "REVISION_EXACT_CONFLICT_REQUIRED",
    )
    group_ids = _exact_set(proposal.candidate_group_ids, closure, "REVISION_GROUP_SET")
    hashes = {h for group_id in closure for h in json.loads(groups[group_id]["source_row_hashes"])}
    row_hashes = _exact_set(proposal.source_row_hashes, hashes, "REVISION_RAW_HASH_SET")
    linked = {
        item["supplement_hash"]
        for item in supplements
        if item["review_event_id"] == proposal.review_event_id
        and item["candidate_group_id"] in closure
    }
    supplement_hashes = _exact_set(proposal.supplement_hashes, linked, "REVISION_SUPPLEMENT_SET")
    require(bool(linked), "REVISION_SUPPLEMENT_REQUIRED")
    require(all(h in raw for h in hashes), "REVISION_RAW_ROW_MISSING")
    observations = []
    requests = set()
    implementation_dates = set()
    for group_id in group_ids:
        group = groups[group_id]
        require(
            group["canonical_ts_code"] == proposal.canonical_ts_code
            and group["end_date"] == proposal.end_date
            and group["record_date"] == proposal.record_date
            and group["ex_date"] == proposal.ex_date
            and group["div_proc"] == "IMPLEMENTED",
            "REVISION_EVENT_IDENTITY",
        )
        implementation_dates.update(json.loads(group["implementation_announcements"]))
        for row_hash in sorted(json.loads(group["source_row_hashes"])):
            row = raw[row_hash]
            body = json.loads(row["raw_row_json"])
            require(
                payload_hash(body) == row_hash
                and row["ts_code"] == proposal.canonical_ts_code
                and body.get("end_date") == proposal.end_date
                and body.get("record_date") == proposal.record_date
                and body.get("ex_date") == proposal.ex_date
                and body.get("imp_ann_date") == proposal.imp_ann_date
                and body.get("div_proc") == proposal.div_proc,
                "REVISION_RAW_IDENTITY",
            )
            requests.add(row["request_id"])
            observations.append({"candidate_group_id": group_id, **row})
    require(
        implementation_dates == {proposal.imp_ann_date} and requests == {proposal.request_id},
        "REVISION_REQUEST_OR_ANNOUNCEMENT",
    )
    require(
        event["canonical_ts_code"] == proposal.canonical_ts_code
        and event["observed_adjustment_date"] == proposal.ex_date
        and set(json.loads(event["exact_candidate_group_ids"])) == closure,
        "REVISION_EVENT_IDENTITY",
    )
    for value in (proposal.end_date, proposal.record_date, proposal.ex_date, proposal.imp_ann_date):
        require(date_value(value) == value, "REVISION_DATE_REQUIRED")
    require(proposal.record_date < proposal.ex_date, "REVISION_DATE_ORDER")
    return event, group_ids, row_hashes, supplement_hashes, observations


def _official_terms(
    proposal: Proposal, document: dict[str, Any], parent: Path, observations: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    require(
        proposal.document_sha256 == document["sha256"]
        and proposal.document_id == document["document_id"],
        "REVISION_DOCUMENT_IDENTITY",
    )
    file = safe_file(parent, document["file"])
    require(file_hash(file) == proposal.document_sha256, "REVISION_DOCUMENT_HASH")
    pages = {proposal.document_sha256: document_pages(file, document["format"])}
    citations = [c.model_dump() for c in proposal.identity_citations]
    require(bool(citations), "REVISION_IDENTITY_CITATION_REQUIRED")
    require(
        all(c["document_sha256"] == proposal.document_sha256 for c in citations),
        "REVISION_DOCUMENT_IDENTITY",
    )
    locator = " ".join(_cite(Citation.model_validate(c), pages) for c in citations)
    require(proposal.canonical_ts_code.split(".")[0] in locator, "REVISION_DOCUMENT_SECURITY")
    for date in (proposal.record_date, proposal.ex_date, proposal.imp_ann_date):
        variants = (
            date,
            f"{date[:4]}年{int(date[4:6])}月{int(date[6:])}日",
            f"{date[:4]}/{int(date[4:6])}/{int(date[6:])}",
        )
        require(any(v in locator for v in variants), "REVISION_DOCUMENT_DATE")
    require(proposal.end_date[:4] in locator, "REVISION_DOCUMENT_REPORT_PERIOD")
    fields = set()
    terms = []
    for claim in proposal.authoritative_terms:
        require(claim.document_sha256 == proposal.document_sha256, "REVISION_DOCUMENT_IDENTITY")
        _cite(claim, pages)
        require(claim.field not in fields, "REVISION_DUPLICATE_FIELD")
        fields.add(claim.field)
        expected = (
            claim.disclosed_value / 10
            if claim.basis == "EXPLICIT_PER_TEN_SHARES"
            else claim.disclosed_value
        )
        require(
            math.isclose(claim.value, expected, abs_tol=1e-12, rel_tol=1e-10)
            and (claim.basis != "EXPLICIT_ZERO" or claim.value == claim.disclosed_value == 0),
            "REVISION_UNIT_CONFLICT",
        )
        terms.append(claim.model_dump())
    require(fields <= AMOUNT_FIELDS, "REVISION_FIELD_UNSUPPORTED")
    pay_date_unknown = all(
        json.loads(r["raw_row_json"]).get("pay_date") is None for r in observations
    )
    unresolved = _exact_set(
        proposal.unresolved_fields,
        set(AMOUNT_FIELDS - fields) | ({"pay_date"} if pay_date_unknown else set()),
        "REVISION_UNRESOLVED_FIELDS",
    )
    require(set(unresolved) <= UNRESOLVED_FIELDS, "REVISION_UNRESOLVED_FIELDS")
    return sorted(terms, key=lambda item: item["field"]), unresolved


def derive(
    parent: Path, body: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    require(
        set(body)
        == {"contract", "parent_artifact_id", "parent_manifest_hash", "resolutions", "decisions"}
        and body["contract"] == CONTRACT
        and body["parent_artifact_id"] == parent.name
        and body["parent_manifest_hash"] == file_hash(parent / "manifest.json"),
        "REVISION_PARENT_INPUT",
    )
    require(bool(body["resolutions"]), "REVISION_EMPTY_PROPOSALS")
    parent_manifest = read_json(parent / "manifest.json")
    d2 = Path(parent_manifest["locators"]["parent"])
    queue = {
        row["review_event_id"]: row
        for row in records(pd.read_parquet(d2 / "economic_event_review_queue.parquet"))
    }
    groups = {
        row["candidate_group_id"]: row
        for row in records(pd.read_parquet(d2 / "provider_revision_groups.parquet"))
    }
    raw = {
        row["provider_raw_row_hash"]: row
        for row in records(pd.read_parquet(d2 / "relevant_provider_rows.parquet"))
    }
    source = read_json(parent / "supplements.json")
    documents = {doc["sha256"]: doc for doc in source["documents"]}
    supplements = read_json(parent / "field_proposals.json")["proposals"]
    decisions: dict[str, Decision] = {}
    for item in body["decisions"]:
        parsed_decision = Decision.model_validate(item)
        require(
            bool(parsed_decision.reviewed_by.strip() and parsed_decision.review_reason.strip()),
            "REVISION_REVIEW_MISSING",
        )
        require(parsed_decision.resolution_hash not in decisions, "REVISION_DUPLICATE_DECISION")
        decisions[parsed_decision.resolution_hash] = parsed_decision
    proposals: list[dict[str, Any]] = []
    observations_by_resolution: list[dict[str, Any]] = []
    reviewed: list[dict[str, Any]] = []
    seen_events: set[str] = set()
    for item in body["resolutions"]:
        proposal = Proposal.model_validate(item)
        require(
            proposal.parent_artifact_id == parent.name
            and proposal.parent_manifest_hash == body["parent_manifest_hash"],
            "REVISION_PARENT_INPUT",
        )
        require(proposal.review_event_id not in seen_events, "REVISION_DUPLICATE_EVENT")
        seen_events.add(proposal.review_event_id)
        _, group_ids, row_hashes, supplement_hashes, observations = _closure(
            proposal, queue, groups, supplements, raw
        )
        require(proposal.document_sha256 in documents, "REVISION_DOCUMENT_NOT_IN_PARENT")
        terms, unresolved = _official_terms(
            proposal, documents[proposal.document_sha256], parent, observations
        )
        normalized = proposal.model_dump()
        normalized.update(
            candidate_group_ids=group_ids,
            source_row_hashes=row_hashes,
            supplement_hashes=supplement_hashes,
            identity_citations=sorted(normalized["identity_citations"], key=payload_hash),
            authoritative_terms=terms,
            unresolved_fields=unresolved,
        )
        resolution_hash = payload_hash(normalized)
        decision = decisions.get(resolution_hash)
        approved = bool(
            decision and decision.decision == "APPROVE_OFFICIAL_RECONCILIATION_AS_EVIDENCE"
        )
        values = {term["field"]: term["value"] for term in terms}
        complete = not (set(unresolved) & AMOUNT_FIELDS) and set(values) == AMOUNT_FIELDS
        consistent = (
            complete
            and values["cash_div"] <= values["cash_div_tax"] + 1e-10
            and math.isclose(
                values["stk_div"],
                values["stk_bo_rate"] + values["stk_co_rate"],
                abs_tol=1e-10,
                rel_tol=1e-8,
            )
        )
        require(not approved or consistent, "REVISION_NOT_APPROVABLE")
        status = (
            {
                "APPROVE_OFFICIAL_RECONCILIATION_AS_EVIDENCE": "REVIEWED_QUALIFICATION_ONLY",
                "REQUIRE_OFFICIAL_DOCUMENT": "OFFICIAL_DOCUMENT_REQUIRED",
                "REJECT_NOT_SAME_EVENT": "REJECTED_NOT_SAME_EVENT",
            }[decision.decision]
            if decision
            else "PENDING_HUMAN_REVIEW"
        )
        review = decision.model_dump(exclude={"review_timestamp"}) if decision else None
        proposals.append(
            {
                "resolution_hash": resolution_hash,
                "proposal": normalized,
                "provider_supersession": "UNKNOWN",
                "status": status,
                "complete_and_consistent": bool(consistent),
                "review": review,
                "qualification_only": True,
                "execution_authorized": False,
            }
        )
        observations_by_resolution.append(
            {
                "resolution_hash": resolution_hash,
                "review_event_id": proposal.review_event_id,
                "provider_supersession": "UNKNOWN",
                "observations": sorted(
                    observations,
                    key=lambda row: (row["candidate_group_id"], row["provider_raw_row_hash"]),
                ),
            }
        )
        if approved:
            reviewed.append(
                {
                    "resolution_hash": resolution_hash,
                    "review_event_id": proposal.review_event_id,
                    "canonical_ts_code": proposal.canonical_ts_code,
                    "candidate_group_ids": json.dumps(group_ids),
                    "source_row_hashes": json.dumps(row_hashes),
                    "supplement_hashes": json.dumps(supplement_hashes),
                    "provider_supersession": "UNKNOWN",
                    "record_date": proposal.record_date,
                    "ex_date": proposal.ex_date,
                    "authoritative_economic_terms": json.dumps(values, sort_keys=True),
                    "review_decision_hash": payload_hash(review),
                    "evidence_status": "REVIEWED_QUALIFICATION_ONLY",
                    "qualification_only": True,
                    "execution_authorized": False,
                    "cash_tax_semantics": "UNRESOLVED_EXECUTION_POLICY",
                }
            )
    require(
        set(decisions) <= {p["resolution_hash"] for p in proposals}, "REVISION_UNKNOWN_DECISION"
    )
    proposals.sort(key=lambda p: p["resolution_hash"])
    observations_by_resolution.sort(key=lambda item: item["resolution_hash"])
    logical_decisions = sorted(
        (d.model_dump(exclude={"review_timestamp"}) for d in decisions.values()),
        key=lambda item: item["resolution_hash"],
    )
    semantic = {"proposals": [p["proposal"] for p in proposals], "decisions": logical_decisions}
    table = pd.DataFrame(reviewed, columns=REVIEWED_COLUMNS)
    table = table.astype({"qualification_only": "boolean", "execution_authorized": "boolean"})
    objects = {
        "resolution_proposals.json": {"contract": CONTRACT, "proposals": proposals},
        "provider_observations.json": {
            "contract": CONTRACT,
            "resolutions": observations_by_resolution,
        },
        "authoritative_event_terms.json": {
            "contract": CONTRACT,
            "events": [
                {
                    "resolution_hash": p["resolution_hash"],
                    "terms": p["proposal"]["authoritative_terms"],
                    "unresolved_fields": p["proposal"]["unresolved_fields"],
                }
                for p in proposals
            ],
        },
        "review_template.json": {
            "contract": CONTRACT,
            "parent_artifact_id": parent.name,
            "parent_manifest_hash": body["parent_manifest_hash"],
            "decisions": [
                {
                    "resolution_hash": p["resolution_hash"],
                    "decision": None,
                    "reviewed_by": "",
                    "review_reason": "",
                }
                for p in proposals
                if p["review"] is None
            ],
        },
        "summary.json": {
            "contract": CONTRACT,
            "proposal_events": len(proposals),
            "reviewed_resolution_events": len(reviewed),
            "original_conflicts_preserved": True,
            "economic_event_evidence_status": "BLOCKED",
            "qualification_only": True,
            "execution_authorized": False,
        },
    }
    return table, objects, semantic


def _identity(parent: Path, semantic: dict[str, Any]) -> dict[str, Any]:
    base = Path(__file__).parent
    return {
        "contract": CONTRACT,
        "parent_artifact_id": parent.name,
        "parent_manifest_hash": file_hash(parent / "manifest.json"),
        "implementation_hash": payload_hash(
            {
                name: file_hash(base / name)
                for name in (
                    "economic_event_revision_resolution.py",
                    "economic_event_supplements.py",
                    "economic_events.py",
                )
            }
        ),
        "resolution_semantic_hash": payload_hash(semantic),
        "qualification_only": True,
        "execution_authorized": False,
    }


def _validate(path: Path, parent: Path) -> dict[str, Any]:
    manifest = validate_envelope(path, KIND)
    body = read_json(path / "resolution_input.json")
    table, objects, semantic = derive(parent, body)
    require(manifest["logical_identity"] == _identity(parent, semantic), "REVISION_IDENTITY")
    require(
        set(manifest["artifact_hashes"])
        == set(objects) | {"resolution_input.json", "reviewed_revision_resolutions.parquet"},
        "REVISION_FILE_SET",
    )
    for name, obj in objects.items():
        require(read_json(path / name) == obj, "REVISION_BUSINESS_MISMATCH")
    require(
        content_hash(pd.read_parquet(path / "reviewed_revision_resolutions.parquet"))
        == content_hash(table),
        "REVISION_BUSINESS_MISMATCH",
    )
    return manifest


def compile_revision_resolutions(
    parent: Path, expected_hash: str, input_json: Path, output_root: Path
) -> Path:
    require(file_hash(parent / "manifest.json") == expected_hash, "REVISION_PARENT_HASH")
    _validated_parent(parent)
    body = read_json(input_json)
    table, objects, semantic = derive(parent, body)
    return publish(
        output_root,
        KIND,
        _identity(parent, semantic),
        {"parent": str(parent.resolve())},
        {},
        objects | {"resolution_input.json": body},
        lambda path: _validate(path, parent),
        {"reviewed_revision_resolutions.parquet": table.to_parquet(index=False)},
    )


def validate_revision_resolution_artifact(path: Path) -> dict[str, Any]:
    manifest = validate_envelope(path, KIND)
    parent = Path(manifest["locators"]["parent"])
    require(
        file_hash(parent / "manifest.json") == manifest["logical_identity"]["parent_manifest_hash"],
        "REVISION_PARENT_HASH",
    )
    _validated_parent(parent)
    return _validate(path, parent)


def _validated_parent(parent: Path) -> None:
    manifest = validate_supplements(parent)
    summary = read_json(parent / "summary.json")
    require(
        manifest["logical_identity"].get("status_contract_version") == STATUS_CONTRACT_VERSION
        and manifest["logical_identity"].get("qualification_only") is True
        and manifest["logical_identity"].get("execution_authorized") is False
        and summary.get("economic_event_evidence_status") == "BLOCKED"
        and summary.get("qualification_only") is True
        and summary.get("execution_authorized") is False,
        "REVISION_PARENT_V2_QUALIFICATION_REQUIRED",
    )
