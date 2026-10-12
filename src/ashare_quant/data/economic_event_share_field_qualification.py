"""Qualification of individual official share-field claims, never whole events."""

from __future__ import annotations

import json
import re
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
    Claim,
    Document,
    Supplement,
    _cite,
    document_pages,
    validate_supplements,
)
from ashare_quant.data.economic_events import StrictRecord, require

CONTRACT = "economic_event_share_field_qualification_v1"
KIND = "economic_event_share_field_qualification"
SHARE_FIELDS = frozenset({"stk_bo_rate", "stk_co_rate"})
# Add a historical digest only after reviewing its exact source bytes and proving
# equivalent business reconstruction with fixtures. V1 has no prior producer.
COMPATIBLE_HISTORICAL_IMPLEMENTATIONS: frozenset[str] = frozenset()
REVIEWED_COLUMNS = [
    "field_claim_hash",
    "review_event_id",
    "candidate_group_id",
    "canonical_ts_code",
    "source_row_hashes",
    "source_supplement_hash",
    "parent_manifest_hash",
    "field",
    "approved_value",
    "basis",
    "disclosed_value",
    "document_sha256",
    "document_id",
    "page",
    "quote",
    "review_decision_hash",
    "evidence_status",
    "qualification_only",
    "execution_authorized",
]


class Decision(StrictRecord):
    field_claim_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal[
        "APPROVE_FIELD_AS_EVIDENCE", "REQUIRE_OFFICIAL_DOCUMENT", "REJECT_FIELD_CLAIM"
    ]
    reviewed_by: str = Field(min_length=1)
    review_reason: str = Field(min_length=1)
    review_timestamp: str | None = None


class Input(StrictRecord):
    contract: Literal["economic_event_share_field_qualification_v1"]
    parent_artifact_id: str
    parent_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    supplement_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decisions: list[Decision]


def _implementation_hash() -> str:
    base = Path(__file__).parent
    sources = (
        "economic_event_share_field_qualification.py",
        "economic_event_supplements.py",
        "economic_events.py",
        "economic_event_sources.py",
        "economic_event_review.py",
        "economic_review_artifacts.py",
    )
    components = {name: file_hash(base / name) for name in sources}
    components["backtest/continuous_source.py"] = file_hash(
        base.parent / "backtest/continuous_source.py"
    )
    return payload_hash(components)


def compatible_share_field_implementation(digest: object) -> bool:
    return (
        isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
        and (digest == _implementation_hash() or digest in COMPATIBLE_HISTORICAL_IMPLEMENTATIONS)
    )


def _parent(parent: Path) -> dict[str, Any]:
    manifest = validate_supplements(parent)
    body = read_json(parent / "supplements.json")
    summary = read_json(parent / "summary.json")
    require(
        manifest["logical_identity"].get("status_contract_version") == 2
        and body["decisions"] == []
        and summary["supplement_reviewed_events"] == 0
        and summary["qualification_only"] is True
        and summary["execution_authorized"] is False,
        "FIELD_PARENT_PROPOSAL_ONLY_REQUIRED",
    )
    return manifest


def _source_claims(parent: Path, source_hash: str) -> list[dict[str, Any]]:
    body = read_json(parent / "supplements.json")
    supplements = [
        Supplement.model_validate(item)
        for item in body["supplements"]
        if payload_hash(Supplement.model_validate(item).model_dump()) == source_hash
    ]
    require(len(supplements) == 1, "FIELD_SOURCE_SUPPLEMENT")
    source = supplements[0]
    proposals = read_json(parent / "field_proposals.json")["proposals"]
    matched = [p for p in proposals if p["supplement_hash"] == source_hash]
    require(len(matched) == 1, "FIELD_PARENT_CLOSURE")
    original = matched[0]
    require(
        original["status"] == "PENDING_HUMAN_REVIEW"
        and original["review"] is None
        and original["reviewed_terms"] is None
        and original["conflict"] is False
        and original["issues"] == []
        and source.review_event_id == original["review_event_id"]
        and source.candidate_group_id == original["candidate_group_id"]
        and sorted(source.source_row_hashes) == sorted(original["source_row_hashes"]),
        "FIELD_PARENT_CLOSURE",
    )
    d2 = Path(read_json(parent / "manifest.json")["locators"]["parent"])
    groups = {
        row["candidate_group_id"]: row
        for row in records(pd.read_parquet(d2 / "provider_revision_groups.parquet"))
    }
    queue = {
        row["review_event_id"]: row
        for row in records(pd.read_parquet(d2 / "economic_event_review_queue.parquet"))
    }
    require(
        source.candidate_group_id in groups and source.review_event_id in queue,
        "FIELD_PARENT_CLOSURE",
    )
    group = groups[source.candidate_group_id]
    event = queue[source.review_event_id]
    require(
        source.canonical_ts_code == group["canonical_ts_code"] == event["canonical_ts_code"]
        and source.record_date == group["record_date"]
        and source.ex_date == group["ex_date"] == event["observed_adjustment_date"]
        and source.candidate_group_id in json.loads(event["exact_candidate_group_ids"])
        and sorted(source.source_row_hashes) == sorted(json.loads(group["source_row_hashes"])),
        "FIELD_PARENT_CLOSURE",
    )
    documents = {
        Document.model_validate(item).sha256: Document.model_validate(item)
        for item in body["documents"]
    }
    pages: dict[str, list[str]] = {}
    for digest in {c.document_sha256 for c in [*source.identity_citations, *source.claims]}:
        require(digest in documents, "FIELD_DOCUMENT_REFERENCE")
        doc = documents[digest]
        path = safe_file(parent, doc.file)
        require(file_hash(path) == digest, "FIELD_DOCUMENT_HASH")
        pages[digest] = document_pages(path, doc.format)
    for citation in source.identity_citations:
        _cite(citation, pages)
    require(bool(source.identity_citations), "FIELD_IDENTITY_CITATION_REQUIRED")
    claims: list[dict[str, Any]] = []
    seen: set[str] = set()
    identity = sorted((c.model_dump() for c in source.identity_citations), key=payload_hash)
    for claim in source.claims:
        require(claim.field in SHARE_FIELDS, "FIELD_UNSUPPORTED")
        require(claim.field not in seen, "FIELD_DUPLICATE_CLAIM")
        seen.add(claim.field)
        require(
            original["raw_terms"][claim.field] is None and group[claim.field] is None,
            "FIELD_PROVIDER_NOT_NULL",
        )
        require(
            claim.document_sha256 in {c.document_sha256 for c in source.identity_citations},
            "FIELD_DOCUMENT_IDENTITY",
        )
        _cite(claim, pages)
        expected = (
            claim.disclosed_value / 10
            if claim.basis == "EXPLICIT_PER_TEN_SHARES"
            else claim.disclosed_value
        )
        require(
            claim.value == expected
            and (claim.basis != "EXPLICIT_ZERO" or claim.value == claim.disclosed_value == 0),
            "FIELD_UNIT_CONFLICT",
        )
        normalized = {
            "parent_artifact_id": parent.name,
            "parent_manifest_hash": file_hash(parent / "manifest.json"),
            "source_supplement_hash": source_hash,
            "review_event_id": source.review_event_id,
            "candidate_group_id": source.candidate_group_id,
            "canonical_ts_code": source.canonical_ts_code,
            "record_date": source.record_date,
            "ex_date": source.ex_date,
            "source_row_hashes": sorted(source.source_row_hashes),
            "document_id": documents[claim.document_sha256].document_id,
            "identity_citations": identity,
            "claim": Claim.model_validate(claim.model_dump()).model_dump(),
        }
        claims.append(
            {
                "field_claim_hash": payload_hash(normalized),
                "field": claim.field,
                "value": claim.value,
                **normalized,
            }
        )
    require(bool(claims), "FIELD_EMPTY_PROPOSAL")
    return sorted(claims, key=lambda item: item["field"])


def derive(
    parent: Path, raw: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    body = Input.model_validate(raw)
    require(
        body.parent_artifact_id == parent.name
        and body.parent_manifest_hash == file_hash(parent / "manifest.json"),
        "FIELD_PARENT_HASH",
    )
    claims = _source_claims(parent, body.supplement_hash)
    decisions: dict[str, Decision] = {}
    for reviewed_decision in body.decisions:
        require(
            bool(reviewed_decision.reviewed_by.strip() and reviewed_decision.review_reason.strip()),
            "FIELD_REVIEW_MISSING",
        )
        require(reviewed_decision.field_claim_hash not in decisions, "FIELD_DUPLICATE_DECISION")
        decisions[reviewed_decision.field_claim_hash] = reviewed_decision
    require(set(decisions) <= {c["field_claim_hash"] for c in claims}, "FIELD_UNKNOWN_DECISION")
    proposals = []
    reviewed = []
    for claim in claims:
        digest = claim["field_claim_hash"]
        decision = decisions.get(digest)
        review = decision.model_dump(exclude={"review_timestamp"}) if decision else None
        status = (
            {
                "APPROVE_FIELD_AS_EVIDENCE": "REVIEWED_QUALIFICATION_ONLY",
                "REQUIRE_OFFICIAL_DOCUMENT": "OFFICIAL_DOCUMENT_REQUIRED",
                "REJECT_FIELD_CLAIM": "REJECTED_FIELD_CLAIM",
            }[decision.decision]
            if decision
            else "PENDING_HUMAN_REVIEW"
        )
        proposals.append(
            {
                **claim,
                "status": status,
                "review": review,
                "qualification_only": True,
                "execution_authorized": False,
            }
        )
        if decision and decision.decision == "APPROVE_FIELD_AS_EVIDENCE":
            c = claim["claim"]
            reviewed.append(
                {
                    "field_claim_hash": digest,
                    "review_event_id": claim["review_event_id"],
                    "candidate_group_id": claim["candidate_group_id"],
                    "canonical_ts_code": claim["canonical_ts_code"],
                    "source_row_hashes": json.dumps(claim["source_row_hashes"]),
                    "source_supplement_hash": claim["source_supplement_hash"],
                    "parent_manifest_hash": claim["parent_manifest_hash"],
                    "field": c["field"],
                    "approved_value": c["value"],
                    "basis": c["basis"],
                    "disclosed_value": c["disclosed_value"],
                    "document_sha256": c["document_sha256"],
                    "document_id": claim["document_id"],
                    "page": c["page"],
                    "quote": c["quote"],
                    "review_decision_hash": payload_hash(review),
                    "evidence_status": "REVIEWED_QUALIFICATION_ONLY",
                    "qualification_only": True,
                    "execution_authorized": False,
                }
            )
    table = pd.DataFrame(reviewed, columns=REVIEWED_COLUMNS).astype(
        {
            "approved_value": "Float64",
            "disclosed_value": "Float64",
            "page": "Int64",
            "qualification_only": "boolean",
            "execution_authorized": "boolean",
        }
    )
    logical_decisions = sorted(
        (d.model_dump(exclude={"review_timestamp"}) for d in decisions.values()),
        key=lambda item: item["field_claim_hash"],
    )
    semantic = {"claims": claims, "decisions": logical_decisions}
    objects = {
        "field_proposals.json": {"contract": CONTRACT, "proposals": proposals},
        "review_template.json": {
            "contract": CONTRACT,
            "parent_artifact_id": parent.name,
            "parent_manifest_hash": body.parent_manifest_hash,
            "decisions": [
                {
                    "field_claim_hash": p["field_claim_hash"],
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
            "field_proposals": len(proposals),
            "pending_fields": sum(p["review"] is None for p in proposals),
            "reviewed_fields": len(reviewed),
            "economic_event_evidence_status": "BLOCKED",
            "qualification_only": True,
            "execution_authorized": False,
        },
    }
    return table, objects, semantic


def _identity(parent: Path, semantic: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": CONTRACT,
        "parent_artifact_id": parent.name,
        "parent_manifest_hash": file_hash(parent / "manifest.json"),
        "implementation_hash": _implementation_hash(),
        "field_semantic_hash": payload_hash(semantic),
        "qualification_only": True,
        "execution_authorized": False,
    }


def _validate(path: Path, parent: Path) -> dict[str, Any]:
    manifest = validate_envelope(path, KIND)
    producer = manifest["logical_identity"].get("implementation_hash")
    require(compatible_share_field_implementation(producer), "FIELD_IMPLEMENTATION_UNSUPPORTED")
    raw = read_json(path / "field_qualification_input.json")
    table, objects, semantic = derive(parent, raw)
    expected = _identity(parent, semantic)
    expected["implementation_hash"] = producer
    require(manifest["logical_identity"] == expected, "FIELD_IDENTITY")
    require(
        set(manifest["artifact_hashes"])
        == set(objects)
        | {"field_qualification_input.json", "reviewed_field_qualifications.parquet"},
        "FIELD_FILE_SET",
    )
    for name, expected_obj in objects.items():
        require(read_json(path / name) == expected_obj, "FIELD_BUSINESS_MISMATCH")
    require(
        content_hash(pd.read_parquet(path / "reviewed_field_qualifications.parquet"))
        == content_hash(table),
        "FIELD_BUSINESS_MISMATCH",
    )
    return manifest


def compile_share_field_qualification(
    parent: Path, expected_hash: str, input_json: Path, output_root: Path
) -> Path:
    require(file_hash(parent / "manifest.json") == expected_hash, "FIELD_PARENT_HASH")
    _parent(parent)
    raw = read_json(input_json)
    table, objects, semantic = derive(parent, raw)
    return publish(
        output_root,
        KIND,
        _identity(parent, semantic),
        {"parent": str(parent.resolve())},
        {},
        objects | {"field_qualification_input.json": raw},
        validate_share_field_qualification,
        {"reviewed_field_qualifications.parquet": table.to_parquet(index=False)},
    )


def validate_share_field_qualification(path: Path) -> dict[str, Any]:
    manifest = validate_envelope(path, KIND)
    parent = Path(manifest["locators"]["parent"])
    require(
        file_hash(parent / "manifest.json") == manifest["logical_identity"]["parent_manifest_hash"],
        "FIELD_PARENT_HASH",
    )
    _parent(parent)
    return _validate(path, parent)
