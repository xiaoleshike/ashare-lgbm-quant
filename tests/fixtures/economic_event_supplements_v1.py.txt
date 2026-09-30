"""Event-specific official field evidence; no universal nullable-field inference.

The original review artifact stays authoritative for membership and conflicts.
Documents/claims are proposals until a separate human decision binds their hash.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import pandas as pd
from pydantic import Field

from ashare_quant.backtest.continuous_source import file_hash, payload_hash, read_json
from ashare_quant.data.economic_event_review import MATERIAL, _parse
from ashare_quant.data.economic_event_sources import (
    content_hash,
    implementation_hash,
    publish,
    records,
    safe_file,
    validate_envelope,
)
from ashare_quant.data.economic_events import StrictRecord, amount, require
from ashare_quant.data.economic_review_artifacts import (
    review_table,
    validate_reviewed_economic_evidence_artifact,
)

CONTRACT = "economic_official_field_supplement_v1"
KIND = "economic_event_supplement_review"


class Document(StrictRecord):
    file: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    url: str
    document_id: str = Field(min_length=1)
    format: Literal["PDF", "HTML"]


class Citation(StrictRecord):
    document_sha256: str
    page: int = Field(ge=1)
    quote: str = Field(min_length=1)


class Claim(Citation):
    field: Literal["cash_div", "cash_div_tax", "stk_div", "stk_bo_rate", "stk_co_rate"]
    value: float = Field(ge=0, allow_inf_nan=False)
    basis: Literal["EXPLICIT_ZERO", "EXPLICIT_PER_SHARE", "EXPLICIT_PER_TEN_SHARES"]
    disclosed_value: float = Field(ge=0, allow_inf_nan=False)


class Supplement(StrictRecord):
    review_event_id: str
    candidate_group_id: str
    canonical_ts_code: str
    record_date: str
    ex_date: str
    source_row_hashes: list[str]
    identity_citations: list[Citation]
    claims: list[Claim]


class Decision(StrictRecord):
    supplement_hash: str
    decision: Literal["APPROVE_AS_EVIDENCE", "REQUIRE_OFFICIAL_DOCUMENT", "REJECT_NOT_SAME_EVENT"]
    reviewed_by: str = Field(min_length=1)
    review_reason: str = Field(min_length=1)
    review_timestamp: str | None = None


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def document_pages(path: Path, fmt: str) -> list[str]:
    """Native extraction, never OCR. HTML supplements must declare UTF-8 bytes."""
    if fmt == "PDF":
        require(path.read_bytes().startswith(b"%PDF"), "OFFICIAL_DOCUMENT_FORMAT")
        executable = shutil.which("pdftotext")
        require(executable is not None, "NATIVE_PDF_EXTRACTOR_REQUIRED")
        # Fixed executable and arguments, no shell; document path is resolved under its root.
        result = subprocess.run(  # noqa: S603
            [str(executable), "-layout", str(path.resolve()), "-"],
            check=True,
            capture_output=True,
            timeout=60,
        )
        return result.stdout.decode("utf-8").split("\f")
    parser = _HTMLText()
    parser.feed(path.read_bytes().decode("utf-8", errors="strict"))
    return [" ".join(parser.parts)]


def _compact(text: str) -> str:
    return "".join(text.split())


def _cite(c: Citation, pages: dict[str, list[str]]) -> str:
    require(c.document_sha256 in pages, "OFFICIAL_DOCUMENT_REFERENCE")
    text = pages[c.document_sha256]
    require(c.page <= len(text), "OFFICIAL_DOCUMENT_PAGE")
    require(_compact(c.quote) in _compact(text[c.page - 1]), "OFFICIAL_QUOTE_MISMATCH")
    return _compact(c.quote)


def derive(
    parent: Path, body: dict[str, Any], document_root: Path
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    require(body.get("contract") == CONTRACT, "SUPPLEMENT_CONTRACT")
    require(
        set(body) == {"contract", "parent_manifest_hash", "documents", "supplements", "decisions"},
        "SUPPLEMENT_INPUT_FIELDS",
    )
    require(
        body.get("parent_manifest_hash") == file_hash(parent / "manifest.json"),
        "SUPPLEMENT_PARENT_HASH",
    )
    pages = {}
    for raw in body["documents"]:
        doc = Document.model_validate(raw)
        host = urlparse(doc.url).hostname or ""
        require(
            urlparse(doc.url).scheme == "https"
            and any(
                host == h or host.endswith("." + h)
                for h in ("cninfo.com.cn", "sse.com.cn", "szse.cn", "bse.cn")
            ),
            "OFFICIAL_SOURCE_AUTHORITY",
        )
        path = safe_file(document_root, doc.file)
        require(file_hash(path) == doc.sha256, "OFFICIAL_DOCUMENT_HASH")
        require(doc.sha256 not in pages, "DUPLICATE_OFFICIAL_DOCUMENT")
        pages[doc.sha256] = document_pages(path, doc.format)
    queue = records(pd.read_parquet(parent / "economic_event_review_queue.parquet"))
    by_event = {q["review_event_id"]: q for q in queue}
    groups = {
        g["candidate_group_id"]: g
        for g in records(pd.read_parquet(parent / "provider_revision_groups.parquet"))
    }
    decisions = {}
    for raw in body.get("decisions", []):
        decision = Decision.model_validate(raw)
        require(
            bool(decision.reviewed_by.strip() and decision.review_reason.strip()),
            "SUPPLEMENT_REVIEW_MISSING",
        )
        require(decision.supplement_hash not in decisions, "DUPLICATE_SUPPLEMENT_DECISION")
        decisions[decision.supplement_hash] = decision
    proposals: list[dict[str, Any]] = []
    seen = set()
    for raw in body["supplements"]:
        s = Supplement.model_validate(raw)
        digest = payload_hash(s.model_dump())
        key = (s.review_event_id, s.candidate_group_id)
        require(key not in seen, "DUPLICATE_SUPPLEMENT")
        seen.add(key)
        require(
            s.review_event_id in by_event and s.candidate_group_id in groups,
            "SUPPLEMENT_EVENT_REFERENCE",
        )
        q, g = by_event[s.review_event_id], groups[s.candidate_group_id]
        require(
            s.candidate_group_id in json.loads(q["exact_candidate_group_ids"])
            and s.canonical_ts_code == q["canonical_ts_code"] == g["canonical_ts_code"]
            and s.ex_date == q["observed_adjustment_date"] == g["ex_date"]
            and s.record_date == g["record_date"]
            and bool(s.record_date and s.ex_date),
            "SUPPLEMENT_EXACT_IDENTITY_REQUIRED",
        )
        require(
            sorted(s.source_row_hashes) == sorted(json.loads(g["source_row_hashes"]))
            and len(set(s.source_row_hashes)) == len(s.source_row_hashes),
            "SUPPLEMENT_RAW_HASH_SET",
        )
        identity_documents = {c.document_sha256 for c in s.identity_citations}
        for digest_doc in identity_documents:
            citations = " ".join(
                _cite(c, pages) for c in s.identity_citations if c.document_sha256 == digest_doc
            )
            # Locator guard only: a human must still review the document's actual semantics.
            require(s.canonical_ts_code.split(".")[0] in citations, "DOCUMENT_SECURITY_MISMATCH")
            for date in (s.record_date, s.ex_date):
                variants = (
                    date,
                    f"{date[:4]}年{int(date[4:6])}月{int(date[6:])}日",
                    f"{date[:4]}/{int(date[4:6])}/{int(date[6:])}",
                )
                require(any(v in citations for v in variants), "DOCUMENT_EVENT_DATE_MISMATCH")
        terms = {k: g[k] for k in MATERIAL}
        fields = set()
        for c in s.claims:
            _cite(c, pages)
            require(
                c.document_sha256 in identity_documents,
                "CLAIM_DOCUMENT_IDENTITY_MISSING",
            )
            require(c.field not in fields, "DUPLICATE_FIELD_CLAIM")
            fields.add(c.field)
            require(terms[c.field] is None, "SUPPLEMENT_CANNOT_OVERRIDE_PROVIDER")
            expected = (
                c.disclosed_value / 10
                if c.basis == "EXPLICIT_PER_TEN_SHARES"
                else c.disclosed_value
            )
            require(
                c.value == expected and (c.basis != "EXPLICIT_ZERO" or c.value == 0),
                "SUPPLEMENT_UNIT_CONFLICT",
            )
            terms[c.field] = amount(c.value)
        require(bool(fields), "EMPTY_SUPPLEMENT")
        parsed = _parse({"ts_code": s.canonical_ts_code, "raw_row_json": json.dumps(terms)})
        blocked = (
            bool(json.loads(g["conflict_group_ids"])) or q["exact_ex_date_candidate_count"] != 1
        )
        issues = sorted(
            set(json.loads(g["issues"])) - {"MISSING_ECONOMIC_COMPONENTS"}
            | set(json.loads(parsed["issues"]))
        )
        d = decisions.get(digest)
        approved = bool(d and d.decision == "APPROVE_AS_EVIDENCE")
        require(
            not approved
            or (
                not blocked
                and not issues
                and parsed["share_consistency_diagnostic"] == "CONSISTENT"
            ),
            "SUPPLEMENT_NOT_APPROVABLE",
        )
        proposals.append(
            dict(
                supplement_hash=digest,
                review_event_id=s.review_event_id,
                candidate_group_id=s.candidate_group_id,
                source_row_hashes=s.source_row_hashes,
                raw_terms={k: g[k] for k in MATERIAL},
                proposed_terms=terms,
                issues=issues,
                conflict=blocked,
                field_claims=[c.model_dump() for c in s.claims],
                status="REVIEWED_QUALIFICATION_ONLY"
                if approved
                else "CONFLICT_REQUIRES_SEPARATE_REVIEW"
                if blocked
                else "PENDING_HUMAN_REVIEW",
                reviewed_terms=terms if approved else None,
                review=d.model_dump(exclude={"review_timestamp"}) if d else None,
                cash_tax_semantics="UNRESOLVED_EXECUTION_POLICY",
                qualification_only=True,
                execution_authorized=False,
            )
        )
    require(
        set(decisions) <= {p["supplement_hash"] for p in proposals}, "UNKNOWN_SUPPLEMENT_DECISION"
    )
    for q in queue:
        linked = [p for p in proposals if p["review_event_id"] == q["review_event_id"]]
        q["supplement_hashes"] = json.dumps([p["supplement_hash"] for p in linked])
        q["supplement_status"] = (
            linked[0]["status"] if len(linked) == 1 else "NO_REVIEWED_SUPPLEMENT"
        )
        q["execution_authorized"] = False
    summary = read_json(parent / "summary.json") | {
        "contract": CONTRACT,
        "supplement_proposals": len(proposals),
        "supplement_reviewed_events": sum(p["reviewed_terms"] is not None for p in proposals),
        "qualification_only": True,
        "execution_authorized": False,
        # This additive qualification artifact does not replace D2's global completeness gate.
        "economic_event_evidence_status": "BLOCKED",
    }
    template = {
        "contract": CONTRACT,
        "parent_manifest_hash": body["parent_manifest_hash"],
        "decisions": [
            {
                "supplement_hash": p["supplement_hash"],
                "decision": None,
                "reviewed_by": "",
                "review_reason": "",
            }
            for p in proposals
            if p["review"] is None
        ],
    }
    accepted = [
        {
            "supplement_hash": p["supplement_hash"],
            "review_event_id": p["review_event_id"],
            "canonical_ts_code": by_event[p["review_event_id"]]["canonical_ts_code"],
            "source_row_hashes": json.dumps(p["source_row_hashes"]),
            "review_decision_hash": payload_hash(p["review"]),
            **p["reviewed_terms"],
            "qualification_only": True,
            "execution_authorized": False,
            "cash_tax_semantics": "UNRESOLVED_EXECUTION_POLICY",
        }
        for p in proposals
        if p["reviewed_terms"] is not None
    ]
    accepted_columns = [
        "supplement_hash",
        "review_event_id",
        "canonical_ts_code",
        "source_row_hashes",
        "review_decision_hash",
        *MATERIAL,
        "qualification_only",
        "execution_authorized",
        "cash_tax_semantics",
    ]
    return {
        "economic_event_review_queue.parquet": pd.DataFrame(queue),
        "reviewed_corporate_actions.parquet": pd.DataFrame(accepted, columns=accepted_columns),
    }, {
        "field_proposals.json": {"proposals": proposals},
        "summary.json": summary,
        "review_template.json": template,
    }


def _identity(parent: Path, body: dict[str, Any]) -> dict[str, Any]:
    semantic = dict(body)
    semantic["decisions"] = [
        Decision.model_validate(d).model_dump(exclude={"review_timestamp"})
        for d in body.get("decisions", [])
    ]
    return dict(
        contract=CONTRACT,
        parent_id=parent.name,
        parent_manifest_hash=file_hash(parent / "manifest.json"),
        proposal_hash=payload_hash(semantic),
        implementation_hash=file_hash(Path(__file__)),
        evidence_implementation_hash=implementation_hash(),
        qualification_only=True,
        execution_authorized=False,
    )


def _table(frame: pd.DataFrame) -> pd.DataFrame:
    table = review_table(frame)
    if "qualification_only" in frame:
        table["qualification_only"] = frame["qualification_only"].astype("boolean")
    return table


def _validate(path: Path, parent: Path) -> dict[str, Any]:
    m = validate_envelope(path, KIND)
    body = read_json(path / "supplements.json")
    require(m["logical_identity"] == _identity(parent, body), "SUPPLEMENT_IDENTITY")
    tables, objects = derive(parent, body, path)
    expected = (
        set(tables) | set(objects) | {"supplements.json"} | {d["file"] for d in body["documents"]}
    )
    require(set(m["artifact_hashes"]) == expected, "SUPPLEMENT_FILE_SET")
    for name, table in tables.items():
        require(
            content_hash(pd.read_parquet(path / name)) == content_hash(_table(table)),
            "SUPPLEMENT_BUSINESS_MISMATCH",
        )
    for name, value in objects.items():
        require(read_json(path / name) == value, "SUPPLEMENT_BUSINESS_MISMATCH")
    return m


def compile_supplements(parent: Path, expected_hash: str, input_json: Path, output: Path) -> Path:
    require(file_hash(parent / "manifest.json") == expected_hash, "SUPPLEMENT_PARENT_HASH")
    validate_reviewed_economic_evidence_artifact(parent)
    body = read_json(input_json)
    tables, objects = derive(parent, body, input_json.parent)
    documents = {}
    for raw in body["documents"]:
        d = Document.model_validate(raw)
        require(d.file.startswith("documents/"), "SUPPLEMENT_DOCUMENT_PATH")
        documents[d.file] = safe_file(input_json.parent, d.file).read_bytes()
    documents.update({n: _table(t).to_parquet(index=False) for n, t in tables.items()})
    return publish(
        output,
        KIND,
        _identity(parent, body),
        {"parent": str(parent.resolve())},
        {},
        objects | {"supplements.json": body},
        lambda p: _validate(p, parent),
        documents,
    )


def validate_supplements(path: Path) -> dict[str, Any]:
    m = validate_envelope(path, KIND)
    parent = Path(m["locators"]["parent"])
    require(
        file_hash(parent / "manifest.json") == m["logical_identity"]["parent_manifest_hash"],
        "SUPPLEMENT_PARENT_HASH",
    )
    validate_reviewed_economic_evidence_artifact(parent)
    return _validate(path, parent)
