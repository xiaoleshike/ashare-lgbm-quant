from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.backtest.continuous_source import file_hash, payload_hash, read_json
from ashare_quant.cli import main
from ashare_quant.data import economic_event_share_field_qualification as fields
from ashare_quant.data.economic_event_supplements import compile_supplements
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.utils.manifest import atomic_write_json
from test_economic_event_evidence import dividend, source  # noqa: F401
from test_economic_event_supplements import publish as publish_supplement
from test_economic_event_supplements import setup  # noqa: F401


@pytest.fixture
def parent_and_input(request, tmp_path):
    make = request.getfixturevalue("setup")
    d2, supplement = make(
        [
            dividend(
                stk_div=0.0,
                stk_bo_rate=None,
                stk_co_rate=None,
                cash_div=0.8,
                cash_div_tax=0.8,
            )
        ]
    )
    parent = publish_supplement(d2, supplement, tmp_path)
    body = {
        "contract": fields.CONTRACT,
        "parent_artifact_id": parent.name,
        "parent_manifest_hash": file_hash(parent / "manifest.json"),
        "supplement_hash": payload_hash(supplement["supplements"][0]),
        "decisions": [],
    }
    return parent, body


def compile_fields(parent, body, tmp_path):
    input_path = tmp_path / "fields.json"
    atomic_write_json(input_path, body)
    return fields.compile_share_field_qualification(
        parent, file_hash(parent / "manifest.json"), input_path, tmp_path / "field_output"
    )


def approved(body, claim_hash, *, decision="APPROVE_FIELD_AS_EVIDENCE"):
    body["decisions"].append(
        {
            "field_claim_hash": claim_hash,
            "decision": decision,
            "reviewed_by": "fixture reviewer",
            "review_reason": "Synthetic official statement checked",
        }
    )


def test_proposal_only_and_two_independent_claims(parent_and_input, tmp_path):
    parent, body = parent_and_input
    original = file_hash(parent / "manifest.json")
    out = compile_fields(parent, body, tmp_path)
    fields.validate_share_field_qualification(out)
    proposals = read_json(out / "field_proposals.json")["proposals"]
    assert {p["field"] for p in proposals} == {"stk_bo_rate", "stk_co_rate"}
    assert len({p["field_claim_hash"] for p in proposals}) == 2
    assert all(p["status"] == "PENDING_HUMAN_REVIEW" for p in proposals)
    assert len(read_json(out / "review_template.json")["decisions"]) == 2
    assert pd.read_parquet(out / "reviewed_field_qualifications.parquet").empty
    assert read_json(out / "summary.json") == {
        "contract": fields.CONTRACT,
        "field_proposals": 2,
        "pending_fields": 2,
        "reviewed_fields": 0,
        "economic_event_evidence_status": "BLOCKED",
        "qualification_only": True,
        "execution_authorized": False,
    }
    assert compile_fields(parent, body, tmp_path) == out
    assert file_hash(parent / "manifest.json") == original


@pytest.mark.parametrize("reviewed_count", [1, 2])
def test_approval_is_field_scoped_and_cash_never_reviewed(
    parent_and_input, tmp_path, reviewed_count
):
    parent, body = parent_and_input
    pending = compile_fields(parent, body, tmp_path)
    proposals = read_json(pending / "field_proposals.json")["proposals"]
    for proposal in proposals[:reviewed_count]:
        approved(body, proposal["field_claim_hash"])
    out = compile_fields(parent, body, tmp_path)
    assert out != pending
    fields.validate_share_field_qualification(out)
    reviewed = pd.read_parquet(out / "reviewed_field_qualifications.parquet")
    assert len(reviewed) == reviewed_count
    assert set(reviewed.field) <= {"stk_bo_rate", "stk_co_rate"}
    assert set(reviewed.approved_value) == {0.0}
    assert not {"cash_div", "cash_div_tax", "reviewed_terms"} & set(reviewed.columns)
    assert reviewed.qualification_only.all()
    assert not reviewed.execution_authorized.any()
    assert all(reviewed.review_decision_hash.notna())
    assert len(read_json(out / "review_template.json")["decisions"]) == 2 - reviewed_count
    summary = read_json(out / "summary.json")
    assert summary["reviewed_fields"] == reviewed_count
    assert summary["economic_event_evidence_status"] == "BLOCKED"
    assert not summary["execution_authorized"]
    assert (
        read_json(parent / "field_proposals.json")["proposals"][0]["raw_terms"]["cash_div"] == 0.8
    )


@pytest.mark.parametrize(
    "decision,status",
    [
        ("REQUIRE_OFFICIAL_DOCUMENT", "OFFICIAL_DOCUMENT_REQUIRED"),
        ("REJECT_FIELD_CLAIM", "REJECTED_FIELD_CLAIM"),
    ],
)
def test_nonapproval_status(parent_and_input, tmp_path, decision, status):
    parent, body = parent_and_input
    first = read_json(compile_fields(parent, body, tmp_path) / "field_proposals.json")["proposals"][
        0
    ]
    approved(body, first["field_claim_hash"], decision=decision)
    out = compile_fields(parent, body, tmp_path)
    proposals = read_json(out / "field_proposals.json")["proposals"]
    assert (
        next(p for p in proposals if p["field_claim_hash"] == first["field_claim_hash"])["status"]
        == status
    )
    assert pd.read_parquet(out / "reviewed_field_qualifications.parquet").empty
    assert len(read_json(out / "review_template.json")["decisions"]) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_event",
        "wrong_group",
        "wrong_raw_hash",
        "wrong_doc_hash",
        "wrong_page",
        "wrong_quote",
        "known_provider_field",
        "cash_claim",
    ],
)
def test_modified_parent_claims_fail_closed(parent_and_input, tmp_path, mutation):
    parent, body = parent_and_input
    supplement_input = read_json(parent / "supplements.json")
    supplement = supplement_input["supplements"][0]
    claim = supplement["claims"][0]
    if mutation == "wrong_event":
        supplement["review_event_id"] = "wrong"
    elif mutation == "wrong_group":
        supplement["candidate_group_id"] = "wrong"
    elif mutation == "wrong_raw_hash":
        supplement["source_row_hashes"] = ["f" * 64]
    elif mutation == "wrong_doc_hash":
        claim["document_sha256"] = "f" * 64
    elif mutation == "wrong_page":
        claim["page"] = 99
    elif mutation == "wrong_quote":
        claim["quote"] = "not in document"
    elif mutation == "known_provider_field":
        claim["field"] = "stk_div"
    elif mutation == "cash_claim":
        claim["field"] = "cash_div"
    atomic_write_json(parent / "supplements.json", supplement_input)
    manifest = read_json(parent / "manifest.json")
    manifest["artifact_hashes"]["supplements.json"] = file_hash(parent / "supplements.json")
    atomic_write_json(parent / "manifest.json", manifest)
    # The recursively validated parent rejects a rehashed input before this layer
    # can interpret it. This is stronger than accepting a forged field proposal.
    with pytest.raises(DataValidationError, match="SUPPLEMENT_IDENTITY"):
        compile_fields(parent, body, tmp_path)


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("empty_reviewer", "FIELD_REVIEW_MISSING"),
        ("wrong_hash", "FIELD_UNKNOWN_DECISION"),
        ("duplicate", "FIELD_DUPLICATE_DECISION"),
        ("cash_div_decision", "FIELD_UNKNOWN_DECISION"),
        ("cash_div_tax_decision", "FIELD_UNKNOWN_DECISION"),
    ],
)
def test_decision_fail_closed(parent_and_input, tmp_path, mutation, error):
    parent, body = parent_and_input
    first = read_json(compile_fields(parent, body, tmp_path) / "field_proposals.json")["proposals"][
        0
    ]
    approved(body, first["field_claim_hash"])
    if mutation == "empty_reviewer":
        body["decisions"][0]["reviewed_by"] = " "
    elif mutation == "wrong_hash":
        body["decisions"][0]["field_claim_hash"] = "f" * 64
    elif mutation == "duplicate":
        body["decisions"].append(deepcopy(body["decisions"][0]))
    elif mutation in {"cash_div_decision", "cash_div_tax_decision"}:
        body["decisions"][0]["field_claim_hash"] = payload_hash(
            {"field": mutation.removesuffix("_decision")}
        )
    with pytest.raises(DataValidationError, match=error):
        compile_fields(parent, body, tmp_path)


def test_changed_claim_invalidates_old_decision(parent_and_input, tmp_path):
    parent, body = parent_and_input
    first = read_json(compile_fields(parent, body, tmp_path) / "field_proposals.json")["proposals"][
        0
    ]
    approved(body, first["field_claim_hash"])
    supplement_input = read_json(parent / "supplements.json")
    supplement_input["supplements"][0]["claims"][0]["quote"] = "No bonus shares"
    atomic_write_json(parent / "supplements.json", supplement_input)
    manifest = read_json(parent / "manifest.json")
    manifest["artifact_hashes"]["supplements.json"] = file_hash(parent / "supplements.json")
    atomic_write_json(parent / "manifest.json", manifest)
    with pytest.raises(DataValidationError):
        compile_fields(parent, body, tmp_path)


def test_valid_new_parent_does_not_inherit_old_field_decision(parent_and_input, tmp_path):
    parent, body = parent_and_input
    old_claim = read_json(compile_fields(parent, body, tmp_path) / "field_proposals.json")[
        "proposals"
    ][0]
    approved(body, old_claim["field_claim_hash"])
    supplement_input = read_json(parent / "supplements.json")
    supplement_input["supplements"][0]["claims"][0]["quote"] = "No bonus shares"
    d2 = read_json(parent / "manifest.json")["locators"]["parent"]
    new_input = tmp_path / "new_supplement.json"
    atomic_write_json(new_input, supplement_input)
    new_parent = compile_supplements(
        Path(d2), file_hash(Path(d2) / "manifest.json"), new_input, tmp_path / "new_parent"
    )
    body.update(
        parent_artifact_id=new_parent.name,
        parent_manifest_hash=file_hash(new_parent / "manifest.json"),
        supplement_hash=payload_hash(supplement_input["supplements"][0]),
    )
    with pytest.raises(DataValidationError, match="FIELD_UNKNOWN_DECISION"):
        compile_fields(new_parent, body, tmp_path)


def test_rehashed_child_still_fails_business_validation(parent_and_input, tmp_path):
    parent, body = parent_and_input
    out = compile_fields(parent, body, tmp_path)
    child = out / "field_proposals.json"
    value = read_json(child)
    value["proposals"][0]["value"] = 9
    atomic_write_json(child, value)
    manifest = read_json(out / "manifest.json")
    manifest["artifact_hashes"][child.name] = file_hash(child)
    atomic_write_json(out / "manifest.json", manifest)
    with pytest.raises(DataValidationError, match="FIELD_BUSINESS_MISMATCH"):
        fields.validate_share_field_qualification(out)


def test_rehashed_reviewed_table_still_fails(parent_and_input, tmp_path):
    parent, body = parent_and_input
    pending = compile_fields(parent, body, tmp_path)
    first = read_json(pending / "field_proposals.json")["proposals"][0]
    approved(body, first["field_claim_hash"])
    out = compile_fields(parent, body, tmp_path)
    child = out / "reviewed_field_qualifications.parquet"
    frame = pd.read_parquet(child)
    frame.loc[0, "approved_value"] = 9.0
    frame.to_parquet(child, index=False)
    manifest = read_json(out / "manifest.json")
    manifest["artifact_hashes"][child.name] = file_hash(child)
    atomic_write_json(out / "manifest.json", manifest)
    with pytest.raises(DataValidationError, match="FIELD_BUSINESS_MISMATCH"):
        fields.validate_share_field_qualification(out)


def test_parent_document_tamper_rejected(parent_and_input, tmp_path):
    parent, body = parent_and_input
    out = compile_fields(parent, body, tmp_path)
    document = parent / "documents/fixture.html"
    document.write_text("forged document")
    parent_manifest = read_json(parent / "manifest.json")
    parent_manifest["artifact_hashes"]["documents/fixture.html"] = file_hash(document)
    atomic_write_json(parent / "manifest.json", parent_manifest)
    with pytest.raises(DataValidationError):
        fields.validate_share_field_qualification(out)


def test_offline_cli_consumer(parent_and_input, tmp_path, capsys, monkeypatch):
    parent, body = parent_and_input
    input_path = tmp_path / "cli_input.json"
    atomic_write_json(input_path, body)

    def no_network(*args, **kwargs):
        raise AssertionError("network access in offline evidence qualification")

    monkeypatch.setattr("requests.sessions.Session.request", no_network)
    assert (
        main(
            [
                "data",
                "economic-review-share-field-qualification",
                "--parent",
                str(parent),
                "--parent-manifest-sha256",
                file_hash(parent / "manifest.json"),
                "--fields",
                str(input_path),
                "--output-root",
                str(tmp_path / "cli_output"),
            ]
        )
        == 0
    )
    artifact = json.loads(capsys.readouterr().out)["artifact"]
    assert (
        main(["data", "economic-review-share-field-qualification-validate", "--artifact", artifact])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["validation"] == "PASS"


def test_unknown_producer_and_timestamp_identity(parent_and_input, tmp_path):
    parent, body = parent_and_input
    pending = compile_fields(parent, body, tmp_path)
    first = read_json(pending / "field_proposals.json")["proposals"][0]
    approved(body, first["field_claim_hash"])
    approved_path = compile_fields(parent, body, tmp_path)
    body["decisions"][0]["review_timestamp"] = "2026-10-12T12:00:00Z"
    assert compile_fields(parent, body, tmp_path) == approved_path
    manifest = read_json(approved_path / "manifest.json")
    manifest["logical_identity"]["implementation_hash"] = "f" * 64
    forged_id = fields.KIND + "_" + payload_hash(manifest["logical_identity"])[:24]
    manifest["artifact_id"] = forged_id
    atomic_write_json(approved_path / "manifest.json", manifest)
    forged = approved_path.rename(approved_path.parent / forged_id)
    with pytest.raises(DataValidationError, match="FIELD_IMPLEMENTATION_UNSUPPORTED"):
        fields.validate_share_field_qualification(forged)
