"""Offline evidence fixtures; no real reviewed resolution is published here."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.backtest.continuous_source import file_hash, payload_hash, read_json
from ashare_quant.cli import main
from ashare_quant.data.economic_event_revision_resolution import (
    _provider_pay_date_unresolved,
    compatible_revision_implementation,
    compile_revision_resolutions,
    derive,
    validate_revision_resolution_artifact,
)
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.utils.manifest import atomic_write_json
from test_economic_event_evidence import dividend, source  # noqa: F401
from test_economic_event_supplements import add_conflicting_supplement, publish, setup  # noqa: F401


def _make_conflict_fixture(setup_fn, tmp_path, pay_dates):
    from ashare_quant.data.economic_event_revision_resolution import CONTRACT

    rows = [
        dividend(
            cash_div=None,
            cash_div_tax=None,
            stk_bo_rate=None,
            stk_co_rate=0.49,
            stk_div=0.49,
            pay_date=pay_dates[0],
            div_listdate="20240104",
        ),
        dividend(
            cash_div=0.0,
            cash_div_tax=0.0,
            stk_bo_rate=None,
            stk_co_rate=0.49,
            stk_div=0.49,
            pay_date=pay_dates[1],
            div_listdate="20240104",
            ann_date="20231225",
        ),
    ]
    d2, supplement_input = setup_fn(rows)
    add_conflicting_supplement(d2, supplement_input)
    html = tmp_path / "documents/fixture.html"
    html.write_text(
        "<html><p>AAA 2023年年度 20240103 20240104 20240102 实施。"
        "Cash after tax 0. Cash before tax 0. No bonus shares. "
        "Reserve per share 0.49. Total distribution per share 0.49.</p></html>"
    )
    doc_hash = file_hash(html)
    supplement_input["documents"][0]["sha256"] = doc_hash
    for item in supplement_input["supplements"]:
        for citation in [*item["identity_citations"], *item["claims"]]:
            citation["document_sha256"] = doc_hash
        item["identity_citations"][0]["quote"] = "AAA 2023年年度 20240103 20240104 20240102"
        for claim in item["claims"]:
            claim["quote"] = "No bonus shares."
    parent = publish(d2, supplement_input, tmp_path)
    q = pd.read_parquet(d2 / "economic_event_review_queue.parquet").iloc[0]
    groups = pd.read_parquet(d2 / "provider_revision_groups.parquet")
    exact = json.loads(q.exact_candidate_group_ids)
    source_hashes = sorted(
        h
        for _, group in groups[groups.candidate_group_id.isin(exact)].iterrows()
        for h in json.loads(group.source_row_hashes)
    )
    relevant = pd.read_parquet(d2 / "relevant_provider_rows.parquet")
    request_id = relevant[relevant.provider_raw_row_hash.isin(source_hashes)].request_id.iloc[0]
    linked = read_json(parent / "field_proposals.json")["proposals"]
    quotes = {
        "cash_div": "Cash after tax 0.",
        "cash_div_tax": "Cash before tax 0.",
        "stk_bo_rate": "No bonus shares.",
        "stk_co_rate": "Reserve per share 0.49.",
        "stk_div": "Total distribution per share 0.49.",
    }
    terms = [
        dict(
            field=field,
            value=value,
            disclosed_value=value,
            basis="EXPLICIT_PER_SHARE",
            document_sha256=doc_hash,
            page=1,
            quote=quotes[field],
        )
        for field, value in (
            ("cash_div", 0.0),
            ("cash_div_tax", 0.0),
            ("stk_bo_rate", 0.0),
            ("stk_co_rate", 0.49),
            ("stk_div", 0.49),
        )
    ]
    proposal = dict(
        contract=CONTRACT,
        parent_artifact_id=parent.name,
        parent_manifest_hash=file_hash(parent / "manifest.json"),
        review_event_id=q.review_event_id,
        canonical_ts_code="AAA.SZ",
        end_date="20231231",
        record_date="20240103",
        ex_date="20240104",
        imp_ann_date="20240102",
        div_proc="实施",
        resolution_mode="OFFICIAL_FIELD_RECONCILIATION_NO_PROVIDER_SUPERSESSION",
        provider_supersession="UNKNOWN",
        candidate_group_ids=exact,
        source_row_hashes=source_hashes,
        supplement_hashes=[p["supplement_hash"] for p in linked],
        request_id=request_id,
        document_sha256=doc_hash,
        document_id="fixture",
        identity_citations=[
            dict(
                document_sha256=doc_hash, page=1, quote="AAA 2023年年度 20240103 20240104 20240102"
            )
        ],
        authoritative_terms=terms,
        unresolved_fields=["pay_date"],
        qualification_only=True,
        execution_authorized=False,
    )
    return parent, dict(
        contract=CONTRACT,
        parent_artifact_id=parent.name,
        parent_manifest_hash=file_hash(parent / "manifest.json"),
        resolutions=[proposal],
        decisions=[],
    )


@pytest.fixture
def conflict_fixture(setup, tmp_path):  # noqa: F811
    return _make_conflict_fixture(setup, tmp_path, (None, None))


def test_proposal_and_approved_fixture_are_distinct(conflict_fixture, tmp_path):
    parent, body = conflict_fixture

    input_file = tmp_path / "resolution_input.json"
    atomic_write_json(input_file, body)
    proposal = compile_revision_resolutions(
        parent, file_hash(parent / "manifest.json"), input_file, tmp_path / "results"
    )
    validate_revision_resolution_artifact(proposal)
    assert read_json(proposal / "summary.json")["economic_event_evidence_status"] == "BLOCKED"
    assert read_json(proposal / "review_template.json")["decisions"][0]["decision"] is None
    resolution_hash = read_json(proposal / "resolution_proposals.json")["proposals"][0][
        "resolution_hash"
    ]
    body["decisions"] = [
        dict(
            resolution_hash=resolution_hash,
            decision="APPROVE_OFFICIAL_RECONCILIATION_AS_EVIDENCE",
            reviewed_by="fixture reviewer",
            review_reason="Fixture official statement reviewed",
        )
    ]
    atomic_write_json(input_file, body)
    approved = compile_revision_resolutions(
        parent, file_hash(parent / "manifest.json"), input_file, tmp_path / "results"
    )
    assert approved != proposal
    validate_revision_resolution_artifact(approved)
    accepted = pd.read_parquet(approved / "reviewed_revision_resolutions.parquet")
    assert len(accepted) == 1
    assert accepted.iloc[0].provider_supersession == "UNKNOWN"
    assert not accepted.iloc[0].execution_authorized

    observations = read_json(approved / "provider_observations.json")["resolutions"][0]
    assert observations["provider_supersession"] == "UNKNOWN"
    cash_values = sorted(
        (json.loads(r["raw_row_json"])["cash_div"] for r in observations["observations"]),
        key=lambda value: (value is not None, value or 0),
    )
    assert cash_values == [None, 0.0]


def _input_path(tmp_path, body):
    path = tmp_path / "resolution_input.json"
    atomic_write_json(path, body)
    return path


def _decision(parent, body, kind="APPROVE_OFFICIAL_RECONCILIATION_AS_EVIDENCE"):
    proposal = derive(parent, body)[1]["resolution_proposals.json"]["proposals"][0]
    return dict(
        resolution_hash=proposal["resolution_hash"],
        decision=kind,
        reviewed_by="fixture reviewer",
        review_reason="Fixture review only",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_group_ids", ["missing"]),
        ("source_row_hashes", ["missing"]),
        ("supplement_hashes", ["missing"]),
        ("review_event_id", "wrong"),
        ("canonical_ts_code", "BBB.SZ"),
        ("end_date", "20221231"),
        ("record_date", "20240102"),
        ("ex_date", "20240105"),
        ("parent_manifest_hash", "f" * 64),
        ("provider_supersession", "LATEST_ROW_WINS"),
    ],
)
def test_candidate_and_identity_mutations_fail_closed(conflict_fixture, tmp_path, field, value):
    parent, body = conflict_fixture
    body["resolutions"][0][field] = value
    with pytest.raises((DataValidationError, ValueError)):
        compile_revision_resolutions(
            parent,
            file_hash(parent / "manifest.json"),
            _input_path(tmp_path, body),
            tmp_path / "out",
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(candidate_group_ids=p["candidate_group_ids"][:1]),
        lambda p: p.update(source_row_hashes=p["source_row_hashes"][:1]),
        lambda p: p.update(supplement_hashes=p["supplement_hashes"][:1]),
        lambda p: p.update(candidate_group_ids=p["candidate_group_ids"] + ["unrelated"]),
        lambda p: p.update(resolution_mode="LATEST_ROW_WINS"),
        lambda p: p.update(selected_provider_row_hash=p["source_row_hashes"][-1]),
        lambda p: p.update(unresolved_fields=[]),
        lambda p: p.update(unresolved_fields=["pay_date", "cash_div"]),
        lambda p: p["authoritative_terms"].append(deepcopy(p["authoritative_terms"][0])),
        lambda p: p["authoritative_terms"][0].update(disclosed_value=1),
        lambda p: p["authoritative_terms"][0].update(quote=" \t\n\u3000"),
        lambda p: p["authoritative_terms"][0].update(quote="Not in document"),
        lambda p: p["authoritative_terms"][0].update(page=2),
        lambda p: p["authoritative_terms"][0].update(document_sha256="f" * 64),
        lambda p: p["identity_citations"][0].update(quote="  \n\t"),
        lambda p: p["identity_citations"][0].update(quote="Wrong event"),
        lambda p: p["identity_citations"][0].update(page=2),
    ],
)
def test_evidence_mutations_fail_closed(conflict_fixture, tmp_path, mutate):
    parent, body = conflict_fixture
    mutate(body["resolutions"][0])
    with pytest.raises((DataValidationError, ValueError)):
        compile_revision_resolutions(
            parent,
            file_hash(parent / "manifest.json"),
            _input_path(tmp_path, body),
            tmp_path / "out",
        )


@pytest.mark.parametrize("kind", ["REJECT_NOT_SAME_EVENT", "REQUIRE_OFFICIAL_DOCUMENT"])
def test_nonapproval_decisions_remain_qualified_only(conflict_fixture, tmp_path, kind):
    parent, body = conflict_fixture
    body["decisions"] = [_decision(parent, body, kind)]
    result = compile_revision_resolutions(
        parent, file_hash(parent / "manifest.json"), _input_path(tmp_path, body), tmp_path / "out"
    )
    validate_revision_resolution_artifact(result)
    assert pd.read_parquet(result / "reviewed_revision_resolutions.parquet").empty
    assert read_json(result / "review_template.json")["decisions"] == []
    assert read_json(result / "summary.json")["economic_event_evidence_status"] == "BLOCKED"


def test_decision_binding_idempotence_and_timestamp(conflict_fixture, tmp_path):
    parent, body = conflict_fixture
    body["decisions"] = [_decision(parent, body)]
    path = _input_path(tmp_path, body)
    result = compile_revision_resolutions(
        parent, file_hash(parent / "manifest.json"), path, tmp_path / "out"
    )
    body["decisions"][0]["review_timestamp"] = "2026-09-30T00:00:00Z"
    atomic_write_json(path, body)
    assert (
        compile_revision_resolutions(
            parent, file_hash(parent / "manifest.json"), path, tmp_path / "out"
        )
        == result
    )
    for key in ("reviewed_by", "review_reason", "decision"):
        changed = deepcopy(body)
        changed["decisions"][0][key] = (
            "REQUIRE_OFFICIAL_DOCUMENT" if key == "decision" else "different fixture review"
        )
        atomic_write_json(path, changed)
        assert (
            compile_revision_resolutions(
                parent, file_hash(parent / "manifest.json"), path, tmp_path / "out"
            )
            != result
        )


@pytest.mark.parametrize("mutation", ["raw", "term", "citation", "document"])
def test_stale_decision_does_not_follow_changed_proposal(conflict_fixture, tmp_path, mutation):
    parent, body = conflict_fixture
    body["decisions"] = [_decision(parent, body)]
    p = body["resolutions"][0]
    if mutation == "raw":
        p["source_row_hashes"][0] = "a" * 64
    elif mutation == "term":
        p["authoritative_terms"][0]["value"] = 0.1
    elif mutation == "citation":
        p["authoritative_terms"][0]["quote"] = "Cash before tax 0."
    else:
        p["document_sha256"] = "a" * 64
    with pytest.raises((DataValidationError, ValueError)):
        compile_revision_resolutions(
            parent,
            file_hash(parent / "manifest.json"),
            _input_path(tmp_path, body),
            tmp_path / "out",
        )


@pytest.mark.parametrize("mutation", ["reviewed_by", "review_reason", "duplicate"])
def test_invalid_decisions_fail(conflict_fixture, tmp_path, mutation):
    parent, body = conflict_fixture
    body["decisions"] = [_decision(parent, body)]
    if mutation == "duplicate":
        body["decisions"].append(deepcopy(body["decisions"][0]))
    else:
        body["decisions"][0][mutation] = " \t"
    with pytest.raises((DataValidationError, ValueError)):
        compile_revision_resolutions(
            parent,
            file_hash(parent / "manifest.json"),
            _input_path(tmp_path, body),
            tmp_path / "out",
        )


def test_incomplete_official_amounts_cannot_be_approved(conflict_fixture, tmp_path):
    parent, body = conflict_fixture
    proposal = body["resolutions"][0]
    proposal["authoritative_terms"] = [
        term for term in proposal["authoritative_terms"] if term["field"] != "cash_div_tax"
    ]
    proposal["unresolved_fields"] = ["pay_date", "cash_div_tax"]
    body["decisions"] = [_decision(parent, body)]
    with pytest.raises(DataValidationError, match="REVISION_NOT_APPROVABLE"):
        compile_revision_resolutions(
            parent,
            file_hash(parent / "manifest.json"),
            _input_path(tmp_path, body),
            tmp_path / "out",
        )


def test_candidate_order_is_nonlogical(conflict_fixture, tmp_path):
    parent, body = conflict_fixture
    path = _input_path(tmp_path, body)
    first = compile_revision_resolutions(
        parent, file_hash(parent / "manifest.json"), path, tmp_path / "out"
    )
    proposal = body["resolutions"][0]
    for key in ("candidate_group_ids", "source_row_hashes", "supplement_hashes"):
        proposal[key].reverse()
    proposal["authoritative_terms"].reverse()
    atomic_write_json(path, body)
    second = compile_revision_resolutions(
        parent, file_hash(parent / "manifest.json"), path, tmp_path / "out"
    )
    assert first == second


def test_rehashed_derived_child_fails_business_validation(conflict_fixture, tmp_path):
    parent, body = conflict_fixture
    result = compile_revision_resolutions(
        parent, file_hash(parent / "manifest.json"), _input_path(tmp_path, body), tmp_path / "out"
    )
    child = result / "authoritative_event_terms.json"
    value = read_json(child)
    value["events"][0]["terms"][0]["value"] = 99
    atomic_write_json(child, value)
    manifest = read_json(result / "manifest.json")
    manifest["artifact_hashes"][child.name] = file_hash(child)
    atomic_write_json(result / "manifest.json", manifest)
    with pytest.raises(DataValidationError):
        validate_revision_resolution_artifact(result)


def test_ordinary_supplement_conflict_gate_still_rejects_approval(conflict_fixture, tmp_path):
    from ashare_quant.data.economic_event_supplements import compile_supplements

    parent, _ = conflict_fixture
    input_body = read_json(parent / "supplements.json")
    proposal = read_json(parent / "field_proposals.json")["proposals"][0]
    assert proposal["conflict"]
    input_body["decisions"] = [
        dict(
            supplement_hash=proposal["supplement_hash"],
            decision="APPROVE_AS_EVIDENCE",
            reviewed_by="fixture reviewer",
            review_reason="Fixture only",
        )
    ]
    path = tmp_path / "ordinary.json"
    atomic_write_json(path, input_body)
    d2 = Path(read_json(parent / "manifest.json")["locators"]["parent"])
    with pytest.raises(DataValidationError, match="SUPPLEMENT_NOT_APPROVABLE"):
        compile_supplements(d2, input_body["parent_manifest_hash"], path, tmp_path / "ordinary")


def test_cli_compiles_and_validates_offline_fixture(conflict_fixture, tmp_path, capsys):
    parent, body = conflict_fixture
    path = _input_path(tmp_path, body)
    assert (
        main(
            [
                "data",
                "economic-review-revision-resolution",
                "--parent",
                str(parent),
                "--parent-manifest-sha256",
                file_hash(parent / "manifest.json"),
                "--resolutions",
                str(path),
                "--output-root",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)["artifact"]
    assert (
        main(
            [
                "data",
                "economic-review-revision-resolution-validate",
                "--artifact",
                result,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["validation"] == "PASS"


@pytest.mark.parametrize(
    "pay_dates,unresolved",
    [
        ((None, None), True),
        ((None, "20240105"), True),
        (("20240105", None), True),
        (("20240105", "20240106"), True),
        (("20240105", "20240105"), False),
    ],
)
def test_provider_pay_date_conflict_policy(pay_dates, unresolved):
    observations = [{"raw_row_json": json.dumps({"pay_date": date})} for date in pay_dates]
    assert _provider_pay_date_unresolved(observations) is unresolved


@pytest.mark.parametrize(
    "pay_dates,unresolved",
    [
        ((None, None), True),
        ((None, "20240105"), True),
        (("20240105", None), True),
        (("20240105", "20240106"), True),
        (("20240105", "20240105"), False),
    ],
)
def test_provider_pay_dates_reconcile_through_compiler(setup, tmp_path, pay_dates, unresolved):  # noqa: F811
    parent, body = _make_conflict_fixture(setup, tmp_path, pay_dates)
    body["resolutions"][0]["unresolved_fields"] = ["pay_date"] if unresolved else []
    path = compile_revision_resolutions(
        parent, file_hash(parent / "manifest.json"), _input_path(tmp_path, body), tmp_path / "out"
    )
    validate_revision_resolution_artifact(path)
    proposal = read_json(path / "resolution_proposals.json")["proposals"][0]["proposal"]
    assert ("pay_date" in proposal["unresolved_fields"]) is unresolved


@pytest.mark.parametrize("invalid", ["20240230", "2024010X", "", "99991231"])
def test_invalid_pay_date_fails_closed(invalid):
    observations = [
        {"raw_row_json": json.dumps({"pay_date": "20240105"})},
        {"raw_row_json": json.dumps({"pay_date": invalid})},
    ]
    if invalid == "":
        assert _provider_pay_date_unresolved(observations)
    else:
        with pytest.raises(DataValidationError, match="INVALID_DATE"):
            _provider_pay_date_unresolved(observations)


def _legacy_fingerprint():
    fixture = Path(__file__).parent / "fixtures/economic_revision_v1_legacy_implementation.json"
    value = read_json(fixture)
    assert payload_hash(value["component_sha256"]) == value["implementation_hash"]
    return value["implementation_hash"]


def _reidentify_artifact(artifact, digest, target_root):
    manifest = read_json(artifact / "manifest.json")
    manifest["logical_identity"]["implementation_hash"] = digest
    artifact_id = (
        "economic_event_revision_resolution_" + payload_hash(manifest["logical_identity"])[:24]
    )
    target = target_root / artifact_id
    shutil.copytree(artifact, target)
    manifest["artifact_id"] = artifact_id
    atomic_write_json(target / "manifest.json", manifest)
    return target


@pytest.fixture
def legacy_artifact(conflict_fixture, tmp_path):
    parent, body = conflict_fixture
    current = compile_revision_resolutions(
        parent, file_hash(parent / "manifest.json"), _input_path(tmp_path, body), tmp_path / "out"
    )
    validate_revision_resolution_artifact(current)
    legacy = _reidentify_artifact(current, _legacy_fingerprint(), tmp_path / "legacy")
    return parent, current, legacy


def test_pinned_legacy_implementation_recursively_validates(legacy_artifact):
    _, current, legacy = legacy_artifact
    assert validate_revision_resolution_artifact(current)["status"] == "COMPLETE"
    assert compatible_revision_implementation(_legacy_fingerprint())
    assert validate_revision_resolution_artifact(legacy)["status"] == "COMPLETE"


def test_unknown_implementation_hash_fails_closed(legacy_artifact, tmp_path):
    _, current, _ = legacy_artifact
    unknown = _reidentify_artifact(current, "f" * 64, tmp_path / "unknown")
    assert not compatible_revision_implementation("f" * 64)
    with pytest.raises(DataValidationError, match="REVISION_IMPLEMENTATION_UNSUPPORTED"):
        validate_revision_resolution_artifact(unknown)


def test_legacy_child_rehash_cannot_bypass_business_reconstruction(legacy_artifact):
    _, _, legacy = legacy_artifact
    child = legacy / "authoritative_event_terms.json"
    value = read_json(child)
    value["events"][0]["terms"][0]["value"] = 99
    atomic_write_json(child, value)
    manifest = read_json(legacy / "manifest.json")
    manifest["artifact_hashes"][child.name] = file_hash(child)
    atomic_write_json(legacy / "manifest.json", manifest)
    with pytest.raises(DataValidationError, match="REVISION_BUSINESS_MISMATCH"):
        validate_revision_resolution_artifact(legacy)


@pytest.mark.parametrize("tamper", ["parent", "raw", "document"])
def test_legacy_path_still_validates_parent_raw_and_documents(legacy_artifact, tamper):
    parent, _, legacy = legacy_artifact
    if tamper == "parent":
        child = parent / "summary.json"
        value = read_json(child)
        value["qualification_only"] = False
        atomic_write_json(child, value)
    elif tamper == "raw":
        d2 = Path(read_json(parent / "manifest.json")["locators"]["parent"])
        child = d2 / "relevant_provider_rows.parquet"
        frame = pd.read_parquet(child)
        frame.loc[0, "raw_row_json"] = "{}"
        frame.to_parquet(child, index=False)
    else:
        child = parent / "documents/fixture.html"
        child.write_text("altered fixture document")
    with pytest.raises(DataValidationError):
        validate_revision_resolution_artifact(legacy)
