from __future__ import annotations

import json

import pandas as pd
import pytest

from ashare_quant.backtest.continuous_source import file_hash, payload_hash, read_json
from ashare_quant.cli import main
from ashare_quant.data import economic_event_evidence as evidence
from ashare_quant.data import economic_event_sources as sources
from ashare_quant.data import economic_event_supplements as supplements
from ashare_quant.data import economic_review_artifacts as reviews
from ashare_quant.data.economic_event_review import MATERIAL
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.utils.manifest import atomic_write_json
from test_economic_event_evidence import (
    Provider,
    dividend,
    source,  # noqa: F401
    terminal_package,
)


@pytest.fixture
def setup(source, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(reviews, "load_exposure_source", lambda *a, **kw: source)

    def make(rows=None):
        provider = sources.probe_dividends(
            source,
            Provider(rows or [dividend(stk_bo_rate=None, stk_co_rate=None)]),
            tmp_path / "provider",
        )
        initial = evidence.compile_economic_evidence(source, provider, tmp_path / "initial")
        inputs = reviews.load_review_inputs(
            initial,
            provider,
            file_hash(initial / "manifest.json"),
            file_hash(provider / "manifest.json"),
        )
        parent = reviews.compile_review(inputs, tmp_path / "review")
        queue = pd.read_parquet(parent / "economic_event_review_queue.parquet")
        q = queue.iloc[0]
        gid = json.loads(q.candidate_group_ids)[0]
        groups = pd.read_parquet(parent / "provider_revision_groups.parquet")
        g = groups[groups.candidate_group_id.eq(gid)].iloc[0]
        docs = tmp_path / "documents"
        docs.mkdir(exist_ok=True)
        doc = docs / "fixture.html"
        doc.write_text(
            "<html><p>AAA 20240103 20240104 implemented. "
            "No bonus shares and no reserve conversion. "
            "Bonus per share 0.2; reserve per share 0.3; "
            "cash after tax 0.08; before tax 0.1.</p></html>"
        )
        dh = file_hash(doc)
        claim = dict(
            document_sha256=dh,
            page=1,
            quote="No bonus shares and no reserve conversion.",
            value=0.0,
            disclosed_value=0.0,
            basis="EXPLICIT_ZERO",
        )
        claims = [dict(claim, field=k) for k in ("stk_bo_rate", "stk_co_rate") if pd.isna(g[k])]
        s = dict(
            review_event_id=q.review_event_id,
            candidate_group_id=gid,
            canonical_ts_code="AAA.SZ",
            record_date=g.record_date,
            ex_date=g.ex_date,
            source_row_hashes=json.loads(g.source_row_hashes),
            identity_citations=[dict(document_sha256=dh, page=1, quote="AAA 20240103 20240104")],
            claims=claims,
        )
        body = dict(
            contract=supplements.CONTRACT,
            parent_manifest_hash=file_hash(parent / "manifest.json"),
            documents=[
                dict(
                    file="documents/fixture.html",
                    sha256=dh,
                    url="https://www.szse.cn/fixture",
                    document_id="fixture",
                    format="HTML",
                )
            ],
            supplements=[s],
            decisions=[],
        )
        return parent, body

    return make


def publish(parent, body, tmp_path):
    path = tmp_path / "input.json"
    atomic_write_json(path, body)
    return supplements.compile_supplements(
        parent, file_hash(parent / "manifest.json"), path, tmp_path / "output"
    )


def approve(body):
    body["decisions"] = [
        dict(
            supplement_hash=payload_hash(body["supplements"][0]),
            decision="APPROVE_AS_EVIDENCE",
            reviewed_by="fixture human",
            review_reason="Fixture document reviewed",
        )
    ]


def test_cash_nulls_do_not_approve_and_parent_unchanged(setup, tmp_path):
    parent, body = setup()
    before = {p.name: file_hash(p) for p in parent.iterdir() if p.is_file()}
    output = publish(parent, body, tmp_path)
    supplements.validate_supplements(output)
    p = read_json(output / "field_proposals.json")["proposals"][0]
    assert p["raw_terms"]["stk_bo_rate"] is None
    assert p["proposed_terms"]["stk_bo_rate"] == 0
    assert p["reviewed_terms"] is None and not p["execution_authorized"]
    assert publish(parent, body, tmp_path) == output
    old = pd.read_parquet(parent / "economic_event_review_queue.parquet")
    new = pd.read_parquet(output / "economic_event_review_queue.parquet")
    pd.testing.assert_frame_equal(old, new[old.columns])
    assert before == {p.name: file_hash(p) for p in parent.iterdir() if p.is_file()}
    approve(body)
    accepted = publish(parent, body, tmp_path)
    assert accepted != output
    assert (
        read_json(accepted / "field_proposals.json")["proposals"][0]["reviewed_terms"][
            "stk_bo_rate"
        ]
        == 0
    )
    assert read_json(accepted / "summary.json")["economic_event_evidence_status"] == "BLOCKED"


@pytest.mark.parametrize("bonus,reserve,cash", [(0.2, 0.0, 0.0), (0.0, 0.3, 0.0), (0.2, 0.3, 0.08)])
def test_share_mixed_explicit_values(setup, tmp_path, bonus, reserve, cash):
    parent, body = setup(
        [
            dividend(
                stk_div=bonus + reserve,
                stk_bo_rate=None,
                stk_co_rate=None,
                cash_div=cash,
                cash_div_tax=0.1 if cash else 0.0,
            )
        ]
    )
    for c, value in zip(body["supplements"][0]["claims"], (bonus, reserve), strict=True):
        c.update(value=value, disclosed_value=value, basis="EXPLICIT_PER_SHARE")
        if value:
            c["quote"] = (
                f"{'Bonus' if c['field'] == 'stk_bo_rate' else 'reserve'} per share {value}"
            )
    approve(body)
    output = publish(parent, body, tmp_path)
    supplements.validate_supplements(output)
    assert read_json(output / "summary.json")["supplement_reviewed_events"] == 1


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("row_hash", "RAW_HASH_SET"),
        ("wrong_code", "EXACT_IDENTITY"),
        ("nearby", "EXACT_IDENTITY"),
        ("missing_record", "EXACT_IDENTITY"),
        ("quote", "QUOTE_MISMATCH"),
        ("document_hash", "DOCUMENT_HASH"),
        ("unit", "UNIT_CONFLICT"),
        ("override_zero", "CANNOT_OVERRIDE"),
        ("decision_hash", "UNKNOWN_SUPPLEMENT"),
        ("missing_component", "NOT_APPROVABLE"),
    ],
)
def test_fail_closed_consumer(setup, tmp_path, mutation, error):
    parent, body = setup()
    s = body["supplements"][0]
    if mutation == "row_hash":
        s["source_row_hashes"] = ["f" * 64]
    elif mutation == "wrong_code":
        s["canonical_ts_code"] = "BBB.SZ"
    elif mutation == "nearby":
        s["ex_date"] = "20240105"
    elif mutation == "missing_record":
        s["record_date"] = ""
    elif mutation == "quote":
        s["claims"][0]["quote"] = "not in this official document"
    elif mutation == "document_hash":
        body["documents"][0]["sha256"] = "f" * 64
    elif mutation == "unit":
        s["claims"][0]["value"] = 2.0
    elif mutation == "override_zero":
        s["claims"][0]["field"] = "stk_div"
    elif mutation == "missing_component":
        s["claims"].pop()
    approve(body)
    if mutation == "decision_hash":
        body["decisions"][0]["supplement_hash"] = "f" * 64
    with pytest.raises(DataValidationError, match=error):
        publish(parent, body, tmp_path)


def test_conflict_cannot_be_fixed_by_null_supplement(setup, tmp_path):
    parent, body = setup(
        [
            dividend(stk_bo_rate=None, stk_co_rate=None),
            dividend(stk_bo_rate=None, stk_co_rate=None, cash_div_tax=0.2),
        ]
    )
    output = publish(parent, body, tmp_path)
    assert read_json(output / "field_proposals.json")["proposals"][0]["conflict"]
    approve(body)
    with pytest.raises(DataValidationError, match="NOT_APPROVABLE"):
        publish(parent, body, tmp_path)


@pytest.mark.parametrize("field", ["record_date", "ex_date"])
def test_source_missing_date_stays_blocked(setup, tmp_path, field):
    parent, body = setup()
    # A different exposed date cannot be authorized even by a reviewed document.
    body["supplements"][0][field] = ""
    approve(body)
    with pytest.raises(DataValidationError, match="EXACT_IDENTITY"):
        publish(parent, body, tmp_path)


@pytest.mark.parametrize(
    "child", ["field_proposals.json", "summary.json", "economic_event_review_queue.parquet"]
)
def test_rehash_cannot_hide_derived_tamper(setup, tmp_path, child):
    parent, body = setup()
    output = publish(parent, body, tmp_path)
    if child.endswith("parquet"):
        f = pd.read_parquet(output / child)
        f.loc[0, "execution_authorized"] = True
        f.to_parquet(output / child, index=False)
    else:
        value = read_json(output / child)
        if "proposals" in value:
            value["proposals"][0]["proposed_terms"]["cash_div_tax"] = 99
        else:
            value["execution_authorized"] = True
        atomic_write_json(output / child, value)
    m = read_json(output / "manifest.json")
    m["artifact_hashes"][child] = file_hash(output / child)
    atomic_write_json(output / "manifest.json", m)
    with pytest.raises(DataValidationError, match="BUSINESS_MISMATCH"):
        supplements.validate_supplements(output)


def test_changed_document_or_input_rejected(setup, tmp_path):
    parent, body = setup()
    output = publish(parent, body, tmp_path)
    (output / "documents/fixture.html").write_text("Wrong document")
    with pytest.raises(DataValidationError, match="CHILD_HASH"):
        supplements.validate_supplements(output)


def test_identical_revisions_require_every_source_hash(setup, tmp_path):
    parent, body = setup(
        [
            dividend(stk_bo_rate=None, stk_co_rate=None),
            dividend(stk_bo_rate=None, stk_co_rate=None, ann_date="20240101"),
        ]
    )
    assert len(body["supplements"][0]["source_row_hashes"]) == 2
    accepted = publish(parent, body, tmp_path)
    assert (
        len(read_json(accepted / "field_proposals.json")["proposals"][0]["source_row_hashes"]) == 2
    )
    body["supplements"][0]["source_row_hashes"].pop()
    with pytest.raises(DataValidationError, match="RAW_HASH_SET"):
        publish(parent, body, tmp_path)


def test_missing_cash_is_unknown_not_implicit_zero(setup, tmp_path):
    parent, body = setup([dividend(stk_bo_rate=None, stk_co_rate=None, cash_div=None)])
    output = publish(parent, body, tmp_path)
    p = read_json(output / "field_proposals.json")["proposals"][0]
    assert p["proposed_terms"]["cash_div"] is None
    assert "MISSING_ECONOMIC_COMPONENTS" in p["issues"]
    approve(body)
    with pytest.raises(DataValidationError, match="NOT_APPROVABLE"):
        publish(parent, body, tmp_path)


def test_wrong_same_security_document_date_rejected(setup, tmp_path):
    parent, body = setup()
    doc = tmp_path / "documents/fixture.html"
    doc.write_text(doc.read_text().replace("20240104", "20240105"))
    h = file_hash(doc)
    body["documents"][0]["sha256"] = h
    s = body["supplements"][0]
    s["identity_citations"][0].update(document_sha256=h, quote="AAA 20240103 20240105")
    for c in s["claims"]:
        c["document_sha256"] = h
    with pytest.raises(DataValidationError, match="DOCUMENT_EVENT_DATE_MISMATCH"):
        publish(parent, body, tmp_path)


def test_decision_does_not_survive_changed_claim(setup, tmp_path):
    parent, body = setup()
    approve(body)
    body["supplements"][0]["claims"][0]["quote"] = "no reserve conversion."
    with pytest.raises(DataValidationError, match="UNKNOWN_SUPPLEMENT_DECISION"):
        publish(parent, body, tmp_path)


def test_review_timestamp_not_logical(setup, tmp_path):
    parent, body = setup()
    approve(body)
    first = publish(parent, body, tmp_path)
    body["decisions"][0]["review_timestamp"] = "2026-09-29T00:00:00Z"
    assert publish(parent, body, tmp_path) == first


@pytest.mark.parametrize(
    "kind,changes",
    [
        ("CASH_SETTLEMENT", dict(settlement_date="")),
        (
            "ZERO_RECOVERY_PROVEN",
            dict(cash_per_share=0.0, explicit_zero_recovery=True, effective_date=""),
        ),
    ],
)
def test_terminal_blank_date_compile_not_complete(source, tmp_path, kind, changes):  # noqa: F811
    package = terminal_package(tmp_path, economic_resolution_type=kind, **changes)
    provider = sources.probe_dividends(source, Provider(), tmp_path / "provider")
    output = evidence.compile_economic_evidence(
        source, provider, tmp_path / "evidence", terminal_packages=(package,)
    )
    evidence.validate_economic_event_evidence_artifact(output)
    coverage = pd.read_parquet(output / "terminal_coverage.parquet")
    assert not coverage.coverage.eq("EVIDENCE_COMPLETE").any()
    assert read_json(output / "summary.json")["economic_event_evidence_status"] == "BLOCKED"


def test_cli_consumes_new_contract(setup, tmp_path, capsys):
    parent, body = setup()
    input_path = tmp_path / "input.json"
    atomic_write_json(input_path, body)
    assert (
        main(
            [
                "data",
                "economic-review-supplement",
                "--parent",
                str(parent),
                "--parent-manifest-sha256",
                file_hash(parent / "manifest.json"),
                "--supplements",
                str(input_path),
                "--output-root",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)["artifact"]
    assert main(["data", "economic-review-supplement-validate", "--artifact", output]) == 0


def test_legacy_fingerprint_is_explicit_not_any_hash():
    assert sources.compatible_implementation(
        "b6b3bdc88be0be193f225b58141e3186dfd239b03dad1318a29fdad5ff8289c4"
    )
    assert not sources.compatible_implementation("f" * 64)
    assert set(MATERIAL) >= {"cash_div", "stk_bo_rate", "stk_co_rate"}
