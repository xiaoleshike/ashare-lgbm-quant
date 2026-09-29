from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from ashare_quant.backtest.continuous_source import file_hash, read_json
from ashare_quant.cli import main
from ashare_quant.data import economic_event_sources as sources
from ashare_quant.data import economic_review_artifacts as artifacts
from ashare_quant.data.economic_event_evidence import compile_economic_evidence
from ashare_quant.data.economic_event_review import (
    CONTRACT,
    DECISIONS,
    RULES,
    apply_decisions,
    build_triage,
    decision_body,
)
from ashare_quant.data.exceptions import DataValidationError, TusharePermissionError
from ashare_quant.data.security_identity import SecurityAlias, SecurityIdentityResolver
from ashare_quant.data.terminal_announcement_sources import (
    probe_announcements,
    validate_announcement_source,
)
from ashare_quant.utils.manifest import atomic_write_json
from test_economic_event_evidence import Provider, dividend, terminal_package
from test_economic_event_evidence import source as exposure_source  # noqa: F401

REAL_LOAD_INPUTS = artifacts.load_review_inputs


@pytest.fixture
def source(request):
    return request.getfixturevalue("exposure_source")


def raw(source, rows):
    return sources.raw_rows(
        [{"request_id": "fixture", "ts_code": "AAA.SZ", "rows": rows}], source.mapping
    )


def triage(source, rows=None):
    return build_triage(source, raw(source, [dividend()] if rows is None else rows))


def decision(tables, state="APPROVE_AS_EVIDENCE"):
    q = tables["economic_event_review_queue.parquet"].iloc[0]
    gid = json.loads(q.exact_candidate_group_ids)[0]
    g = tables["provider_revision_groups.parquet"]
    hashes = json.loads(g[g.candidate_group_id.eq(gid)].iloc[0].source_row_hashes)
    return dict(
        review_event_id=q.review_event_id,
        candidate_group_id=gid,
        decision=state,
        reviewed_by="fixture reviewer",
        review_reason="Explicit fixture terms reviewed",
        source_row_hashes=hashes,
    )


@pytest.fixture
def inputs(source, tmp_path, monkeypatch):
    provider = sources.probe_dividends(source, Provider(), tmp_path / "provider")
    initial = compile_economic_evidence(source, provider, tmp_path / "initial")
    context = dict(
        initial_evidence_id=initial.name,
        initial_manifest_hash=file_hash(initial / "manifest.json"),
        provider_source_id=provider.name,
        provider_manifest_hash=file_hash(provider / "manifest.json"),
        source=source.identity,
        contract=CONTRACT,
        rules=RULES,
        implementation_hash=artifacts.implementation_hash(),
    )
    result = artifacts.ReviewInputs(initial, provider, source, context)
    monkeypatch.setattr(artifacts, "load_review_inputs", lambda *a, **kw: result)
    return result


def write_decision(inputs, tmp_path, state="APPROVE_AS_EVIDENCE", **changes):
    tables, _ = triage(inputs.source)
    body = {
        "schema": DECISIONS,
        "source": inputs.context,
        "decisions": [decision(tables, state) | changes],
    }
    path = tmp_path / "decision.json"
    atomic_write_json(path, body)
    return path


def test_irrelevant_excluded_and_same_event_shared(source):
    rows = [
        dividend(),
        dividend(
            ex_date="20100104", record_date="20091231", pay_date="20100106", end_date="20091231"
        ),
    ]
    tables, summary = triage(source, rows)
    assert summary["implemented_candidate_rows"] == 2
    assert summary["relevant_candidate_rows"] == 1
    assert summary["irrelevant_historical_rows"] == 1
    q = tables["economic_event_review_queue.parquet"]
    assert len(q) == 1
    assert q.iloc[0].affected_position_count == 3
    assert json.loads(q.iloc[0].affected_top_n) == [10, 20, 50]
    assert q.iloc[0].triage_status == "UNIQUE_EXACT_PROVIDER_EVENT"
    assert not any("pnl" in c.lower() or "return" in c.lower() for c in q)


def test_identical_revisions_preserve_hashes(source):
    tables, _ = triage(source, [dividend(), dividend(ann_date="20240101", imp_ann_date="20240103")])
    groups = tables["provider_revision_groups.parquet"]
    assert len(groups) == 1
    assert len(json.loads(groups.iloc[0].source_row_hashes)) == 2
    assert (
        tables["economic_event_review_queue.parquet"].iloc[0].triage_status
        == "IDENTICAL_REVISIONS_EXACT_EVENT"
    )
    reverse, _ = triage(
        source, [dividend(ann_date="20240101", imp_ann_date="20240103"), dividend()]
    )
    assert sources.content_hash(groups) == sources.content_hash(
        reverse["provider_revision_groups.parquet"]
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(cash_div_tax=0.2),
        dict(stk_div=1),
        dict(record_date="20240102"),
        dict(ex_date="20240105"),
        dict(pay_date="20240111"),
        dict(div_listdate="20240110"),
    ],
)
def test_conflicting_implemented_rows_block(source, changes):
    tables, _ = triage(source, [dividend(), dividend(**changes)])
    assert (
        tables["economic_event_review_queue.parquet"].iloc[0].triage_status
        == "MULTIPLE_CONFLICTING_EXACT_EVENTS"
    )
    with pytest.raises(DataValidationError, match="APPROVAL_NOT_ELIGIBLE"):
        apply_decisions(source, tables, [decision(tables)], [])


def test_exact_plus_unrelated_nearby(source):
    tables, _ = triage(source, [dividend(), dividend(end_date="20230930", ex_date="20240105")])
    q = tables["economic_event_review_queue.parquet"]
    row = q[q.scope.eq("AUDIT_ADJUSTMENT")].iloc[0]
    assert row.exact_ex_date_candidate_count == 1
    assert row.triage_status == "UNIQUE_EXACT_PROVIDER_EVENT"
    assert row.candidate_count == 2


def test_nearby_never_approved(source):
    tables, _ = triage(source, [dividend(ex_date="20240105")])
    q = tables["economic_event_review_queue.parquet"]
    row = q[q.scope.eq("AUDIT_ADJUSTMENT")].iloc[0]
    assert row.match_level == "DATE_NEARBY_REVIEW_REQUIRED"
    assert row.triage_status == "ONLY_NEARBY_EVENT"
    d = decision(tables)
    d["review_event_id"] = row.review_event_id
    with pytest.raises(DataValidationError, match="APPROVAL_NOT_ELIGIBLE"):
        apply_decisions(source, tables, [d], [])


@pytest.mark.parametrize(
    "changes,category",
    [
        ({}, "CASH_ONLY"),
        (
            {
                "cash_div": 0,
                "cash_div_tax": 0,
                "stk_div": 0.2,
                "stk_bo_rate": 0.2,
                "div_listdate": "20240110",
            },
            "SHARES_ONLY",
        ),
        ({"stk_div": 0.2, "stk_co_rate": 0.2, "div_listdate": "20240110"}, "CASH_AND_SHARES"),
        ({"cash_div": None}, "INCOMPLETE_ECONOMICS"),
        ({"cash_div": 0, "cash_div_tax": 0}, "ZERO_ECONOMICS"),
    ],
)
def test_material_categories(source, changes, category):
    tables, _ = triage(source, [dividend(**changes)])
    assert tables["provider_revision_groups.parquet"].iloc[0].materiality_category == category


def test_proposal_not_candidate(source):
    tables, counts = triage(source, [dividend(div_proc="proposal")])
    assert counts["implemented_candidate_rows"] == 0
    assert tables["economic_event_review_queue.parquet"].iloc[0].triage_status == "NO_EXACT_EVENT"


def test_no_event_and_unreviewed_stay_unresolved(source):
    for rows in ([], [dividend()]):
        tables, _ = triage(source, rows)
        result, summary = apply_decisions(source, tables, [], [])
        assert result["reviewed_corporate_actions.parquet"].empty
        assert summary["adjustment_unresolved"] == 1
        assert summary["economic_event_evidence_status"] == "BLOCKED"


def test_explicit_approval_and_rejection(source):
    tables, _ = triage(source)
    approved, summary = apply_decisions(source, tables, [decision(tables)], [])
    assert summary["reviewed_catalog_events"] == 1
    assert summary["terminal_unknown"] == 1
    assert not summary["execution_authorized"]
    assert set(approved["position_entitlement_review.parquet"].eligibility) == {
        "HELD_THROUGH_RECORD_DATE"
    }
    rejected, summary = apply_decisions(
        source, tables, [decision(tables, "REJECT_NOT_SAME_EVENT")], []
    )
    assert rejected["reviewed_corporate_actions.parquet"].empty
    assert summary["adjustment_unresolved"] == 1


def test_alias_preserves_source_code(source):
    mapping = SecurityIdentityResolver(
        mapping_version="fixture",
        mapping_hash="a" * 64,
        aliases=(
            SecurityAlias("ALIAS.SZ", "AAA.SZ", "SZ", None, None, "fixture official mapping"),
        ),
    )
    frame = sources.raw_rows(
        [{"request_id": "fixture", "ts_code": "ALIAS.SZ", "rows": [dividend(ts_code="ALIAS.SZ")]}],
        mapping,
    )
    tables, _ = build_triage(source, frame)
    assert json.loads(tables["provider_revision_groups.parquet"].iloc[0].source_codes) == [
        "ALIAS.SZ"
    ]
    assert tables["economic_event_review_queue.parquet"].iloc[0].canonical_ts_code == "AAA.SZ"


def test_event_without_factor_boundary(source):
    source = replace(
        source, exposures=source.exposures[source.exposures.exposure_type.eq("TERMINAL_EVENT")]
    )
    tables, counts = triage(source)
    assert counts["adjustment_boundaries_total"] == 0
    assert counts["supplemental_entitlement_boundaries"] == 1
    assert tables["economic_event_review_queue.parquet"].iloc[0].scope == "SUPPLEMENTAL_ENTITLEMENT"


def test_timestamp_not_logical_identity(inputs, tmp_path):
    path = write_decision(inputs, tmp_path, review_timestamp="2026-09-29T12:00:00Z")
    a = artifacts.compile_review(inputs, tmp_path / "out", decisions=path)
    write_decision(inputs, tmp_path, review_timestamp="2026-09-30T12:00:00Z")
    assert artifacts.compile_review(inputs, tmp_path / "out", decisions=path) == a
    write_decision(inputs, tmp_path, state="REJECT_NOT_SAME_EVENT")
    b = artifacts.compile_review(inputs, tmp_path / "out", decisions=path)
    assert b != a


def test_publish_validate_idempotent_and_template_unapproved(inputs, tmp_path):
    output = artifacts.compile_review(inputs, tmp_path / "out")
    assert artifacts.validate_reviewed_economic_evidence_artifact(output)["status"] == "COMPLETE"
    assert artifacts.compile_review(inputs, tmp_path / "out") == output
    template = read_json(output / "economic_event_review_template.json")
    assert all(r["decision"] is None for r in template["decisions"])
    assert read_json(output / "summary.json")["reviewed_catalog_events"] == 0
    assert len(read_json(output / "terminal_economic_review_queue.json")["events"]) == 1


@pytest.mark.parametrize(
    "file,column,value",
    [
        ("provider_revision_groups.parquet", "ex_date", "20240105"),
        ("provider_revision_groups.parquet", "cash_div", 9.0),
        ("provider_revision_groups.parquet", "stk_div", 3.0),
        ("provider_revision_groups.parquet", "source_row_hashes", '["bad"]'),
        ("provider_revision_groups.parquet", "candidate_group_id", "bad"),
        ("economic_event_review_queue.parquet", "triage_status", "NO_EXACT_EVENT"),
        ("adjustment_reconciliation.parquet", "review_status", "APPROVED_AS_EVIDENCE"),
        ("terminal_coverage.parquet", "economic_resolution_type", "ZERO_RECOVERY_PROVEN"),
    ],
)
def test_rehashed_tamper_rejected(inputs, tmp_path, file, column, value):
    output = artifacts.compile_review(inputs, tmp_path / "out")
    frame = pd.read_parquet(output / file)
    frame.loc[0, column] = value
    frame.to_parquet(output / file, index=False)
    m = read_json(output / "manifest.json")
    m["artifact_hashes"][file] = file_hash(output / file)
    atomic_write_json(output / "manifest.json", m)
    with pytest.raises(DataValidationError, match="REVIEW_BUSINESS_MISMATCH"):
        artifacts.validate_reviewed_economic_evidence_artifact(output)


def test_wrong_candidate_and_wrong_source_fail(inputs, tmp_path):
    p = write_decision(inputs, tmp_path, candidate_group_id="wrong")
    with pytest.raises(DataValidationError, match="WRONG_REVIEW_CANDIDATE"):
        artifacts.compile_review(inputs, tmp_path / "out", decisions=p)
    body = read_json(write_decision(inputs, tmp_path))
    body["source"]["provider_manifest_hash"] = "bad"
    with pytest.raises(DataValidationError, match="REVIEW_SOURCE_MISMATCH"):
        decision_body(body, inputs.context)


class Announcements:
    def __init__(self, denied=False):
        self.denied = denied
        self.calls = []

    def query(self, endpoint, **params):
        self.calls.append((endpoint, params))
        if self.denied:
            raise TusharePermissionError("secret must not appear")
        return pd.DataFrame(
            [
                dict(
                    ann_date="20240112",
                    ts_code="AAA.SZ",
                    name="fixture",
                    title=title,
                    url="https://example.invalid/notice.pdf",
                )
                for title in ("\u9000\u5e02", "\u6e05\u7b97", "unrelated")
            ]
        )


@pytest.mark.parametrize("denied", [False, True])
def test_optional_announcements_not_economic_evidence(inputs, tmp_path, denied):
    client = Announcements(denied)
    path = probe_announcements(inputs.source, client, tmp_path / "ann")
    frame = validate_announcement_source(path, inputs.source)
    assert len(frame) == (0 if denied else 2)
    response = read_json(path / "responses.json")["requests"][0]
    assert response["status"] == ("PERMISSION_REQUIRED" if denied else "METADATA_CAPTURED")
    assert "secret" not in (path / "responses.json").read_text()
    out = artifacts.compile_review(inputs, tmp_path / "review", announcement_source=path)
    summary = read_json(out / "summary.json")
    assert summary["terminal_unknown"] == 1
    assert summary["terminal_documents_found"] == (0 if denied else 2)
    assert summary["terminal_boundaries_with_documents"] == (0 if denied else 1)


@pytest.mark.parametrize(
    "changes",
    [
        {},
        dict(
            economic_resolution_type="SUCCESSOR_SHARE_ENTITLEMENT",
            cash_per_share=None,
            successor_security="BBB.SZ",
            share_conversion_ratio=2,
        ),
        dict(
            economic_resolution_type="ZERO_RECOVERY_PROVEN",
            cash_per_share=0,
            explicit_zero_recovery=True,
        ),
    ],
)
def test_reviewed_terminal_contract_reused(inputs, tmp_path, changes):
    package = terminal_package(tmp_path, **changes)
    out = artifacts.compile_review(inputs, tmp_path / "out", terminal_packages=(package,))
    assert read_json(out / "summary.json")["terminal_reviewed_complete"] == 1


def test_cli_triage_no_network(inputs, tmp_path, monkeypatch, capsys):
    from ashare_quant.cli import economic_review as cli

    monkeypatch.setattr(cli, "load_review_inputs", lambda *a: inputs)
    monkeypatch.setattr(cli, "TushareClient", lambda **kw: pytest.fail("unexpected provider query"))
    assert (
        main(
            [
                "data",
                "economic-review-triage",
                "--initial-evidence",
                str(inputs.initial),
                "--initial-manifest-sha256",
                inputs.context["initial_manifest_hash"],
                "--provider-source",
                str(inputs.provider),
                "--provider-manifest-sha256",
                inputs.context["provider_manifest_hash"],
                "--output-root",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    assert "artifact" in capsys.readouterr().out


def test_source_call_chain_and_changed_provider_fail(inputs, monkeypatch):
    monkeypatch.setattr(artifacts, "load_exposure_source", lambda *a, **kw: inputs.source)
    validated = REAL_LOAD_INPUTS(
        inputs.initial,
        inputs.provider,
        inputs.context["initial_manifest_hash"],
        inputs.context["provider_manifest_hash"],
    )
    assert validated.context == inputs.context
    path = inputs.provider / "raw_dividend_evidence.parquet"
    path.write_bytes(b"tampered provider fixture")
    with pytest.raises(DataValidationError, match="CHILD_HASH_MISMATCH"):
        REAL_LOAD_INPUTS(
            inputs.initial,
            inputs.provider,
            inputs.context["initial_manifest_hash"],
            inputs.context["provider_manifest_hash"],
        )


def test_forged_approved_catalog_without_decision(inputs, tmp_path):
    output = artifacts.compile_review(inputs, tmp_path / "out")
    tables, _ = triage(inputs.source)
    approved, _ = apply_decisions(inputs.source, tables, [decision(tables)], [])
    name = "reviewed_corporate_actions.parquet"
    artifacts.review_table(approved[name]).to_parquet(output / name, index=False)
    m = read_json(output / "manifest.json")
    m["artifact_hashes"][name] = file_hash(output / name)
    atomic_write_json(output / "manifest.json", m)
    with pytest.raises(DataValidationError, match="REVIEW_BUSINESS_MISMATCH"):
        artifacts.validate_reviewed_economic_evidence_artifact(output)


def test_terminal_unknown_forgery_in_json(inputs, tmp_path):
    output = artifacts.compile_review(inputs, tmp_path / "out")
    name = "terminal_economic_review_queue.json"
    body = read_json(output / name)
    body["events"][0]["economic_resolution_type"] = "ZERO_RECOVERY_PROVEN"
    atomic_write_json(output / name, body)
    m = read_json(output / "manifest.json")
    m["artifact_hashes"][name] = file_hash(output / name)
    atomic_write_json(output / "manifest.json", m)
    with pytest.raises(DataValidationError, match="REVIEW_BUSINESS_MISMATCH"):
        artifacts.validate_reviewed_economic_evidence_artifact(output)


def test_entitlement_unknown_keeps_gate_closed(source):
    source = replace(source, holdings=source.holdings[source.holdings.trade_date.ne("20240103")])
    tables, _ = triage(source)
    result, summary = apply_decisions(source, tables, [decision(tables)], [])
    assert "ENTITLEMENT" in set(result["unresolved_events.parquet"].finding_type)
    assert summary["economic_event_evidence_status"] == "BLOCKED"


def test_missing_dates_not_filled_and_pending_dates_preserved(source):
    tables, _ = triage(source, [dividend(ex_date=None)])
    assert "UNDATED_ENTITLEMENT_REVIEW" in set(tables["economic_event_review_queue.parquet"].scope)
    assert tables["provider_revision_groups.parquet"].iloc[0].ex_date is None
    tables, _ = triage(source, [dividend(pay_date=None)])
    result, _ = apply_decisions(source, tables, [decision(tables)], [])
    assert result["reviewed_corporate_actions.parquet"].iloc[0].pay_date is None
    assert result["position_entitlement_review.parquet"].pay_date.isna().all()


def test_share_consistency_diagnostic_not_imputed(source):
    tables, _ = triage(source, [dividend(stk_div=1, stk_bo_rate=0, stk_co_rate=0)])
    group = tables["provider_revision_groups.parquet"].iloc[0]
    assert group.share_consistency_diagnostic == "REVIEW_REQUIRED"
    assert group.stk_bo_rate == 0 and group.stk_co_rate == 0 and group.stk_div == 1
    result, _ = apply_decisions(source, tables, [decision(tables)], [])
    assert "SHARE_COMPONENT_DIAGNOSTIC" in set(result["unresolved_events.parquet"].finding_type)


def test_approval_after_cutoff_rejected(source):
    changed = dict(source.identity, governed_execution_cutoff="20240103")
    source = replace(source, identity=changed)
    tables, _ = triage(source)
    with pytest.raises(DataValidationError, match="APPROVAL_NOT_ELIGIBLE"):
        apply_decisions(source, tables, [decision(tables)], [])


def test_terminal_package_inherited_not_lost(source, tmp_path, monkeypatch):
    provider = sources.probe_dividends(source, Provider(), tmp_path / "p")
    package = terminal_package(tmp_path)
    initial = compile_economic_evidence(
        source, provider, tmp_path / "initial", terminal_packages=(package,)
    )
    monkeypatch.setattr(artifacts, "load_exposure_source", lambda *a, **kw: source)
    inputs = REAL_LOAD_INPUTS(
        initial,
        provider,
        file_hash(initial / "manifest.json"),
        file_hash(provider / "manifest.json"),
    )
    out = artifacts.compile_review(inputs, tmp_path / "out")
    assert read_json(out / "summary.json")["terminal_reviewed_complete"] == 1


def test_decision_wrong_row_hash_rejected(inputs, tmp_path):
    path = write_decision(inputs, tmp_path, source_row_hashes=["f" * 64])
    with pytest.raises(DataValidationError, match="REVIEW_ROW_HASH_MISMATCH"):
        artifacts.compile_review(inputs, tmp_path / "out", decisions=path)


def test_cli_compile_validate_fixture(inputs, tmp_path, monkeypatch, capsys):
    from ashare_quant.cli import economic_review as cli

    monkeypatch.setattr(cli, "load_review_inputs", lambda *a: inputs)
    path = write_decision(inputs, tmp_path)
    assert (
        main(
            [
                "data",
                "economic-review-compile",
                "--initial-evidence",
                str(inputs.initial),
                "--initial-manifest-sha256",
                inputs.context["initial_manifest_hash"],
                "--provider-source",
                str(inputs.provider),
                "--provider-manifest-sha256",
                inputs.context["provider_manifest_hash"],
                "--decisions",
                str(path),
                "--output-root",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)["artifact"]
    assert main(["data", "economic-review-validate", "--artifact", output]) == 0
    assert '"validation": "PASS"' in capsys.readouterr().out
