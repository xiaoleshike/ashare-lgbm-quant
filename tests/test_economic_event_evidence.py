from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from ashare_quant.backtest.continuous_source import file_hash, payload_hash, read_json
from ashare_quant.cli import main
from ashare_quant.data import economic_event_evidence as evidence
from ashare_quant.data import economic_event_sources as sources
from ashare_quant.data.economic_events import (
    DIVIDEND_FIELDS,
    TERMINAL_CONTRACT,
    TerminalEconomicRecord,
    normalize_dividends,
)
from ashare_quant.data.exceptions import DataValidationError, TusharePermissionError
from ashare_quant.data.security_identity import SecurityIdentityResolver
from ashare_quant.utils.manifest import atomic_write_json
from test_continuous_oos_backtest import service as continuous_service  # noqa: F401


def dividend(**kwargs):
    return (
        dict(
            ts_code="AAA.SZ",
            end_date="20231231",
            ann_date="20240102",
            div_proc="实施",
            stk_div=0.0,
            stk_bo_rate=0.0,
            stk_co_rate=0.0,
            cash_div=0.08,
            cash_div_tax=0.1,
            record_date="20240103",
            ex_date="20240104",
            pay_date="20240110",
            div_listdate=None,
            imp_ann_date="20240102",
            base_date=None,
            base_share=None,
        )
        | kwargs
    )


class Provider:
    def __init__(self, rows=None):
        self.rows = [dividend()] if rows is None else rows
        self.calls = []

    def query(self, endpoint, **params):
        self.calls.append((endpoint, params))
        return pd.DataFrame(self.rows, columns=DIVIDEND_FIELDS)


@pytest.fixture
def source(tmp_path, monkeypatch):
    positions, exposures, holdings = [], [], []
    calendar = (
        "20240102",
        "20240103",
        "20240104",
        "20240105",
        "20240108",
        "20240109",
        "20240110",
        "20240111",
        "20240112",
    )
    for n in (10, 20, 50):
        for terminal in (False, True):
            pid = f"{n}:terminal" if terminal else f"{n}:normal"
            end = "20240112" if terminal else "20240108"
            positions.append(
                dict(
                    top_n=n,
                    position_id=pid,
                    ts_code="AAA.SZ",
                    entry_date="20240102",
                    target_exit_date="20240108",
                    resolution_date=end,
                    entry_gross=1e-12 if n == 50 else 100.0,
                )
            )
            event_type = "TERMINAL_EVENT" if terminal else "ADJ_FACTOR_CHANGE"
            dt = end if terminal else "20240104"
            boundary = payload_hash([event_type, "AAA.SZ", dt])
            exposures.append(
                dict(
                    exposure_id=payload_hash([boundary, n, pid]),
                    economic_boundary_id=boundary,
                    exposure_type=event_type,
                    ts_code="AAA.SZ",
                    position_id=pid,
                    top_n=n,
                    entry_date="20240102",
                    target_exit_date="20240108",
                    resolution_date=end,
                    candidate_event_date=dt,
                    previous_observation=None if terminal else "20240103",
                    adj_factor_before=None if terminal else 1.0,
                    adj_factor_after=None if terminal else 1.01,
                    entry_gross=positions[-1]["entry_gross"],
                    continuous_run_id="continuous_fixture",
                    continuous_manifest_hash="a" * 64,
                    audit_manifest_hash="b" * 64,
                )
            )
            holdings.extend(
                dict(top_n=n, position_id=pid, ts_code="AAA.SZ", trade_date=d)
                for d in calendar
                if d < end
            )
    frame = (
        pd.DataFrame(exposures, columns=sources.EXPOSURE_COLUMNS)
        .sort_values("exposure_id")
        .reset_index(drop=True)
    )
    result = sources.EconomicExposureSource(
        tmp_path / "run",
        tmp_path / "audit",
        {
            "continuous_run_id": "continuous_fixture",
            "continuous_manifest_hash": "a" * 64,
            "audit_manifest_hash": "b" * 64,
            "exposure_inventory_hash": sources.content_hash(frame),
            "execution_end": "20240112",
            "governed_execution_cutoff": "20240112",
        },
        frame,
        pd.DataFrame(positions),
        pd.DataFrame(holdings),
        calendar,
        SecurityIdentityResolver.empty(),
    )
    # Acquisition/compilation tests inject only the already-validated source context.
    # Source loader itself is covered independently and by real read-only preflight.
    monkeypatch.setattr(evidence, "load_exposure_source", lambda *a, **kw: result)
    return result


def approve(path, tmp_path):
    template = read_json(path / "dividend_review_template.json")
    for r in template["reviews"]:
        r.update(
            decision="APPROVED",
            reviewed_by="fixture-reviewer",
            reviewed_fact="Fixture implemented terms reviewed",
        )
    output = tmp_path / "review.json"
    atomic_write_json(output, template)
    return output


def normalized(rows):
    mapping = SecurityIdentityResolver.empty()
    raw = sources.raw_rows([{"ts_code": "AAA.SZ", "request_id": "r", "rows": rows}], mapping)
    reviews = [
        dict(
            provider_source_id="source",
            provider_raw_row_hash=h,
            decision="APPROVED",
            reviewed_by="fixture",
            reviewed_fact="Reviewed",
        )
        for h in sorted(set(raw.provider_raw_row_hash))
    ]
    return normalize_dividends(raw, "source", reviews, "20240112")


@pytest.mark.parametrize(
    "changes,component",
    [
        ({}, "CASH_DIVIDEND"),
        (
            {
                "cash_div": 0.0,
                "cash_div_tax": 0.0,
                "stk_div": 0.2,
                "stk_bo_rate": 0.2,
                "div_listdate": "20240105",
            },
            "STOCK_BONUS",
        ),
        (
            {
                "cash_div": 0.0,
                "cash_div_tax": 0.0,
                "stk_div": 0.3,
                "stk_co_rate": 0.3,
                "div_listdate": "20240105",
            },
            "CAPITAL_RESERVE_CONVERSION",
        ),
        (
            {"stk_div": 0.2, "stk_bo_rate": 0.1, "stk_co_rate": 0.1, "div_listdate": "20240105"},
            "COMPOSITE_DISTRIBUTION",
        ),
    ],
)
def test_normalized_components(changes, component):
    event = normalized([dividend(**changes)]).iloc[0]
    assert event.component_type == component
    assert event.evidence_status == "REVIEWED_PROVIDER_EVIDENCE"
    assert event.cash_tax_semantics == "UNRESOLVED_EXECUTION_POLICY"
    assert event.evidence_use == "POST_HOC_ACCOUNTING_EVIDENCE"


@pytest.mark.parametrize(
    "changes,issue",
    [
        ({"ex_date": None}, "MISSING_ex_date"),
        ({"pay_date": None}, "MISSING_pay_date"),
        ({"stk_div": 0.1, "stk_bo_rate": 0.1}, "MISSING_div_listdate"),
        ({"stk_div": 0.2, "stk_bo_rate": 0.1}, "CONFLICTING_SHARE_COMPONENTS"),
        ({"record_date": "20240230"}, "INVALID_record_date"),
        ({"ex_date": "99991231"}, "INVALID_ex_date"),
        ({"cash_div": float("nan")}, "INVALID_cash_div"),
        ({"cash_div": -0.1}, "INVALID_cash_div"),
        ({"pay_date": "20240102"}, "DATE_ORDER_CONFLICT"),
    ],
)
def test_missing_invalid_terms_never_guessed(changes, issue):
    if any(isinstance(v, float) and pd.isna(v) for v in changes.values()):
        with pytest.raises(ValueError):
            normalized([dividend(**changes)])
        return
    event = normalized([dividend(**changes)]).iloc[0]
    assert event.evidence_status == "AMBIGUOUS_PROVIDER_EVENT"
    assert issue in json.loads(event.issues)


def test_proposals_and_revisions():
    events = normalized([dividend(div_proc="预案", cash_div=0.2), dividend()])
    assert len(events) == 1 and events.iloc[0].cash_div_after_tax_per_share == 0.08
    events = normalized([dividend(), dividend(cash_div_tax=0.2)])
    assert set(events.evidence_status) == {"MULTIPLE_CONFLICTING_EVENTS"}


def test_no_review_no_execution_truth(source, tmp_path):
    provider = Provider()
    p = sources.probe_dividends(source, provider, tmp_path / "sources")
    assert len(provider.calls) == 1
    artifact = evidence.compile_economic_evidence(source, p, tmp_path / "evidence")
    evidence.validate_economic_event_evidence_artifact(artifact)
    assert set(pd.read_parquet(artifact / "dividend_candidates.parquet").evidence_status) == {
        "UNREVIEWED"
    }
    assert pd.read_parquet(artifact / "normalized_corporate_actions.parquet").empty
    assert read_json(artifact / "summary.json")["economic_event_evidence_status"] == "BLOCKED"
    assert evidence.compile_economic_evidence(source, p, tmp_path / "evidence") == artifact


@pytest.mark.parametrize(
    "entry,end,record,expected",
    [
        ("20240102", "20240108", "20240103", "HELD_THROUGH_RECORD_DATE"),
        ("20240104", "20240108", "20240103", "NOT_HELD_ON_RECORD_DATE"),
        ("20240102", "20240105", "20240103", "HELD_THROUGH_RECORD_DATE"),
        ("20240102", "20240103", "20240103", "NOT_HELD_ON_RECORD_DATE"),
        ("20240102", "20240108", "20240106", "RECORD_DATE_NOT_IN_EXECUTION_CALENDAR"),
        ("20240102", "20240108", None, "AMBIGUOUS"),
    ],
)
def test_record_date_eod_ownership(source, entry, end, record, expected):
    # Ex-date remains in-window or record-date ownership surfaces a sold-before-payment claim.
    p = source.positions.iloc[:1].copy()
    p["entry_date"] = entry
    p["resolution_date"] = end
    ex = "20240103" if end == "20240103" else "20240108" if record == "20240106" else "20240104"
    h = source.holdings[
        (source.holdings.position_id == p.iloc[0].position_id)
        & source.holdings.trade_date.ge(entry)
        & source.holdings.trade_date.lt(end)
    ]
    context = replace(source, positions=p, holdings=h)
    result = evidence.position_coverage(
        context, normalized([dividend(record_date=record, ex_date=ex)])
    )
    assert len(result) == 1 and result.iloc[0].eligibility == expected


def test_bidirectional_event_without_factor_and_unmatched_factor(source):
    events = normalized([dividend(ex_date="20240105", record_date="20240104")])
    coverage = evidence.position_coverage(source, events)
    assert len(coverage) == 6 and not coverage.observed_adjustment.any()
    assert set(coverage.eligibility) == {"HELD_THROUGH_RECORD_DATE"}
    recon = evidence.reconcile_adjustments(source, events)
    assert len(recon) == 3 and set(recon.classification) == {"NO_EVENT_FOUND"}


def terminal_record(doc_hash, **kwargs):
    return (
        dict(
            ts_code="AAA.SZ",
            terminal_date="20240112",
            economic_resolution_type="CASH_SETTLEMENT",
            cash_per_share=2.0,
            effective_date="20240112",
            settlement_date="20240115",
            source_type="EXCHANGE",
            source_url="https://www.szse.cn/fixture",
            document_id="fixture-1",
            document_file="notice.txt",
            document_sha256=doc_hash,
            document_location="paragraph 1",
            reviewed_fact="Exact economic terms",
            reviewed_by="fixture-reviewer",
            review_status="VERIFIED",
            evidence_scope="ECONOMIC_ENTITLEMENT",
        )
        | kwargs
    )


def terminal_package(tmp_path, **kwargs):
    docs = tmp_path / "documents"
    docs.mkdir(exist_ok=True)
    (docs / "notice.txt").write_text("Synthetic authoritative fixture, not real evidence.")
    review = tmp_path / "terminal-review.json"
    atomic_write_json(
        review,
        {
            "schema": TERMINAL_CONTRACT,
            "records": [terminal_record(file_hash(docs / "notice.txt"), **kwargs)],
        },
    )
    return evidence.publish_terminal_evidence(review, docs, tmp_path / "terminal-packages")


@pytest.mark.parametrize(
    "changes,complete",
    [
        ({}, True),
        (
            {
                "economic_resolution_type": "SUCCESSOR_SHARE_ENTITLEMENT",
                "cash_per_share": None,
                "successor_security": "BBB.SZ",
                "share_conversion_ratio": 2.0,
            },
            True,
        ),
        (
            {
                "economic_resolution_type": "LIQUIDATION_RECEIVABLE",
                "cash_per_share": None,
                "receivable_terms": "Uncertain claim, transferable under reviewed terms",
            },
            True,
        ),
        (
            {
                "economic_resolution_type": "TRANSFERRED_SECURITY_RIGHT",
                "cash_per_share": None,
                "receivable_terms": "Reviewed security-right terms",
            },
            True,
        ),
        (
            {
                "economic_resolution_type": "ZERO_RECOVERY_PROVEN",
                "cash_per_share": 0.0,
                "explicit_zero_recovery": True,
            },
            True,
        ),
        ({"economic_resolution_type": "ECONOMIC_RECOVERY_UNKNOWN", "cash_per_share": None}, False),
        ({"settlement_date": None}, False),
    ],
)
def test_terminal_classifications(source, tmp_path, changes, complete):
    p = terminal_package(tmp_path, **changes)
    rows = evidence.validate_terminal_evidence(p)
    coverage = evidence.terminal_coverage(source, rows)
    assert len(coverage) == 1 and coverage.iloc[0].exposure_count == 3
    assert (coverage.iloc[0].coverage == "EVIDENCE_COMPLETE") == complete


def test_terminal_listing_date_is_not_zero(source, tmp_path):
    coverage = evidence.terminal_coverage(source, [])
    assert coverage.iloc[0].coverage == "ECONOMIC_RECOVERY_UNKNOWN"
    with pytest.raises(DataValidationError, match="ZERO_RECOVERY_NOT_PROVEN"):
        terminal_package(
            tmp_path, economic_resolution_type="ZERO_RECOVERY_PROVEN", cash_per_share=0.0
        )
    with pytest.raises(ValueError):
        TerminalEconomicRecord.model_validate(
            terminal_record("a" * 64, evidence_scope="LISTING_ONLY")
        )


def test_complete_and_conflicting_terminal_gate(source, tmp_path):
    provider = sources.probe_dividends(source, Provider(), tmp_path / "provider")
    review = approve(provider, tmp_path)
    terminal = terminal_package(tmp_path)
    artifact = evidence.compile_economic_evidence(
        source, provider, tmp_path / "output", dividend_review=review, terminal_packages=(terminal,)
    )
    evidence.validate_economic_event_evidence_artifact(artifact)
    summary = read_json(artifact / "summary.json")
    assert (
        summary["economic_event_evidence_status"] == "COMPLETE"
        and not summary["execution_authorized"]
    )
    assert summary["by_top_n"]["50"]["raw_position_count"] == 2
    assert summary["by_top_n"]["50"]["capital_material_position_count"] == 0
    other = terminal_package(tmp_path, cash_per_share=3.0)
    conflict = evidence.compile_economic_evidence(
        source,
        provider,
        tmp_path / "output",
        dividend_review=review,
        terminal_packages=(terminal, other),
    )
    assert (
        conflict != artifact
        and read_json(conflict / "summary.json")["economic_event_evidence_status"] == "BLOCKED"
    )


@pytest.mark.parametrize(
    "file,column,value",
    [
        ("normalized_corporate_actions.parquet", "cash_div_before_tax_per_share", 9.0),
        ("normalized_corporate_actions.parquet", "record_date", "20240102"),
        ("normalized_corporate_actions.parquet", "ex_date", "20240105"),
        ("normalized_corporate_actions.parquet", "cash_pay_date", "20240111"),
        ("normalized_corporate_actions.parquet", "stock_bonus_rate", 1.0),
        ("raw_dividend_evidence.parquet", "raw_row_json", "{}"),
        ("adjustment_event_reconciliation.parquet", "event_ids", "[]"),
        ("terminal_coverage.parquet", "coverage", "EVIDENCE_COMPLETE"),
        ("unresolved_events.parquet", "reason", "invented"),
        ("exposure_inventory.parquet", "audit_manifest_hash", "f" * 64),
    ],
)
def test_rehash_cannot_hide_business_tamper(source, tmp_path, file, column, value):
    provider = sources.probe_dividends(source, Provider(), tmp_path / "provider")
    artifact = evidence.compile_economic_evidence(
        source, provider, tmp_path / "output", dividend_review=approve(provider, tmp_path)
    )
    df = pd.read_parquet(artifact / file)
    df.loc[0, column] = value
    df.to_parquet(artifact / file, index=False)
    m = read_json(artifact / "manifest.json")
    m["artifact_hashes"][file] = file_hash(artifact / file)
    atomic_write_json(artifact / "manifest.json", m)
    with pytest.raises(DataValidationError, match="RECONSTRUCTION"):
        evidence.validate_economic_event_evidence_artifact(artifact)


def test_provider_rehashed_raw_tamper(source, tmp_path):
    p = sources.probe_dividends(source, Provider(), tmp_path / "provider")
    f = p / "raw_dividend_evidence.parquet"
    df = pd.read_parquet(f)
    df.loc[0, "raw_row_json"] = "{}"
    df.to_parquet(f, index=False)
    m = read_json(p / "manifest.json")
    m["artifact_hashes"][f.name] = file_hash(f)
    atomic_write_json(p / "manifest.json", m)
    with pytest.raises(DataValidationError, match="RAW_RECONSTRUCTION"):
        sources.validate_provider_source(p, source)


@pytest.mark.parametrize(
    "rows,expected", [(2000, "POSSIBLY_TRUNCATED"), (0, "EMPTY_RESPONSE_NOT_PROOF")]
)
def test_provider_capture_limits(source, tmp_path, rows, expected):
    p = sources.probe_dividends(source, Provider([dividend()] * rows), tmp_path / "provider")
    assert read_json(p / "responses.json")["requests"][0]["status"] == expected


def test_permission_failure_is_sanitized(source, tmp_path):
    class Denied:
        def query(self, *args, **kwargs):
            raise TusharePermissionError("sensitive provider exception text")

    p = sources.probe_dividends(source, Denied(), tmp_path / "provider")
    assert "sensitive" not in (p / "responses.json").read_text()
    assert read_json(p / "responses.json")["requests"][0]["status"] == "OPERATOR_ACTION_REQUIRED"


def test_future_effective_events_not_activated(source):
    event = normalized([dividend(ex_date="20250101", record_date="20241231", pay_date="20250102")])
    assert not event.iloc[0].within_execution_support
    assert evidence.position_coverage(source, event).empty
    event = normalized([dividend(imp_ann_date="20240201")])
    assert event.iloc[0].announcement_timing == "POST_EVENT"


def test_cli_read_only_preflight(source, monkeypatch, capsys):
    from ashare_quant.cli import economic_events as cli

    monkeypatch.setattr(cli, "load_exposure_source", lambda *a, **kw: source)
    monkeypatch.setattr(
        cli, "TushareClient", lambda *a, **kw: pytest.fail("network client on preflight")
    )
    result = main(
        [
            "--config",
            "config/default.yaml",
            "data",
            "economic-evidence-preflight",
            "--continuous-run",
            str(source.run),
            "--continuous-manifest-sha256",
            "a" * 64,
            "--audit-root",
            str(source.audit),
            "--audit-manifest-sha256",
            "b" * 64,
        ]
    )
    assert result == 0 and json.loads(capsys.readouterr().out)["exposures"] == 6


def test_real_validator_call_chain_with_synthetic_replay(request, tmp_path, monkeypatch):
    """Real completed replay validator -> audit loader -> probe -> compiler -> validator."""
    from types import SimpleNamespace

    replay_service = request.getfixturevalue("continuous_service")
    run = replay_service.run()
    audit = tmp_path / "audit"
    audit.mkdir()
    position_frames, change_rows, crossing_rows = [], [], []
    for n in (10, 20, 50):
        t = pd.read_parquet(run / f"top_{n}/trades.parquet")
        h = pd.read_parquet(run / f"top_{n}/holdings.parquet")
        b = t[t.side.eq("buy") & t.status.eq("filled")].set_index("position_id")
        e = t[t.side.eq("sell") & t.status.eq("filled")].set_index("position_id")
        for pid, row in b.iterrows():
            target = h[h.position_id.eq(pid)].iloc[0].target_exit_date
            p = dict(
                position_id=pid,
                top_n=n,
                ts_code=row.ts_code,
                entry_date=row.trade_date,
                target_exit_date=target,
                resolution_date=e.loc[pid].trade_date,
                entry_gross=row.gross_value,
            )
            position_frames.append(p)
            dt = h[h.position_id.eq(pid)].trade_date.iloc[1]
            crossing_rows.append(p | dict(change_count=1, change_dates=json.dumps([dt])))
            change_rows.append(
                dict(
                    position_id=pid,
                    top_n=n,
                    ts_code=row.ts_code,
                    previous_observation=row.trade_date,
                    change_date=dt,
                    previous_factor=1.0,
                    new_factor=1.01,
                )
            )
    pd.DataFrame(position_frames).to_csv(audit / "positions.csv", index=False)
    pd.DataFrame(crossing_rows).to_csv(audit / "adjustment_crossing_audit.csv", index=False)
    pd.DataFrame(change_rows).to_csv(audit / "adjustment_change_details.csv", index=False)
    pd.DataFrame(columns=["position_id", "top_n", "writeoff_date"]).to_csv(
        audit / "terminal_writeoff_audit.csv", index=False
    )
    pd.DataFrame(columns=["canonical_ts_code"]).to_csv(
        audit / "terminal_official_index_matches.csv", index=False
    )
    rh = file_hash(run / "manifest.json")
    atomic_write_json(
        audit / "audit_manifest.json",
        dict(
            status="COMPLETE",
            source_run_id=run.name,
            source_manifest_hash=rh,
            analysis_identity="fixture",
            definitions={"fixture": 1},
            output_hashes={n: file_hash(audit / n) for n in sources.AUDIT_FILES},
            input_child_hashes={},
            analysis_script_hashes={},
        ),
    )
    mapping = tmp_path / "mapping.json"
    atomic_write_json(
        mapping,
        {
            "schema_version": 1,
            "artifact_name": "security_identity_mapping",
            "mapping_version": "fixture",
            "aliases": [],
        },
    )
    monkeypatch.setattr(
        sources,
        "load_settings",
        lambda *a: SimpleNamespace(security_identity=SimpleNamespace(mapping_path=mapping)),
    )
    replay_service.transitions_path.mkdir()
    atomic_write_json(replay_service.transitions_path / "manifest.json", {})
    context = sources.load_exposure_source(
        run, audit, continuous_hash=rh, audit_hash=file_hash(audit / "audit_manifest.json")
    )
    provider = sources.probe_dividends(context, Provider(), tmp_path / "provider")
    artifact = evidence.compile_economic_evidence(context, provider, tmp_path / "compiled")
    evidence.validate_economic_event_evidence_artifact(artifact)
    with pytest.raises(DataValidationError, match="AUDIT_HASH"):
        sources.load_exposure_source(run, audit, continuous_hash=rh, audit_hash="0" * 64)
    # A changed audit membership cannot pass even if its CSV and manifest are rehashed.
    frame = pd.read_csv(audit / "positions.csv")
    frame.loc[0, "entry_date"] = 20240101
    frame.to_csv(audit / "positions.csv", index=False)
    am = read_json(audit / "audit_manifest.json")
    am["output_hashes"]["positions.csv"] = file_hash(audit / "positions.csv")
    atomic_write_json(audit / "audit_manifest.json", am)
    with pytest.raises(DataValidationError, match="POSITION_LINEAGE"):
        sources.load_exposure_source(
            run, audit, continuous_hash=rh, audit_hash=file_hash(audit / "audit_manifest.json")
        )


def test_terminal_document_tamper(tmp_path):
    p = terminal_package(tmp_path)
    doc = next((p / "documents").iterdir())
    doc.write_text("tampered")
    m = read_json(p / "manifest.json")
    m["artifact_hashes"][str(doc.relative_to(p))] = file_hash(doc)
    atomic_write_json(p / "manifest.json", m)
    with pytest.raises(DataValidationError, match="TERMINAL_DOCUMENT_HASH"):
        evidence.validate_terminal_evidence(p)


def test_missing_components_are_not_zero():
    event = normalized([dividend(stk_div=None, stk_bo_rate=None, stk_co_rate=None)]).iloc[0]
    assert pd.isna(event.total_stock_distribution_rate)
    assert event.evidence_status == "AMBIGUOUS_PROVIDER_EVENT"


def test_empty_provider_cannot_complete(source, tmp_path):
    p = sources.probe_dividends(source, Provider([]), tmp_path / "provider")
    artifact = evidence.compile_economic_evidence(source, p, tmp_path / "evidence")
    evidence.validate_economic_event_evidence_artifact(artifact)
    assert read_json(artifact / "summary.json")["economic_event_evidence_status"] == "BLOCKED"


def test_same_economics_duplicate_sources_preserve_references():
    raw = sources.raw_rows(
        [
            {
                "ts_code": "AAA.SZ",
                "request_id": "r",
                "rows": [dividend(), dividend(base_share=123.0)],
            }
        ],
        SecurityIdentityResolver.empty(),
    )
    reviews = [
        dict(
            provider_source_id="source",
            provider_raw_row_hash=h,
            decision="APPROVED",
            reviewed_by="fixture",
            reviewed_fact="Same implemented economics",
        )
        for h in raw.provider_raw_row_hash
    ]
    events = normalize_dividends(raw, "source", reviews, "20240112")
    assert len(events) == 1
    assert len(json.loads(events.iloc[0].provider_raw_row_hashes)) == 2


def test_pending_after_account_exit_still_covered(source):
    p = source.positions.iloc[:1].copy()
    p["resolution_date"] = "20240104"
    h = source.holdings[
        (source.holdings.position_id == p.iloc[0].position_id)
        & source.holdings.trade_date.lt("20240104")
    ]
    context = replace(source, positions=p, holdings=h)
    events = normalized([dividend(ex_date="20240105", pay_date="20240111")])
    coverage = evidence.position_coverage(context, events)
    assert len(coverage) == 1 and coverage.iloc[0].eligibility == "HELD_THROUGH_RECORD_DATE"


def test_missing_history_is_not_inferred(source):
    events = normalized([dividend(record_date="20240101")])
    assert set(evidence.position_coverage(source, events).eligibility) == {
        "INSUFFICIENT_POSITION_HISTORY"
    }
    events = normalized([dividend()])
    context = replace(source, holdings=source.holdings[source.holdings.trade_date.ne("20240103")])
    assert set(evidence.position_coverage(context, events).eligibility) == {
        "INSUFFICIENT_POSITION_HISTORY"
    }


def test_missing_ex_date_is_not_silently_out_of_scope(source, tmp_path):
    provider = sources.probe_dividends(
        source, Provider([dividend(ex_date=None)]), tmp_path / "provider"
    )
    artifact = evidence.compile_economic_evidence(
        source, provider, tmp_path / "out", dividend_review=approve(provider, tmp_path)
    )
    unresolved = pd.read_parquet(artifact / "unresolved_events.parquet")
    assert "EX_DATE_UNKNOWN_CANNOT_EXCLUDE_EXPOSURE" in set(unresolved.reason)
    assert pd.read_parquet(artifact / "normalized_corporate_actions.parquet").empty


def test_summary_rehash_cannot_claim_complete(source, tmp_path):
    provider = sources.probe_dividends(source, Provider(), tmp_path / "provider")
    artifact = evidence.compile_economic_evidence(source, provider, tmp_path / "out")
    summary = read_json(artifact / "summary.json")
    summary["economic_event_evidence_status"] = "COMPLETE"
    atomic_write_json(artifact / "summary.json", summary)
    m = read_json(artifact / "manifest.json")
    m["artifact_hashes"]["summary.json"] = file_hash(artifact / "summary.json")
    atomic_write_json(artifact / "manifest.json", m)
    with pytest.raises(DataValidationError, match="COMPLETENESS_MISMATCH"):
        evidence.validate_economic_event_evidence_artifact(artifact)


def test_invalid_terminal_component_and_dates(tmp_path):
    with pytest.raises(DataValidationError, match="TERMINAL_COMPONENT_CONFLICT"):
        terminal_package(tmp_path, successor_security="BBB.SZ", share_conversion_ratio=2.0)
    with pytest.raises(DataValidationError, match="INVALID_DATE"):
        terminal_package(tmp_path, settlement_date="20240230")
