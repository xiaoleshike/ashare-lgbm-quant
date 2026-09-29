from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.backtest import continuous
from ashare_quant.backtest.continuous import (
    ContinuousStrictOOSReplayService,
    ReplayPreflight,
    validate_continuous_strict_oos_artifact,
    validate_signal_calendar,
)
from ashare_quant.backtest.continuous_evidence import accounting_evidence, comparison_metrics
from ashare_quant.backtest.continuous_source import file_hash, load_continuous_source, read_json
from ashare_quant.backtest.engine import BacktestInputs, simulate_portfolio
from ashare_quant.config.settings import AppSettings, BacktestSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_identity import SecurityIdentityResolver
from ashare_quant.data.security_identity_transition import (
    SecurityIdentityTransition,
    SecurityIdentityTransitionResolver,
)
from ashare_quant.data.security_lifecycle import SecurityLifecycleResolver
from ashare_quant.models.walk_forward_evaluation import REQUIRED_FOLD_ARTIFACTS
from ashare_quant.utils.manifest import atomic_write_json

CALENDAR = (
    "20240130",
    "20240131",
    "20240201",
    "20240202",
    "20240205",
    "20240206",
    "20240207",
    "20240208",
    "20240209",
    "20240212",
    "20240213",
    "20240214",
)


def settings() -> BacktestSettings:
    return BacktestSettings.model_validate(
        {
            "initial_cash": 1000,
            "holding_period_days": 5,
            "top_n": (10, 20, 50),
            "commission": 0,
            "stamp_duty": 0,
            "slippage": 0,
            "sell_delay_max_days": 1,
        }
    )


def price(
    date: str, code: str = "AAA.SZ", *, suspended: bool = False, terminal: bool = False
) -> dict[str, object]:
    return {
        "trade_date": date,
        "ts_code": code,
        "open": float("nan") if suspended else 10.0,
        "close": float("nan") if suspended else 10.0,
        "can_buy": not suspended and not terminal,
        "can_sell": not suspended and not terminal,
        "is_suspended": suspended,
        "is_listed": not terminal,
        "delist_date": date if terminal else None,
    }


def inputs(*, terminal: bool = False, delayed: bool = False) -> BacktestInputs:
    return BacktestInputs(
        signals=pd.DataFrame(
            [
                {"trade_date": CALENDAR[0], "ts_code": "AAA.SZ", "score": 3.0},
                {"trade_date": CALENDAR[2], "ts_code": "AAA.SZ", "score": 3.0},
                {"trade_date": CALENDAR[2], "ts_code": "BBB.SZ", "score": 2.0},
                {"trade_date": CALENDAR[2], "ts_code": "CCC.SZ", "score": 1.0},
            ]
        ),
        prices=pd.DataFrame(
            [
                price(
                    d,
                    code,
                    suspended=(code == "CCC.SZ" or (delayed and code == "AAA.SZ" and i in (6, 7))),
                    terminal=terminal and code == "AAA.SZ" and i >= 8,
                )
                for i, d in enumerate(CALENDAR)
                for code in ("AAA.SZ", "BBB.SZ", "CCC.SZ")
            ]
        ),
        calendar=CALENDAR,
        benchmark=pd.DataFrame({"trade_date": CALENDAR, "close": [100.0] * len(CALENDAR)}),
        identity_transition_version="none",
        identity_transition_hash=SecurityIdentityTransitionResolver.empty().artifact_hash,
    )


def source_fixture(root: Path) -> Path:
    """Synthetic bytes checked by the real completed-run validator, never a trained model."""
    root.mkdir(parents=True)
    config = root.parent / "config.yaml"
    config.write_text("{}\n")
    lineage = {
        key: "fixture"
        for key in (
            "feature_set_id",
            "feature_set_hash",
            "feature_provenance_hash",
            "walk_forward_plan_hash",
            "horizon_plan_hash",
        )
    }
    shared = {
        "schema_version": 3,
        "evaluation_contract_version": 9,
        "horizon": 5,
        "modeling_identity": "fixture-model",
        "execution_identity": "fixture-execution",
        "execution_contract": {},
        **lineage,
    }
    hashes = {}
    for fold, dates, classification in (
        ("replay", ["20240129"], "RETROSPECTIVE_FIXED_FEATURE_REPLAY"),
        ("first", list(CALENDAR[:2]), "STRICT_OOS"),
        ("second", list(CALENDAR[2:4]), "STRICT_OOS"),
    ):
        path = root / "folds" / fold
        path.mkdir(parents=True)
        for name in REQUIRED_FOLD_ARTIFACTS:
            if name == "predictions.parquet":
                pd.DataFrame(
                    {
                        "trade_date": dates,
                        "ts_code": ["AAA.SZ"] * len(dates),
                        "prediction_score": [1.0] * len(dates),
                    }
                ).to_parquet(path / name, index=False)
            elif name.endswith(".parquet"):
                pd.DataFrame().to_parquet(path / name)
            else:
                (path / name).write_text("{}")
        validity = {"research_classification": classification}
        fm = {
            **shared,
            "artifact_name": "walk_forward_fold_evidence",
            "technical_status": "VALID",
            "experiment_identity": "fixture-identity",
            "research_validity": validity,
            "fold": {
                "fold_id": fold,
                "evaluation_start": dates[0],
                "evaluation_end": dates[-1],
                "research_validity": validity,
            },
            "artifact_hashes": {name: file_hash(path / name) for name in REQUIRED_FOLD_ARTIFACTS},
        }
        atomic_write_json(path / "manifest.json", fm)
        hashes[fold] = file_hash(path / "manifest.json")
    atomic_write_json(root / "aggregate_metrics.json", {})
    pd.DataFrame().to_parquet(root / "fold_summary.parquet")
    atomic_write_json(
        root / "manifest.json",
        {
            **shared,
            "artifact_name": "multi_fold_walk_forward_evidence",
            "status": "COMPLETE",
            "run_id": root.name,
            "identity": "fixture-identity",
            "run_scope": {"mode": "all_eligible_folds"},
            "experiment": {"horizon": 5, "config_hash": file_hash(config)},
            "fold_manifest_hashes": hashes,
            "aggregate_metrics_sha256": file_hash(root / "aggregate_metrics.json"),
            "fold_summary_sha256": file_hash(root / "fold_summary.parquet"),
        },
    )
    return root


def rehash_fold(root: Path, fold: str) -> None:
    path = root / "folds" / fold
    fm = read_json(path / "manifest.json")
    fm["artifact_hashes"] = {name: file_hash(path / name) for name in REQUIRED_FOLD_ARTIFACTS}
    atomic_write_json(path / "manifest.json", fm)
    manifest = read_json(root / "manifest.json")
    manifest["fold_manifest_hashes"][fold] = file_hash(path / "manifest.json")
    atomic_write_json(root / "manifest.json", manifest)


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ContinuousStrictOOSReplayService:
    run = source_fixture(tmp_path / "run")

    def context(self, source, app):
        validate_signal_calendar(source, CALENDAR)
        return ReplayPreflight(
            source,
            {
                "source": source.identity,
                "contract_version": continuous.CONTRACT,
                "execution_inputs": {
                    "governed_execution_cutoff": CALENDAR[-1],
                    "lockbox_start": "20240215",
                },
            },
            CALENDAR,
            settings(),
            SecurityIdentityResolver.empty(),
            SecurityLifecycleResolver.empty(),
            SecurityIdentityTransitionResolver.empty(),
        )

    # Only synthetic market provenance is injected; actual source/root and output validators run.
    monkeypatch.setattr(ContinuousStrictOOSReplayService, "_execution_context", context)
    monkeypatch.setattr(continuous, "load_execution_prices", lambda *a, **kw: inputs().prices)
    monkeypatch.setattr(continuous, "load_benchmark", lambda *a, **kw: inputs().benchmark)
    return ContinuousStrictOOSReplayService(
        source_run=run,
        raw_root=tmp_path / "raw",
        execution_processed_root=tmp_path / "execution",
        lifecycle_scan_manifest=tmp_path / "scan",
        identity_transition_artifact=tmp_path / "transitions",
        config_path=tmp_path / "config.yaml",
        output_root=tmp_path / "output",
    )


def test_source_exact_strict_selection_and_label_independence(tmp_path: Path, monkeypatch) -> None:
    run = source_fixture(tmp_path / "run")
    first = load_continuous_source(run)
    assert first.identity["prediction_rows"] == 4
    assert first.lineage.source_fold_id.tolist() == ["first", "first", "second", "second"]
    assert first.lineage.research_classification.eq("STRICT_OOS").all()
    label = tmp_path / "labels_forward.parquet"
    label.write_bytes(b"changed future outcomes")
    original_read = pd.read_parquet

    def guarded(path, *a, **kw):
        if "label" in str(path):
            raise AssertionError("labels are inaccessible")
        return original_read(path, *a, **kw)

    monkeypatch.setattr(pd, "read_parquet", guarded)
    assert load_continuous_source(run).identity == first.identity
    label.unlink()
    assert load_continuous_source(run).identity == first.identity


@pytest.mark.parametrize(
    "defect", ["duplicate", "nonfinite", "overlap", "missing", "incomplete", "model"]
)
def test_bad_source_rejected(tmp_path: Path, defect: str) -> None:
    run = source_fixture(tmp_path / "run")
    path = run / "folds/second"
    if defect in {"duplicate", "nonfinite"}:
        frame = pd.read_parquet(path / "predictions.parquet")
        if defect == "duplicate":
            frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        else:
            frame.loc[0, "prediction_score"] = float("inf")
        frame.to_parquet(path / "predictions.parquet", index=False)
        rehash_fold(run, "second")
    elif defect == "overlap":
        fm = read_json(path / "manifest.json")
        fm["fold"]["evaluation_start"] = "20240131"
        atomic_write_json(path / "manifest.json", fm)
        rehash_fold(run, "second")
    elif defect == "missing":
        (path / "predictions.parquet").unlink()
    elif defect == "model":
        (path / "model.txt").write_text("tampered")
    else:
        m = read_json(run / "manifest.json")
        m["status"] = "INCOMPLETE"
        atomic_write_json(run / "manifest.json", m)
    with pytest.raises(DataValidationError):
        load_continuous_source(run)


@pytest.mark.parametrize("terminal,delayed", [(False, False), (False, True), (True, True)])
def test_instrumentation_preserves_engine_and_cross_month_account(
    terminal: bool, delayed: bool
) -> None:
    kwargs = {
        "top_n": 10,
        "settings": settings(),
        "purpose": "oos_evidence",
        "delayed_exit_policy": "carry_to_calendar_end",
    }
    data = inputs(terminal=terminal, delayed=delayed)
    old = simulate_portfolio(data, **kwargs)
    new = simulate_portfolio(data, **kwargs, record_execution_audit=True)
    for name in ("daily_returns", "trades", "holdings", "corporate_actions"):
        pd.testing.assert_frame_equal(getattr(old, name), getattr(new, name))
    assert old.metrics == new.metrics
    assert old.accounting_summary == new.accounting_summary
    filled = new.trades.query("side == 'buy' and status == 'filled'")
    assert len(filled) == 1
    held = new.holdings
    assert held.position_id.nunique() == held.shares.nunique() == held.entry_date.nunique() == 1
    assert held.target_exit_date.unique().tolist() == ["20240207"]
    assert held.entry_date.unique().tolist() == ["20240131"]
    assert set(new.execution_intents.outcome) == {
        "FILLED",
        "ALREADY_HELD",
        "NO_AVAILABLE_CASH",
        "NOT_BUYABLE",
    }
    assert new.daily_returns.query("trade_date == '20240201'").cash.iloc[0] == pytest.approx(0)
    recon = accounting_evidence(new, settings())
    assert recon.accounting_residual.abs().max() < 1e-6
    if delayed:
        assert recon.locked_capital_value.max() == 1000
        assert new.accounting_summary["maximum_delayed_exit_days"] == 2
    if terminal:
        assert len(new.terminal_events) == 1
        assert new.terminal_events.pre_writeoff_mark_value.iloc[0] == 1000
        assert new.terminal_events.cash_recovery.iloc[0] == 0


def test_supported_and_unsupported_corporate_action_audit() -> None:
    transition = SecurityIdentityTransition(
        "AAA.SZ",
        "DDD.SZ",
        "old",
        "new",
        "CODE_CHANGE_CONTINUITY",
        "20240201",
        "SAME_LISTED_ENTITY",
        2.0,
        "fixture",
        "a" * 64,
    )
    data = inputs()
    prices = pd.DataFrame(
        [price(d, "AAA.SZ" if i < 2 else "DDD.SZ") for i, d in enumerate(CALENDAR)]
    )
    data = replace(
        data,
        signals=data.signals.iloc[:1],
        prices=prices,
        identity_transitions=(transition,),
        identity_transition_version="fixture",
        identity_transition_hash="b" * 64,
    )
    a = simulate_portfolio(
        data,
        top_n=10,
        settings=settings(),
        purpose="oos_evidence",
        delayed_exit_policy="carry_to_calendar_end",
    )
    b = simulate_portfolio(
        data,
        top_n=10,
        settings=settings(),
        purpose="oos_evidence",
        delayed_exit_policy="carry_to_calendar_end",
        record_execution_audit=True,
    )
    pd.testing.assert_frame_equal(a.daily_returns, b.daily_returns)
    pd.testing.assert_frame_equal(a.trades, b.trades)
    pd.testing.assert_frame_equal(a.corporate_actions, b.corporate_actions)
    assert b.corporate_actions.shape[0] == 1
    accounting_evidence(b, settings())
    with pytest.raises(
        DataValidationError, match="CORPORATE_ACTION_EXECUTION_UNSUPPORTED.*top_n=10.*entry_date"
    ):
        simulate_portfolio(
            replace(data, identity_transitions=(replace(transition, share_conversion_ratio=None),)),
            top_n=10,
            settings=settings(),
            purpose="oos_evidence",
            record_execution_audit=True,
        )


def test_calendar_missing_session_and_tail_fail_before_simulation(tmp_path: Path) -> None:
    source = load_continuous_source(source_fixture(tmp_path / "run"))
    validate_signal_calendar(source, CALENDAR)
    with pytest.raises(DataValidationError, match="EXECUTION_TAIL_INSUFFICIENT"):
        validate_signal_calendar(source, CALENDAR[:4])
    source.lineage.drop(index=1, inplace=True)
    with pytest.raises(DataValidationError, match="MISSING_SIGNAL_SESSION"):
        validate_signal_calendar(source, CALENDAR)


def test_publication_idempotence_and_common_window(service, monkeypatch) -> None:
    context = service.preflight()
    path = service.run()
    manifest = validate_continuous_strict_oos_artifact(path)
    assert path.name == context.run_id
    assert manifest["status"] == "COMPLETE"
    original = file_hash(path / "manifest.json")
    monkeypatch.setattr(
        continuous, "simulate_portfolio", lambda *a, **kw: pytest.fail("duplicate simulation")
    )
    assert service.run() == path
    assert file_hash(path / "manifest.json") == original
    compare = read_json(path / "comparison.json")
    assert set(compare["full_lifecycle"]) == {"10", "20", "50"}
    assert compare["common_comparison_end"] == min(
        compare["actual_execution_end_by_top_n"].values()
    )
    for n in (10, 20, 50):
        daily = pd.read_parquet(path / f"top_{n}/daily_returns.parquet")
        metrics = read_json(path / f"top_{n}/metrics.json")
        assert metrics["total_return"] == pytest.approx((1 + daily.net_return).prod() - 1)


@pytest.mark.parametrize(
    "child",
    [
        "stitched_predictions.parquet",
        "signal_lineage.parquet",
        "top_10/daily_returns.parquet",
        "top_10/trades.parquet",
        "top_10/holdings.parquet",
        "top_10/execution_intents.parquet",
        "top_10/corporate_actions.parquet",
        "top_10/metrics.json",
        "source_inventory.json",
    ],
)
def test_each_child_tamper_rejected(service, child: str) -> None:
    path = service.run()
    with (path / child).open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(DataValidationError, match="CHILD_HASH_MISMATCH"):
        validate_continuous_strict_oos_artifact(path)


def test_recursive_source_tamper_rejected(service) -> None:
    path = service.run()
    (service.source_run / "folds/first/model.txt").write_text("changed source")
    with pytest.raises(DataValidationError, match="WALK_FORWARD_CHILD_ARTIFACT_HASH_MISMATCH"):
        validate_continuous_strict_oos_artifact(path)


def test_rehashed_business_corruption_rejected(service) -> None:
    path = service.run()
    child = "top_10/daily_returns.parquet"
    frame = pd.read_parquet(path / child)
    frame.loc[1, "cash"] += 1
    frame.to_parquet(path / child, index=False)
    manifest = read_json(path / "manifest.json")
    manifest["artifact_hashes"][child] = file_hash(path / child)
    atomic_write_json(path / "manifest.json", manifest)
    with pytest.raises(DataValidationError):
        validate_continuous_strict_oos_artifact(path)


def test_common_window_retains_open_marks_without_forced_sales() -> None:
    a = simulate_portfolio(
        inputs(),
        top_n=10,
        settings=settings(),
        purpose="oos_evidence",
        delayed_exit_policy="carry_to_calendar_end",
    )
    b = simulate_portfolio(
        inputs(delayed=True),
        top_n=20,
        settings=settings(),
        purpose="oos_evidence",
        delayed_exit_policy="carry_to_calendar_end",
    )
    comparison = comparison_metrics([a, b], settings(), "20240201")
    assert comparison["actual_execution_end_by_top_n"] == {"10": "20240207", "20": "20240209"}
    assert comparison["common_comparison_end"] == "20240207"
    assert comparison["common_window"]["20"]["filled_trades"] == 1


def test_cli_preflight_does_not_simulate(service, monkeypatch, capsys) -> None:
    from ashare_quant.cli import main

    monkeypatch.setattr(
        ContinuousStrictOOSReplayService,
        "preflight",
        lambda self: service._execution_context(
            load_continuous_source(service.source_run), AppSettings.model_validate({})
        ),
    )
    monkeypatch.setattr(
        continuous, "simulate_portfolio", lambda *a, **kw: pytest.fail("real simulation")
    )
    assert (
        main(
            [
                "--config",
                str(service.config_path),
                "backtest",
                "continuous-oos",
                "--walk-forward-run-id",
                "fixture",
                "--walk-forward-reports-root",
                str(service.source_run.parent),
                "--execution-processed-root",
                str(service.execution_root),
                "--lifecycle-scan-manifest",
                str(service.scan),
                "--identity-transition-artifact",
                str(service.transitions_path),
                "--preflight-only",
            ]
        )
        == 0
    )
    assert '"real_continuous_replay_status": "NOT_RUN"' in capsys.readouterr().out


@pytest.mark.parametrize("terminal", [False, True])
def test_published_delayed_exit_and_terminal_evidence(service, monkeypatch, terminal) -> None:
    monkeypatch.setattr(
        continuous,
        "load_execution_prices",
        lambda *a, **kw: inputs(delayed=True, terminal=terminal).prices,
    )
    path = service.run()
    validate_continuous_strict_oos_artifact(path)
    summary = read_json(path / "top_10/evidence_summary.json")
    assert summary["maximum_delayed_exit_days"] == 2
    assert summary["days_with_locked_capital"] == 1
    assert summary["breach_resolution_types"] == {"terminal_writeoff" if terminal else "filled": 1}


def test_failed_run_never_publishes_complete(service, monkeypatch) -> None:
    prices = inputs().prices
    prices.loc[
        (prices.ts_code == "AAA.SZ") & (prices.trade_date >= CALENDAR[6]), ["can_sell", "can_buy"]
    ] = False
    monkeypatch.setattr(continuous, "load_execution_prices", lambda *a, **kw: prices)
    with pytest.raises(DataValidationError, match="BACKTEST_UNRESOLVED_POSITION"):
        service.run()
    assert not service.output_root.exists()


def test_replay_allocation_is_not_changed_and_no_cash_intents_are_observable() -> None:
    data = inputs()
    result = simulate_portfolio(
        data, top_n=10, settings=settings(), purpose="oos_evidence", record_execution_audit=True
    )
    assert result.trades.query("side == 'buy' and status == 'filled'").gross_value.sum() == 1000
    assert (
        result.execution_intents.query("ts_code == 'BBB.SZ'").outcome.iloc[0] == "NO_AVAILABLE_CASH"
    )


def test_terminal_and_unaffordable_intents() -> None:
    data = inputs()
    frame = data.prices.copy()
    frame.loc[frame.ts_code.eq("CCC.SZ"), "is_listed"] = False
    frame.loc[frame.ts_code.eq("CCC.SZ"), "delist_date"] = CALENDAR[0]
    result = simulate_portfolio(
        replace(data, prices=frame),
        top_n=10,
        settings=settings(),
        purpose="oos_evidence",
        record_execution_audit=True,
    )
    assert result.execution_intents.query("ts_code == 'CCC.SZ'").outcome.iloc[0] == "TERMINAL"
    config = settings().model_dump()
    config["execution_costs"]["schedules"][0]["minimum_commission"] = 2000
    result = simulate_portfolio(
        data,
        top_n=10,
        settings=BacktestSettings.model_validate(config),
        purpose="oos_evidence",
        record_execution_audit=True,
    )
    assert "INSUFFICIENT_AFFORDABLE_GROSS" in set(result.execution_intents.outcome)


def test_governed_cutoff_and_lockbox_are_enforced_by_validator(service, monkeypatch) -> None:
    original = ContinuousStrictOOSReplayService._execution_context

    def restricted(self, source, app):
        ctx = original(self, source, app)
        ctx.logical["execution_inputs"]["lockbox_start"] = "20240207"
        return ctx

    monkeypatch.setattr(ContinuousStrictOOSReplayService, "_execution_context", restricted)
    with pytest.raises(DataValidationError, match="CONTINUOUS_CUTOFF_VIOLATION"):
        service.run()
    assert not list(service.output_root.glob("continuous_strict_oos_*"))


def test_label_files_cannot_affect_service_signal_identity(service, monkeypatch) -> None:
    first = service.preflight()
    original = pd.read_parquet

    def no_labels(path, *a, **kw):
        assert "labels_forward" not in str(path)
        return original(path, *a, **kw)

    monkeypatch.setattr(pd, "read_parquet", no_labels)
    path = service.run()
    assert path.name == first.run_id


def test_source_fold_reference_tamper_and_changed_identity(service) -> None:
    before = service.preflight()
    fm = service.source_run / "folds/first/manifest.json"
    value = read_json(fm)
    value["artifact_hashes"]["model.txt"] = "0" * 64
    atomic_write_json(fm, value)
    with pytest.raises(DataValidationError):
        service.preflight()
    rehash_fold(service.source_run, "first")
    (service.source_run / "folds/first/model.txt").write_text("another validated fixture model")
    rehash_fold(service.source_run, "first")
    assert service.preflight().run_id != before.run_id


def test_unexpected_top_n_or_extra_child_rejected(service) -> None:
    path = service.run()
    (path / "top_30").mkdir()
    with pytest.raises(DataValidationError, match="CHILD_SET_MISMATCH"):
        validate_continuous_strict_oos_artifact(path)


def test_tampered_holdings_cannot_hide_behind_recomputed_hash(service) -> None:
    path = service.run()
    child = "top_10/holdings.parquet"
    holdings = pd.read_parquet(path / child)
    holdings.loc[0, "shares"] *= 2
    holdings.to_parquet(path / child, index=False)
    manifest = read_json(path / "manifest.json")
    manifest["artifact_hashes"][child] = file_hash(path / child)
    atomic_write_json(path / "manifest.json", manifest)
    with pytest.raises(DataValidationError, match="holding shares"):
        validate_continuous_strict_oos_artifact(path)


@pytest.mark.parametrize("supported", [True, False])
def test_service_corporate_action_publication_or_fail_closed(
    service, monkeypatch, supported
) -> None:
    original = ContinuousStrictOOSReplayService._execution_context
    transition = SecurityIdentityTransition(
        "AAA.SZ",
        "DDD.SZ",
        "old",
        "new",
        "CODE_CHANGE_CONTINUITY",
        "20240201",
        "SAME_LISTED_ENTITY",
        2.0 if supported else None,
        "fixture",
        "a" * 64,
    )
    resolver = SecurityIdentityTransitionResolver(
        artifact_version="fixture", artifact_hash="b" * 64, transitions=(transition,)
    )

    def context(self, source, app):
        base = original(self, source, app)
        base.logical["execution_inputs"]["transition_hash"] = resolver.artifact_hash
        return replace(base, transitions=resolver)

    monkeypatch.setattr(ContinuousStrictOOSReplayService, "_execution_context", context)
    prices = pd.DataFrame(
        [price(d, "AAA.SZ" if i < 2 else "DDD.SZ") for i, d in enumerate(CALENDAR)]
    )
    monkeypatch.setattr(continuous, "load_execution_prices", lambda *a, **kw: prices)
    if supported:
        path = service.run()
        validate_continuous_strict_oos_artifact(path)
        assert len(pd.read_parquet(path / "top_10/corporate_actions.parquet")) == 1
    else:
        with pytest.raises(
            DataValidationError, match="UNSUPPORTED_REAL_TRANSITION_CROSSING.*top_n=10.*entry_date"
        ):
            service.run()
        assert not service.output_root.exists()


def test_source_bytes_unchanged_by_service(service) -> None:
    before = {
        p.relative_to(service.source_run): file_hash(p)
        for p in service.source_run.rglob("*")
        if p.is_file()
    }
    service.run()
    after = {
        p.relative_to(service.source_run): file_hash(p)
        for p in service.source_run.rglob("*")
        if p.is_file()
    }
    assert before == after


def test_selected_fold_horizon_must_match_experiment(service) -> None:
    path = service.source_run / "folds/first/manifest.json"
    fold = read_json(path)
    fold["horizon"] = 10
    atomic_write_json(path, fold)
    rehash_fold(service.source_run, "first")
    with pytest.raises(DataValidationError, match="CONTINUOUS_SOURCE_HORIZON_MISMATCH"):
        service.preflight()
