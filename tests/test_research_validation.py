from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_quant.backtest.executable_validation import _signals
from ashare_quant.config.settings import AppSettings, PathSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_lifecycle import SecurityLifecycleResolver
from ashare_quant.models.compute.benchmark import TrainingBackendBenchmarkService
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.feature_provenance import (
    FeatureSetProvenance,
    create_governed_feature_set,
    load_feature_set_provenance,
    validate_governed_feature_set,
)
from ashare_quant.models.ranker_data import RankerPredictionDataset
from ashare_quant.models.research_policy import enforce_research_window, load_research_policy
from ashare_quant.models.temporal_isolation import (
    required_temporal_gap_sessions,
    resolve_temporal_gaps,
)
from ashare_quant.models.walk_forward_evaluation import (
    EXECUTION_TAIL_LOAD_CHUNK_SESSIONS,
    EXECUTION_TAIL_POLICY,
    EXECUTION_TAIL_POLICY_VERSION,
    FoldExecutionResult,
    MultiFoldEvaluationRunner,
    RankerFoldExecutor,
    WalkForwardRecoveryInspector,
    _build_prediction_frame,
    _delayed_exit_evidence,
    _governed_execution_cutoff,
    _optional_ratio,
    _ranking_metrics,
    _with_research_classification,
    walk_forward_status,
)
from ashare_quant.utils.manifest import atomic_write_json


def test_horizon_safe_gap_resolution() -> None:
    assert [required_temporal_gap_sessions(value) for value in (5, 10, 20, 60)] == [
        6,
        11,
        21,
        61,
    ]
    for horizon, expected in ((5, 6), (10, 11), (20, 21), (60, 61)):
        resolved = resolve_temporal_gaps((horizon,), purge="auto", embargo="auto")
        assert resolved.resolved_purge == expected
        assert resolved.resolved_embargo == expected
        assert resolved.gap_policy == "AUTO"


def test_gap_override_and_shared_multi_horizon_policy() -> None:
    shared = resolve_temporal_gaps((5, 10, 20, 60), purge="auto", embargo="auto")
    assert shared.required_gap == 61
    assert shared.resolved_purge == 61
    assert resolve_temporal_gaps((20,), purge=21, embargo=30).resolved_embargo == 30
    with pytest.raises(DataValidationError, match="unsafe explicit purge_sessions"):
        resolve_temporal_gaps((20,), purge=20, embargo=21)


def test_multi_fold_prediction_schema_matches_shared_executable_adapter() -> None:
    evaluation = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240102"],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "unused_feature": [1.0, 2.0],
        }
    )
    scores = np.asarray([0.25, -0.5], dtype=float)

    predictions = _build_prediction_frame(evaluation, scores)
    assert predictions.columns.tolist() == ["trade_date", "ts_code", "prediction_score"]
    signals = _signals(predictions)
    assert signals.columns.tolist() == ["trade_date", "ts_code", "score"]
    assert signals["score"].tolist() == scores.tolist()

    with pytest.raises(KeyError, match="prediction_score"):
        _signals(predictions.rename(columns={"prediction_score": "score"}))


def test_evaluation_label_changes_do_not_change_predictions_or_signals() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20240102"] * 5,
            "ts_code": [f"{index:06d}.SZ" for index in range(5)],
            "f1": np.arange(5, dtype=np.float32),
        }
    )
    scores = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0])
    frozen = _build_prediction_frame(frame, scores)
    first_labels = frozen.assign(
        exit_date="20240110",
        future_excess_ret_5d=np.linspace(0.01, 0.05, 5),
        is_label_available=True,
        label_unavailable_reason="",
        is_label_mature=True,
        relevance=pd.Series([0, 1, 2, 3, 4], dtype="Int32"),
    )
    changed_labels = first_labels.copy()
    changed_labels.loc[0, ["future_excess_ret_5d", "is_label_available"]] = [np.nan, False]
    changed_labels.loc[0, "label_unavailable_reason"] = "missing_exit_price"
    dataset = RankerPredictionDataset(
        frame=frame,
        feature_names=("f1",),
        coverage_by_date=pd.DataFrame(
            {
                "trade_date": ["20240102"],
                "expected_universe_rows": [5],
                "feature_rows_present": [5],
            }
        ),
    )
    settings = AppSettings.model_validate({"ranker": {"minimum_group_size": 3}})

    first = _ranking_metrics(dataset, first_labels, settings)
    changed = _ranking_metrics(dataset, changed_labels, settings)

    pd.testing.assert_frame_equal(_signals(frozen), _signals(frozen.copy()))
    assert frozen.iloc[0]["ts_code"] == "000000.SZ"
    top_name = "top_5pct_mean_future_excess_ret"
    assert first["top_n_label_proxies"][top_name]["available_label_rows"] == 1
    assert changed["top_n_label_proxies"][top_name]["available_label_rows"] == 0
    assert changed["top_n_label_proxies"][top_name]["unavailable_label_rows"] == 1


def test_prediction_and_label_coverage_use_distinct_denominators() -> None:
    count = 90
    frame = pd.DataFrame(
        {
            "trade_date": ["20240102"] * count,
            "ts_code": [f"{index:06d}.SZ" for index in range(count)],
            "f1": np.arange(count, dtype=np.float32),
        }
    )
    predictions = _build_prediction_frame(frame, np.arange(count, dtype=float))
    predictions = predictions.assign(
        exit_date="20240110",
        future_excess_ret_5d=np.arange(count, dtype=float) / 100,
        is_label_available=[True] * 80 + [False] * 10,
        label_unavailable_reason=[""] * 80 + ["missing_exit_price"] * 10,
        is_label_mature=True,
        relevance=pd.Series(list(range(5)) * 18, dtype="Int32"),
    )
    dataset = RankerPredictionDataset(
        frame=frame,
        feature_names=("f1",),
        coverage_by_date=pd.DataFrame(
            {
                "trade_date": ["20240102"],
                "expected_universe_rows": [100],
                "feature_rows_present": [90],
            }
        ),
    )

    metrics = _ranking_metrics(dataset, predictions, AppSettings.model_validate({}))

    assert metrics["prediction_coverage"] == pytest.approx(0.9)
    assert metrics["label_coverage"] == pytest.approx(80 / 90)
    assert metrics["unavailable_label_rows"] == 10
    assert _optional_ratio(0, 0) is None


def test_feature_selection_oos_classification_uses_information_maturity() -> None:
    fold = {
        "fold_id": "fold_fixture",
        "validation_end": "20200131",
        "evaluation_start": "20200203",
    }
    strict = _with_research_classification(fold, "20200130")
    replay = _with_research_classification(fold, "20200210")

    assert strict["research_validity"]["research_classification"] == "STRICT_OOS"
    assert replay["research_validity"]["research_classification"] == (
        "RETROSPECTIVE_FIXED_FEATURE_REPLAY"
    )


def test_research_lockbox_enforcement() -> None:
    policy = load_research_policy(Path("config/research_policy.yaml"))
    enforce_research_window(
        policy,
        consumer="feature_selection",
        start_date="20200101",
        end_date="20260809",
    )
    with pytest.raises(DataValidationError, match="RESEARCH_LOCKBOX_VIOLATION"):
        enforce_research_window(
            policy,
            consumer="feature_selection",
            start_date="20200101",
            end_date="20260810",
        )
    enforce_research_window(
        policy,
        consumer="production_shadow",
        start_date="20260810",
        end_date="20260810",
    )


def test_legacy_robust_feature_provenance_is_readable_but_not_governed() -> None:
    path = Path("config/feature_sets/robust_20_v1.provenance.json")
    provenance = load_feature_set_provenance(path)
    assert provenance.provenance_status == "LEGACY_PROVENANCE_INCOMPLETE"
    with pytest.raises(DataValidationError, match="LEGACY_PROVENANCE_INCOMPLETE"):
        validate_governed_feature_set(path)


def test_governed_feature_provenance_requires_complete_sources(tmp_path: Path) -> None:
    features = ("f1", "f2")
    with pytest.raises(ValueError, match="complete selection provenance"):
        FeatureSetProvenance(
            schema_version=1,
            artifact_name="feature_set_provenance",
            feature_set_name="fixture",
            feature_set_version="v1",
            provenance_status="GOVERNED",
            features=features,
            feature_list_hash=feature_list_hash(features),
            selection_policy="fixture",
        )


def test_governed_feature_set_is_created_from_exact_diagnostics(tmp_path: Path) -> None:
    diagnostics = tmp_path / "reports" / "feature_diagnostics" / "run_1"
    diagnostics.mkdir(parents=True)
    atomic_write_json(
        diagnostics / "manifest.json",
        {
            "artifact_name": "feature_diagnostics",
            "split": {
                "train_start": "20100101",
                "train_end": "20181231",
                "validation_start": "20190101",
                "validation_end": "20201231",
            },
            "source_manifests": {"features_daily": {"sha256": "features-fixture"}},
        },
    )
    atomic_write_json(
        diagnostics / "recommended_features.json",
        {"recommended_features": ["f1", "f2"]},
    )
    created = create_governed_feature_set(
        diagnostics_dir=diagnostics,
        output_root=tmp_path / "reports" / "feature_selection",
        feature_set_name="fixture",
        feature_set_version="v1",
        created_by="pytest",
    )
    provenance = validate_governed_feature_set(created)
    assert provenance.provenance_status == "GOVERNED"
    assert provenance.selection_end == "20201231"
    assert (created.parent / "manifest.json").is_file()


def test_multi_fold_runner_executes_all_folds_and_is_idempotent(tmp_path: Path) -> None:
    plan, provenance = _research_fixture(tmp_path)
    executor = FakeExecutor()
    runner = MultiFoldEvaluationRunner(
        reports_root=tmp_path / "reports",
        settings=AppSettings.model_validate({}),
        executor=executor,
        research_policy_path=Path("config/research_policy.yaml"),
        lifecycle_audit_required=False,
    )

    first = runner.run(
        experiment_manifest=plan,
        experiment_id="h5_fixture",
        feature_provenance_path=provenance,
        require_executable=True,
    )
    manifest_bytes = (first.output_dir / "manifest.json").read_bytes()
    second = runner.run(
        experiment_manifest=plan,
        experiment_id="h5_fixture",
        feature_provenance_path=provenance,
        require_executable=True,
    )

    assert first.status == "COMPLETE"
    assert first.fold_count == 3
    assert executor.calls == ["fold_1", "fold_2", "fold_3"]
    assert second.experiment_id == first.experiment_id
    assert (first.output_dir / "manifest.json").read_bytes() == manifest_bytes
    assert len(list((first.output_dir / "folds").glob("*/manifest.json"))) == 3
    aggregate = json.loads((first.output_dir / "aggregate_metrics.json").read_text())
    assert aggregate["technical"]["all_required_folds_valid"] is True
    assert aggregate["performance"]["rank_ic"]["minimum"] == pytest.approx(0.01)
    assert aggregate["executable_performance"]["status"] == "COMPLETE"
    assert walk_forward_status(tmp_path / "reports", first.experiment_id)["fold_count"] == 3
    assert WalkForwardRecoveryInspector(tmp_path / "reports").inspect(first.experiment_id) == {
        "status": "CLEAN",
        "issues": [],
    }


def test_multi_fold_runner_reuses_completed_folds_before_final_manifest(tmp_path: Path) -> None:
    plan, provenance = _research_fixture(tmp_path)
    executor = FakeExecutor(fail_once="fold_3")
    runner = MultiFoldEvaluationRunner(
        reports_root=tmp_path / "reports",
        settings=AppSettings.model_validate({}),
        executor=executor,
        lifecycle_audit_required=False,
    )
    with pytest.raises(DataValidationError, match="synthetic interruption"):
        runner.run(
            experiment_manifest=plan,
            experiment_id="h5_fixture",
            feature_provenance_path=provenance,
        )

    result = runner.run(
        experiment_manifest=plan,
        experiment_id="h5_fixture",
        feature_provenance_path=provenance,
    )
    assert result.status == "COMPLETE"
    assert executor.calls == ["fold_1", "fold_2", "fold_3", "fold_3"]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"backend": "cpu"}, {"backend": "cuda"}),
        ({"cost_policy_hash": "cost-a"}, {"cost_policy_hash": "cost-b"}),
        ({"contract_version": 3}, {"contract_version": 4}),
        ({"tail_policy": "carry-v1"}, {"tail_policy": "carry-v2"}),
    ],
)
def test_execution_contract_changes_create_distinct_runs(
    tmp_path: Path,
    first: dict[str, object],
    second: dict[str, object],
) -> None:
    plan, provenance = _research_fixture(tmp_path)
    settings = AppSettings.model_validate({})
    first_result = MultiFoldEvaluationRunner(
        reports_root=tmp_path / "reports",
        settings=settings,
        executor=FakeExecutor(**first),
        lifecycle_audit_required=False,
    ).run(
        experiment_manifest=plan,
        experiment_id="h5_fixture",
        feature_provenance_path=provenance,
    )
    second_result = MultiFoldEvaluationRunner(
        reports_root=tmp_path / "reports",
        settings=settings,
        executor=FakeExecutor(**second),
        lifecycle_audit_required=False,
    ).run(
        experiment_manifest=plan,
        experiment_id="h5_fixture",
        feature_provenance_path=provenance,
    )

    assert first_result.experiment_id != second_result.experiment_id


def test_walk_forward_execution_tail_contract_is_bounded_by_source_and_lockbox(
    tmp_path: Path,
) -> None:
    universe = tmp_path / "processed" / "universe_daily"
    universe.mkdir(parents=True)
    atomic_write_json(
        universe / "_manifest.json",
        {"canonical_artifact": {"max_date": "20260930"}},
    )
    executor = RankerFoldExecutor(
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        settings=AppSettings.model_validate({}),
    )

    contract = executor.execution_contract(
        horizon=5,
        require_executable=True,
        prospective_lockbox_start="20260810",
    )
    tail = contract["execution_tail"]

    assert isinstance(tail, dict)
    assert tail["policy"] == EXECUTION_TAIL_POLICY
    assert tail["sell_delay_alert_sessions"] == 20
    assert tail["security_lifecycle_policy_version"] == "security_lifecycle_events_v3"
    assert _governed_execution_cutoff(tail) == "20260809"


def test_delayed_exit_evidence_records_exact_resolution_date() -> None:
    trades = pd.DataFrame(
        [
            {
                "trade_date": "20170213",
                "position_id": "20:20161230:300319.SZ",
                "ts_code": "300319.SZ",
                "side": "sell",
                "status": "rejected",
                "delayed_exit_days": 21,
                "sell_delay_breached": True,
            },
            {
                "trade_date": "20170329",
                "position_id": "20:20161230:300319.SZ",
                "ts_code": "300319.SZ",
                "side": "sell",
                "status": "filled",
                "delayed_exit_days": 52,
                "sell_delay_breached": True,
            },
        ]
    )

    evidence = _delayed_exit_evidence(trades)

    assert evidence["breached_positions"] == 1
    assert evidence["maximum_delayed_exit_sessions"] == 52
    assert evidence["resolutions"] == [
        {
            "position_id": "20:20161230:300319.SZ",
            "ts_code": "300319.SZ",
            "resolution_date": "20170329",
            "resolution_status": "filled",
            "delayed_exit_sessions": 52,
        }
    ]


def test_walk_forward_executable_metrics_carries_suspension_to_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spring_festival = {"20170127", "20170130", "20170131", "20170201", "20170202"}
    suspension_dates = tuple(
        value.strftime("%Y%m%d")
        for value in pd.bdate_range("2017-01-03", "2017-03-28")
        if value.strftime("%Y%m%d") not in spring_festival
    )
    dates = ("20161229", "20161230", *suspension_dates, "20170329")
    assert len(suspension_dates) == 56
    prices = pd.DataFrame(
        [
            {
                "trade_date": date,
                "ts_code": "300319.SZ",
                "open": 10.0 if index < 2 else (9.0 if index == 58 else np.nan),
                "close": 10.0 if index < 2 else (9.0 if index == 58 else np.nan),
                "can_buy": index < 2 or index == 58,
                "can_sell": index < 2 or index == 58,
                "is_suspended": 2 <= index < 58,
                "is_listed": True,
                "delist_date": None,
            }
            for index, date in enumerate(dates)
        ]
    )
    observed: dict[str, object] = {}

    def calendar_stub(
        raw_root: Path,
        start_date: str,
        end_date: str,
        holding_period: int | None,
        *,
        maximum_date: str | None = None,
    ) -> list[str]:
        del raw_root, start_date, end_date
        observed["holding_period"] = holding_period
        observed["maximum_date"] = maximum_date
        return list(dates)

    monkeypatch.setattr("ashare_quant.models.walk_forward_evaluation.load_calendar", calendar_stub)
    monkeypatch.setattr(
        "ashare_quant.models.walk_forward_evaluation.load_execution_prices",
        lambda *args, **kwargs: prices,
    )
    monkeypatch.setattr(
        "ashare_quant.models.walk_forward_evaluation.load_benchmark",
        lambda *args, **kwargs: pd.DataFrame({"trade_date": dates, "close": [100.0] * len(dates)}),
    )
    executor = RankerFoldExecutor(
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        settings=AppSettings.model_validate(
            {
                "backtest": {
                    "initial_cash": 1000.0,
                    "commission": 0.0,
                    "stamp_duty": 0.0,
                    "slippage": 0.0,
                    "sell_delay_max_days": 20,
                }
            }
        ),
    )
    contract = {
        "execution_tail": {
            "policy_version": EXECUTION_TAIL_POLICY_VERSION,
            "policy": EXECUTION_TAIL_POLICY,
            "sell_delay_alert_sessions": 20,
            "processed_data_end": "20260710",
            "prospective_lockbox_start_exclusive": "20260810",
            "unresolved_at_cutoff": "fail_closed",
            **SecurityLifecycleResolver.from_path(
                Path("config/security_identity/security_lifecycle_events.json")
            ).provenance(),
        }
    }

    metrics = executor._executable_metrics(
        pd.DataFrame(
            {
                "trade_date": [dates[0]],
                "ts_code": ["300319.SZ"],
                "prediction_score": [1.0],
            }
        ),
        5,
        contract,
    )

    assert metrics["status"] == "COMPLETE"
    assert observed == {
        "holding_period": None,
        "maximum_date": "20260710",
    }
    top20 = metrics["delayed_exit_evidence"]["20"]
    assert top20["maximum_delayed_exit_sessions"] == 52
    assert top20["resolutions"][0]["resolution_date"] == "20170329"
    assert metrics["execution_tail"]["actual_execution_end"] == "20170329"


def test_walk_forward_execution_tail_extends_by_chunks_until_authoritative_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = tuple(value.strftime("%Y%m%d") for value in pd.bdate_range("2018-08-01", periods=430))
    resume_index = 400
    all_prices = pd.DataFrame(
        [
            {
                "trade_date": date,
                "ts_code": "300028.SZ",
                "open": 10.0 if index < 2 else (0.69 if index == resume_index else np.nan),
                "close": 10.0 if index < 2 else (0.69 if index == resume_index else np.nan),
                "can_buy": index < 2 or index == resume_index,
                "can_sell": index < 2 or index == resume_index,
                "is_suspended": 2 <= index < resume_index,
                "is_listed": True,
                "delist_date": "20200803",
            }
            for index, date in enumerate(dates)
        ]
    )
    loaded_through: list[str] = []

    monkeypatch.setattr(
        "ashare_quant.models.walk_forward_evaluation.load_calendar",
        lambda *args, **kwargs: list(dates),
    )

    def prices_stub(*args: object, **kwargs: object) -> pd.DataFrame:
        end_date = str(args[3])
        loaded_through.append(end_date)
        assert kwargs["ts_codes"] == ("300028.SZ",)
        return all_prices[all_prices["trade_date"] <= end_date].reset_index(drop=True)

    monkeypatch.setattr(
        "ashare_quant.models.walk_forward_evaluation.load_execution_prices", prices_stub
    )
    monkeypatch.setattr(
        "ashare_quant.models.walk_forward_evaluation.load_benchmark",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "trade_date": [date for date in dates if date <= str(args[3])],
                "close": 100.0,
            }
        ),
    )
    executor = RankerFoldExecutor(
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        settings=AppSettings.model_validate(
            {
                "backtest": {
                    "initial_cash": 1000.0,
                    "commission": 0.0,
                    "stamp_duty": 0.0,
                    "slippage": 0.0,
                    "sell_delay_max_days": 20,
                }
            }
        ),
    )
    lifecycle = SecurityLifecycleResolver.from_path(
        Path("config/security_identity/security_lifecycle_events.json")
    )
    contract = {
        "execution_tail": {
            "policy_version": EXECUTION_TAIL_POLICY_VERSION,
            "policy": EXECUTION_TAIL_POLICY,
            "sell_delay_alert_sessions": 20,
            "processed_data_end": dates[-1],
            "prospective_lockbox_start_exclusive": "20260810",
            "unresolved_at_cutoff": "fail_closed",
            **lifecycle.provenance(),
        }
    }

    metrics = executor._executable_metrics(
        pd.DataFrame(
            {
                "trade_date": [dates[0]],
                "ts_code": ["300028.SZ"],
                "prediction_score": [1.0],
            }
        ),
        5,
        contract,
    )

    assert len(loaded_through) == 2
    assert loaded_through[0] == dates[5 + EXECUTION_TAIL_LOAD_CHUNK_SESSIONS + 1]
    assert loaded_through[1] == dates[-1]
    assert metrics["execution_tail"]["actual_execution_end"] == dates[resume_index]


def test_walk_forward_executable_tail_rejects_calendar_crossing_lockbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ashare_quant.models.walk_forward_evaluation.load_calendar",
        lambda *args, **kwargs: ["20260807", "20260810"],
    )
    executor = RankerFoldExecutor(
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        settings=AppSettings.model_validate({}),
    )
    contract = {
        "execution_tail": {
            "policy_version": EXECUTION_TAIL_POLICY_VERSION,
            "policy": EXECUTION_TAIL_POLICY,
            "sell_delay_alert_sessions": 20,
            "processed_data_end": "20260930",
            "prospective_lockbox_start_exclusive": "20260810",
            "unresolved_at_cutoff": "fail_closed",
            **SecurityLifecycleResolver.from_path(
                Path("config/security_identity/security_lifecycle_events.json")
            ).provenance(),
        }
    }

    with pytest.raises(DataValidationError, match="RESEARCH_LOCKBOX_VIOLATION"):
        executor._executable_metrics(
            pd.DataFrame(
                {
                    "trade_date": ["20260807"],
                    "ts_code": ["000001.SZ"],
                    "prediction_score": [1.0],
                }
            ),
            5,
            contract,
        )


def test_multi_fold_runner_rejects_corrupt_completed_fold(tmp_path: Path) -> None:
    plan, provenance = _research_fixture(tmp_path)
    executor = FakeExecutor(fail_once="fold_3")
    runner = MultiFoldEvaluationRunner(
        reports_root=tmp_path / "reports",
        settings=AppSettings.model_validate({}),
        executor=executor,
        lifecycle_audit_required=False,
    )
    with pytest.raises(DataValidationError):
        runner.run(
            experiment_manifest=plan,
            experiment_id="h5_fixture",
            feature_provenance_path=provenance,
        )
    fold_one = next((tmp_path / "reports" / "research" / "walk_forward").glob("*/folds/fold_1"))
    (fold_one / "ranking_metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DataValidationError, match="HASH_MISMATCH"):
        runner.run(
            experiment_manifest=plan,
            experiment_id="h5_fixture",
            feature_provenance_path=provenance,
        )


def test_executable_multi_fold_requires_lifecycle_audit_by_default(tmp_path: Path) -> None:
    plan, provenance = _research_fixture(tmp_path)
    runner = MultiFoldEvaluationRunner(
        reports_root=tmp_path / "reports",
        settings=AppSettings.model_validate({}),
        executor=FakeExecutor(),
    )

    with pytest.raises(DataValidationError, match="SECURITY_LIFECYCLE_AUDIT_REQUIRED"):
        runner.run(
            experiment_manifest=plan,
            experiment_id="h5_fixture",
            feature_provenance_path=provenance,
        )


def test_backend_benchmark_source_consumes_latest_selection_fold_without_model_copy(
    tmp_path: Path,
) -> None:
    plan, provenance = _research_fixture(tmp_path)
    reports_root = tmp_path / "reports"
    result = MultiFoldEvaluationRunner(
        reports_root=reports_root,
        settings=AppSettings.model_validate({}),
        executor=FakeExecutor(),
        lifecycle_audit_required=False,
    ).run(
        experiment_manifest=plan,
        experiment_id="h5_fixture",
        feature_provenance_path=provenance,
    )
    service = TrainingBackendBenchmarkService(
        AppSettings(
            paths=PathSettings(
                raw_data=tmp_path / "raw",
                processed_data=tmp_path / "processed",
                models=tmp_path / "models-does-not-exist",
                reports=reports_root,
            )
        )
    )

    source = service._walk_forward_source(result.experiment_id, None, provenance)

    assert source.source_kind == "walk_forward_fold"
    assert source.lineage["fold_id"] == "fold_2"
    assert source.features == ("f1", "f2")
    with pytest.raises(DataValidationError, match="selection period"):
        service._walk_forward_source(result.experiment_id, "fold_3", provenance)


@pytest.mark.parametrize(
    "mutation",
    ("aggregate", "summary", "fold_manifest", "fold_child", "extra_fold", "missing_fold"),
)
def test_completed_walk_forward_tamper_fails_status_resume_and_recovery(
    tmp_path: Path,
    mutation: str,
) -> None:
    plan, provenance = _research_fixture(tmp_path)
    runner = MultiFoldEvaluationRunner(
        reports_root=tmp_path / "reports",
        settings=AppSettings.model_validate({}),
        executor=FakeExecutor(),
        lifecycle_audit_required=False,
    )
    result = runner.run(
        experiment_manifest=plan,
        experiment_id="h5_fixture",
        feature_provenance_path=provenance,
    )
    if mutation == "aggregate":
        target = result.output_dir / "aggregate_metrics.json"
        target.write_bytes(target.read_bytes() + b"\n")
    elif mutation == "summary":
        target = result.output_dir / "fold_summary.parquet"
        target.write_bytes(target.read_bytes() + b"tamper")
    elif mutation == "fold_manifest":
        target = result.output_dir / "folds" / "fold_1" / "manifest.json"
        target.write_bytes(target.read_bytes() + b"\n")
    elif mutation == "fold_child":
        target = result.output_dir / "folds" / "fold_1" / "ranking_metrics.json"
        target.write_text("{}", encoding="utf-8")
    elif mutation == "extra_fold":
        extra = result.output_dir / "folds" / "unexpected_fold"
        extra.mkdir()
        atomic_write_json(extra / "manifest.json", {"artifact_name": "unexpected"})
    else:
        shutil.rmtree(result.output_dir / "folds" / "fold_1")
    root_manifest = (result.output_dir / "manifest.json").read_bytes()

    with pytest.raises(DataValidationError):
        walk_forward_status(tmp_path / "reports", result.experiment_id)
    with pytest.raises(DataValidationError):
        runner.run(
            experiment_manifest=plan,
            experiment_id="h5_fixture",
            feature_provenance_path=provenance,
        )
    recovery = WalkForwardRecoveryInspector(tmp_path / "reports").inspect(result.experiment_id)
    assert recovery["status"] == "ACTION_REQUIRED"
    assert recovery["issues"]
    assert (result.output_dir / "manifest.json").read_bytes() == root_manifest


def test_feature_set_identity_is_path_and_time_independent_and_relocatable(
    tmp_path: Path,
) -> None:
    _, first_path = _research_fixture(tmp_path / "checkout_a")
    _, independent_path = _research_fixture(tmp_path / "checkout_b")
    first = load_feature_set_provenance(first_path)
    independent = load_feature_set_provenance(independent_path)
    assert independent.feature_set_id == first.feature_set_id
    second = first.model_copy(
        update={
            "created_at": "2030-01-01T00:00:00+00:00",
            "source_diagnostics_manifest_path": "/different/checkout/manifest.json",
        }
    )
    assert second.feature_set_id == first.feature_set_id

    relocated_root = tmp_path / "checkout_c" / "reports"
    shutil.copytree(tmp_path / "checkout_a" / "reports", relocated_root)
    relocated_path = relocated_root / first_path.relative_to(tmp_path / "checkout_a" / "reports")
    validated = validate_governed_feature_set(relocated_path, reports_root=relocated_root)
    assert validated.feature_set_id == first.feature_set_id


def test_feature_provenance_source_missing_or_changed_fails_closed(tmp_path: Path) -> None:
    _, provenance_path = _research_fixture(tmp_path)
    provenance = load_feature_set_provenance(provenance_path)
    source = tmp_path / "reports" / str(provenance.source_diagnostics_manifest_locator)
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(DataValidationError, match="SOURCE_HASH_MISMATCH"):
        validate_governed_feature_set(provenance_path, reports_root=tmp_path / "reports")
    source.unlink()
    with pytest.raises(DataValidationError, match="SOURCE_MISSING"):
        validate_governed_feature_set(provenance_path, reports_root=tmp_path / "reports")


def test_multi_fold_rejects_different_or_tampered_feature_provenance(tmp_path: Path) -> None:
    plan, provenance_path = _research_fixture(tmp_path)
    original = load_feature_set_provenance(provenance_path)
    mismatched = original.model_copy(
        update={
            "feature_set_name": "different",
            "created_at": "2026-08-09T01:00:00+00:00",
        }
    )
    different_path = provenance_path.parent.parent / "different" / "feature_set.json"
    different_path.parent.mkdir()
    atomic_write_json(different_path, mismatched.model_dump(mode="json"))
    runner = MultiFoldEvaluationRunner(
        reports_root=tmp_path / "reports",
        settings=AppSettings.model_validate({}),
        executor=FakeExecutor(),
        lifecycle_audit_required=False,
    )
    with pytest.raises(DataValidationError, match="FEATURE_PROVENANCE_MISMATCH"):
        runner.run(
            experiment_manifest=plan,
            experiment_id="h5_fixture",
            feature_provenance_path=different_path,
        )


def test_multi_fold_runner_rejects_chronology_and_lockbox(tmp_path: Path) -> None:
    plan, provenance = _research_fixture(tmp_path)
    payload = json.loads(plan.read_text())
    folds_path = Path(payload["folds_manifest"]).parent / "folds.json"
    folds = json.loads(folds_path.read_text())
    folds["folds"][0]["validation_start"] = "20181231"
    atomic_write_json(folds_path, folds)
    payload["folds_hash"] = hashlib.sha256(folds_path.read_bytes()).hexdigest()
    atomic_write_json(plan, payload)
    runner = MultiFoldEvaluationRunner(
        reports_root=tmp_path / "reports",
        settings=AppSettings.model_validate({}),
        executor=FakeExecutor(),
        lifecycle_audit_required=False,
    )
    with pytest.raises(DataValidationError, match="chronology"):
        runner.run(
            experiment_manifest=plan,
            experiment_id="h5_fixture",
            feature_provenance_path=provenance,
        )


def test_multi_fold_runner_rejects_source_hash_and_duplicate_fold(tmp_path: Path) -> None:
    plan, provenance = _research_fixture(tmp_path)
    payload = json.loads(plan.read_text())
    folds_path = Path(payload["folds_manifest"]).parent / "folds.json"
    folds = json.loads(folds_path.read_text())
    folds["folds"][0]["evaluation_end"] = "20200103"
    atomic_write_json(folds_path, folds)
    runner = MultiFoldEvaluationRunner(
        reports_root=tmp_path / "reports",
        settings=AppSettings.model_validate({}),
        executor=FakeExecutor(),
        lifecycle_audit_required=False,
    )
    with pytest.raises(DataValidationError, match="folds hash changed"):
        runner.run(
            experiment_manifest=plan,
            experiment_id="h5_fixture",
            feature_provenance_path=provenance,
        )

    folds["folds"][1]["fold_id"] = "fold_1"
    atomic_write_json(folds_path, folds)
    payload["folds_hash"] = hashlib.sha256(folds_path.read_bytes()).hexdigest()
    atomic_write_json(plan, payload)
    with pytest.raises(DataValidationError, match="non-empty and unique"):
        runner.run(
            experiment_manifest=plan,
            experiment_id="h5_fixture",
            feature_provenance_path=provenance,
        )


def test_multi_fold_runner_rejects_prospective_lockbox_fold(tmp_path: Path) -> None:
    plan, provenance = _research_fixture(tmp_path)
    payload = json.loads(plan.read_text())
    payload["experiments"][0]["final_test_period"]["folds"][0].update(
        {"evaluation_start": "20260810", "evaluation_end": "20260810"}
    )
    atomic_write_json(plan, payload)
    runner = MultiFoldEvaluationRunner(
        reports_root=tmp_path / "reports",
        settings=AppSettings.model_validate({}),
        executor=FakeExecutor(),
        lifecycle_audit_required=False,
    )
    with pytest.raises(DataValidationError, match="RESEARCH_LOCKBOX_VIOLATION"):
        runner.run(
            experiment_manifest=plan,
            experiment_id="h5_fixture",
            feature_provenance_path=provenance,
        )


class FakeExecutor:
    def __init__(
        self,
        fail_once: str | None = None,
        *,
        backend: str = "cpu",
        contract_version: int = 3,
        cost_policy_hash: str = "cost-fixture",
        tail_policy: str = "carry-v1",
    ) -> None:
        self.calls: list[str] = []
        self.fail_once = fail_once
        self.backend = backend
        self.contract_version = contract_version
        self.cost_policy_hash = cost_policy_hash
        self.tail_policy = tail_policy

    def validate_sources(self, plan: dict[str, object]) -> dict[str, object]:
        return {
            "features_manifest_hash": plan.get("features_manifest_hash", "features-fixture"),
            "universe_manifest_hash": plan.get("universe_hash"),
            "labels_fingerprint": plan.get("labels_fingerprint"),
        }

    def execution_contract(
        self,
        *,
        horizon: int,
        require_executable: bool,
        prospective_lockbox_start: str,
    ) -> dict[str, object]:
        return {
            "training_compute": {
                "requested_device_type": self.backend,
                "effective_device_type": self.backend,
                "lightgbm_version": "4.fixture",
            },
            "lightgbm_version": "4.fixture",
            "evaluation_contract_version": self.contract_version,
            "accounting_schema_version": 2,
            "require_executable": require_executable,
            "holding_period_days": horizon,
            "cost_policy_hash": self.cost_policy_hash,
            "prospective_lockbox_start": prospective_lockbox_start,
            "execution_tail_policy": self.tail_policy,
        }

    def mature_information_end(self, signal_end: str, horizon: int) -> str:
        del horizon
        return signal_end

    def execute(
        self,
        *,
        fold: dict[str, object],
        horizon: int,
        features: tuple[str, ...],
        require_executable: bool,
        execution_contract: dict[str, object],
    ) -> FoldExecutionResult:
        del execution_contract
        fold_id = str(fold["fold_id"])
        self.calls.append(fold_id)
        if self.fail_once == fold_id:
            self.fail_once = None
            raise DataValidationError("synthetic interruption")
        index = int(fold_id[-1])
        predictions = pd.DataFrame(
            {
                "trade_date": [fold["evaluation_start"]],
                "ts_code": ["000001.SZ"],
                "prediction_score": [0.1],
            }
        )

        def save(path: Path) -> None:
            path.write_text(f"model-{fold_id}", encoding="utf-8")

        metrics = {
            "rank_ic": index / 100,
            "rank_ic_median": index / 100,
            "rank_ic_std": 0.0,
            "rank_icir": 0.0,
            "positive_rank_ic_ratio": 1.0,
            "ndcg_at_10": 0.8,
            "ndcg_at_50": 0.8,
            "coverage": 1.0,
            "signal_dates": 1,
            "securities_scored": 1,
        }
        executable = {
            "status": "COMPLETE",
            "accounting_schema_version": 2,
            "top_n": {
                str(top_n): {"total_return": index / 100, "sharpe": 0.5} for top_n in (10, 20, 50)
            },
            "accounting_summaries": {},
            "cost_policy_hash": "cost-fixture",
        }
        return FoldExecutionResult(
            predictions=predictions,
            validation_metrics={"rank_ic": 0.01},
            ranking_metrics=metrics,
            executable_metrics=executable if require_executable else {"status": "NOT_REQUIRED"},
            feature_importance=[
                {"feature": feature, "gain": float(len(features) - offset), "split": 1}
                for offset, feature in enumerate(features)
            ],
            training_compute={
                "requested_device_type": self.backend,
                "effective_device_type": self.backend,
                "lightgbm_version": "4.fixture",
            },
            model_saver=save,
        )


def _research_fixture(tmp_path: Path) -> tuple[Path, Path]:
    features = ("f1", "f2")
    reports_root = tmp_path / "reports"
    diagnostics_dir = reports_root / "feature_diagnostics" / "fixture"
    diagnostics_dir.mkdir(parents=True)
    diagnostics_manifest = diagnostics_dir / "manifest.json"
    recommendation = diagnostics_dir / "recommended_features.json"
    atomic_write_json(
        diagnostics_manifest,
        {"artifact_name": "feature_diagnostics", "horizon": 5},
    )
    atomic_write_json(recommendation, {"recommended_features": list(features)})
    provenance = FeatureSetProvenance(
        schema_version=2,
        artifact_name="feature_set_provenance",
        feature_set_name="fixture",
        feature_set_version="v1",
        provenance_status="GOVERNED",
        features=features,
        feature_list_hash=feature_list_hash(features),
        selection_policy="fixture_policy",
        selection_policy_version="1",
        selection_start="20100101",
        selection_end="20191231",
        source_diagnostics_run_id="fixture",
        source_diagnostics_manifest_locator="feature_diagnostics/fixture/manifest.json",
        source_diagnostics_manifest_hash=hashlib.sha256(
            diagnostics_manifest.read_bytes()
        ).hexdigest(),
        source_recommendation_locator="feature_diagnostics/fixture/recommended_features.json",
        source_recommendation_hash=hashlib.sha256(recommendation.read_bytes()).hexdigest(),
        source_feature_universe_hash="feature-universe-fixture",
        created_at="2026-08-09T00:00:00+00:00",
        created_by="pytest",
    )
    provenance_path = (
        reports_root / "feature_selection" / provenance.feature_set_id / "feature_set.json"
    )
    provenance_path.parent.mkdir(parents=True)
    atomic_write_json(provenance_path, provenance.model_dump(mode="json"))
    fold_dir = tmp_path / "walk_forward_plan"
    fold_dir.mkdir()
    fold_rows = []
    references = []
    for index, evaluation in enumerate(("20200102", "20210104", "20220104"), start=1):
        fold_id = f"fold_{index}"
        fold_rows.append(
            {
                "fold_id": fold_id,
                "train_start": "20100104",
                "train_end": "20181231",
                "validation_start": "20190108",
                "validation_end": "20191231",
                "evaluation_start": evaluation,
                "evaluation_end": evaluation,
                "purge_sessions": 6,
                "embargo_sessions": 6,
            }
        )
        references.append(
            {"fold_id": fold_id, "evaluation_start": evaluation, "evaluation_end": evaluation}
        )
    atomic_write_json(fold_dir / "folds.json", {"schema_version": 4, "folds": fold_rows})
    fold_manifest = fold_dir / "manifest.json"
    atomic_write_json(
        fold_manifest,
        {
            "schema_version": 4,
            "artifact_name": "purged_walk_forward_plan",
            "feature_authority": "governed_feature_set",
            "feature_set_id": provenance.feature_set_id,
            "feature_hash": provenance.feature_list_hash,
            "feature_set_hash": provenance.feature_list_hash,
            "feature_provenance_locator": str(provenance_path.relative_to(reports_root)),
            "feature_provenance_hash": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
            "outputs": {"folds": "folds.json"},
        },
    )
    plan = tmp_path / "experiment_manifest.json"
    atomic_write_json(
        plan,
        {
            "schema_version": 3,
            "artifact_name": "multi_horizon_experiment_plan",
            "plan_identity_hash": "plan-fixture",
            "feature_authority": "governed_feature_set",
            "feature_set_id": provenance.feature_set_id,
            "feature_hash": feature_list_hash(features),
            "feature_set_hash": feature_list_hash(features),
            "feature_provenance_hash": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
            "universe_hash": "universe-fixture",
            "labels_fingerprint": "labels-fixture",
            "folds_manifest": str(fold_manifest),
            "folds_manifest_hash": hashlib.sha256(fold_manifest.read_bytes()).hexdigest(),
            "folds_hash": hashlib.sha256((fold_dir / "folds.json").read_bytes()).hexdigest(),
            "experiments": [
                {
                    "experiment_id": "h5_fixture",
                    "horizon": 5,
                    "selection_period": {"folds": references[:2]},
                    "final_test_period": {"folds": references[2:]},
                }
            ],
        },
    )
    return plan, provenance_path
