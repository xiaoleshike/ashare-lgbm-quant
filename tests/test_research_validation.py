from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.feature_provenance import (
    FeatureSetProvenance,
    create_governed_feature_set,
    load_feature_set_provenance,
    validate_governed_feature_set,
)
from ashare_quant.models.research_policy import enforce_research_window, load_research_policy
from ashare_quant.models.temporal_isolation import (
    required_temporal_gap_sessions,
    resolve_temporal_gaps,
)
from ashare_quant.models.walk_forward_evaluation import (
    FoldExecutionResult,
    MultiFoldEvaluationRunner,
    WalkForwardRecoveryInspector,
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


def test_multi_fold_runner_rejects_corrupt_completed_fold(tmp_path: Path) -> None:
    plan, provenance = _research_fixture(tmp_path)
    executor = FakeExecutor(fail_once="fold_3")
    runner = MultiFoldEvaluationRunner(
        reports_root=tmp_path / "reports",
        settings=AppSettings.model_validate({}),
        executor=executor,
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
    )
    with pytest.raises(DataValidationError, match="RESEARCH_LOCKBOX_VIOLATION"):
        runner.run(
            experiment_manifest=plan,
            experiment_id="h5_fixture",
            feature_provenance_path=provenance,
        )


class FakeExecutor:
    def __init__(self, fail_once: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_once = fail_once

    def validate_sources(self, plan: dict[str, object]) -> dict[str, object]:
        return {
            "features_manifest_hash": plan.get("features_manifest_hash", "features-fixture"),
            "universe_manifest_hash": plan.get("universe_hash"),
            "labels_fingerprint": plan.get("labels_fingerprint"),
        }

    def execute(
        self,
        *,
        fold: dict[str, object],
        horizon: int,
        features: tuple[str, ...],
        require_executable: bool,
    ) -> FoldExecutionResult:
        fold_id = str(fold["fold_id"])
        self.calls.append(fold_id)
        if self.fail_once == fold_id:
            self.fail_once = None
            raise DataValidationError("synthetic interruption")
        index = int(fold_id[-1])
        predictions = pd.DataFrame(
            {"trade_date": [fold["evaluation_start"]], "ts_code": ["000001.SZ"], "score": [0.1]}
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
            training_compute={"requested_device_type": "cpu", "effective_device_type": "cpu"},
            model_saver=save,
        )


def _research_fixture(tmp_path: Path) -> tuple[Path, Path]:
    features = ("f1", "f2")
    reports_root = tmp_path / "reports"
    diagnostics_dir = reports_root / "feature_diagnostics" / "fixture"
    diagnostics_dir.mkdir(parents=True)
    diagnostics_manifest = diagnostics_dir / "manifest.json"
    recommendation = diagnostics_dir / "recommended_features.json"
    atomic_write_json(diagnostics_manifest, {"artifact_name": "feature_diagnostics"})
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
