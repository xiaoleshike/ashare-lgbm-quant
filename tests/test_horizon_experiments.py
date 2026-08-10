from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from ashare_quant.cli import main
from ashare_quant.config.settings import AppSettings
from ashare_quant.data.datasets import get_dataset_spec
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.storage import ParquetDataStore
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.feature_provenance import FeatureSetProvenance
from ashare_quant.models.horizon_experiments import MultiHorizonExperimentPlanner
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.utils.manifest import atomic_write_json


def test_horizon_configuration_parses_supported_isolated_targets() -> None:
    settings = AppSettings.model_validate({})

    assert [item.name for item in settings.models.horizon_experiments] == [
        "h5",
        "h10",
        "h20",
        "h60",
    ]
    assert [item.horizon for item in settings.models.horizon_experiments] == [5, 10, 20, 60]
    assert all(item.holding_days == item.horizon for item in settings.models.horizon_experiments)
    assert all(item.execution_rule == "next_open" for item in settings.models.horizon_experiments)


def test_horizon_configuration_rejects_mismatched_holding_period() -> None:
    with pytest.raises(ValidationError, match="holding_days must equal horizon"):
        AppSettings.model_validate(
            {
                "models": {
                    "horizon_experiments": [{"name": "invalid", "horizon": 5, "holding_days": 20}]
                }
            }
        )


def test_horizon_configuration_rejects_unconfigured_label_horizon() -> None:
    with pytest.raises(ValidationError, match=r"missing from labels.horizons: \[60\]"):
        AppSettings.model_validate({"labels": {"horizons": [5, 10, 20]}})


def test_horizon_plan_writes_correct_hashes_unique_ids_and_fold_references(
    tmp_path: Path,
) -> None:
    planner, sources = _planner_fixture(tmp_path)

    result = planner.build(folds_manifest=sources["fold_manifest"])
    manifest_path = result.output_dir / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiments = manifest["experiments"]

    assert result.experiment_count == 4
    assert manifest["feature_hash"] == feature_list_hash(("f1", "f3"))
    assert manifest["feature_hash"] != manifest["reference_champion_feature_hash"]
    assert manifest["feature_set_id"]
    assert manifest["feature_provenance_hash"] == _hash(sources["feature_provenance"])
    assert manifest["universe_hash"] == _hash(sources["universe_manifest"])
    assert manifest["folds_manifest_hash"] == _hash(sources["fold_manifest"])
    assert manifest["config_hash"] == _hash(sources["config"])
    assert len({record["experiment_id"] for record in experiments}) == 4
    assert {record["horizon"] for record in experiments} == {5, 10, 20, 60}
    assert all(record["holding_period"] == record["horizon"] for record in experiments)
    assert all(record["label_maturity_sessions"] == record["horizon"] + 1 for record in experiments)
    assert all(record["required_purge_sessions"] == record["horizon"] + 1 for record in experiments)
    assert all(
        record["required_embargo_sessions"] == record["horizon"] + 1 for record in experiments
    )
    assert all(
        record["folds_manifest"] == str(sources["fold_manifest"].resolve())
        for record in experiments
    )
    assert manifest["isolation_contract"] == {
        "champion_modified": False,
        "folds_regenerated": False,
        "future_return_values_loaded": False,
        "label_availability_metadata_checked": True,
        "label_values_loaded": False,
        "mixed_horizon_training": False,
        "model_trained": False,
        "test_results_used": False,
    }


def test_horizon_plan_is_idempotent_and_byte_deterministic(tmp_path: Path) -> None:
    planner, sources = _planner_fixture(tmp_path)

    first = planner.build(folds_manifest=sources["fold_manifest"])
    first_bytes = (first.output_dir / "experiment_manifest.json").read_bytes()
    second = planner.build(folds_manifest=sources["fold_manifest"])

    assert second.run_id == first.run_id
    assert second.output_dir == first.output_dir
    assert (second.output_dir / "experiment_manifest.json").read_bytes() == first_bytes


def test_horizon_plan_rejects_changed_feature_provenance(tmp_path: Path) -> None:
    planner, sources = _planner_fixture(tmp_path)
    provenance = sources["feature_provenance"]
    provenance.write_bytes(provenance.read_bytes() + b"\n")

    with pytest.raises(DataValidationError, match="FEATURE_PROVENANCE_MISMATCH"):
        planner.build(folds_manifest=sources["fold_manifest"])


def test_horizon_plan_reads_legacy_repository_relative_folds_path(tmp_path: Path) -> None:
    planner, sources = _planner_fixture(tmp_path)
    manifest = json.loads(sources["fold_manifest"].read_text(encoding="utf-8"))
    manifest["outputs"]["folds"] = str(sources["folds"].relative_to(tmp_path))
    atomic_write_json(sources["fold_manifest"], manifest)

    result = planner.build(folds_manifest=sources["fold_manifest"])

    assert result.experiment_count == 4


def test_horizon_plan_rejects_fold_boundaries_unsafe_for_longest_horizon(
    tmp_path: Path,
) -> None:
    planner, sources = _planner_fixture(tmp_path, gap=6)

    with pytest.raises(DataValidationError, match="required=61"):
        planner.build(folds_manifest=sources["fold_manifest"])


def test_horizon_plan_rejects_old_fixed_horizon_fold_manifest(tmp_path: Path) -> None:
    planner, sources = _planner_fixture(tmp_path)
    manifest = json.loads(sources["fold_manifest"].read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest["policy"]["label_horizon"] = 5
    atomic_write_json(sources["fold_manifest"], manifest)

    with pytest.raises(DataValidationError, match="schema-v4 governed plan required"):
        planner.build(folds_manifest=sources["fold_manifest"])


def test_horizon_plan_fails_when_a_required_physical_label_is_missing(tmp_path: Path) -> None:
    planner, sources = _planner_fixture(tmp_path, missing_horizons={60})

    with pytest.raises(DataValidationError, match="future_excess_ret_60d is unavailable"):
        planner.build(folds_manifest=sources["fold_manifest"])


def test_maturity_cutoff_and_selection_test_folds_are_isolated(tmp_path: Path) -> None:
    planner, sources = _planner_fixture(tmp_path)

    result = planner.build(folds_manifest=sources["fold_manifest"])
    manifest = json.loads(
        (result.output_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    h60 = next(record for record in manifest["experiments"] if record["horizon"] == 60)
    sessions = sources["sessions"].read_text(encoding="utf-8").splitlines()

    assert h60["maximum_mature_evaluation_date"] == sessions[-62]
    selection_fold_ids = {fold["fold_id"] for fold in h60["selection_period"]["folds"]}
    test_fold_ids = {fold["fold_id"] for fold in h60["final_test_period"]["folds"]}
    assert selection_fold_ids == {"fold_0001_202201"}
    assert test_fold_ids == {"fold_0002_202601"}
    assert selection_fold_ids.isdisjoint(test_fold_ids)
    assert h60["selection_period"]["may_select_model"] is True
    assert h60["final_test_period"]["may_select_model"] is False
    assert all(
        fold["evaluation_end"] <= h60["maximum_mature_evaluation_date"]
        for fold in h60["final_test_period"]["folds"]
    )


def test_horizon_plan_auto_discovers_existing_compatible_folds(tmp_path: Path) -> None:
    planner, sources = _planner_fixture(tmp_path)

    result = planner.build()

    assert result.folds_manifest == sources["fold_manifest"]


def test_horizon_plan_does_not_require_model_metrics_or_load_label_values(
    tmp_path: Path,
) -> None:
    planner, sources = _planner_fixture(tmp_path)
    sources["metrics"].unlink()

    result = planner.build(folds_manifest=sources["fold_manifest"])

    manifest = json.loads(
        (result.output_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["isolation_contract"]["label_values_loaded"] is False
    assert manifest["isolation_contract"]["test_results_used"] is False


def test_horizon_plan_cli_generates_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, sources = _planner_fixture(tmp_path)

    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "models",
            "--storage-root",
            str(tmp_path / "raw"),
            "--processed-root",
            str(tmp_path / "processed"),
            "--output-root",
            str(tmp_path / "models"),
            "--reports-root",
            str(tmp_path / "reports"),
            "horizon-plan",
            "--folds-manifest",
            str(sources["fold_manifest"]),
        ]
    )

    assert exit_code == 0
    assert "horizon_experiment_plan:" in capsys.readouterr().out


def _planner_fixture(
    tmp_path: Path,
    *,
    gap: int = 61,
    missing_horizons: set[int] | None = None,
) -> tuple[MultiHorizonExperimentPlanner, dict[str, Path]]:
    raw_root = tmp_path / "raw"
    models_root = tmp_path / "models"
    processed_root = tmp_path / "processed"
    reports_root = tmp_path / "reports"
    artifact = _write_model_artifact(models_root)
    registry = ModelRegistry(models_root)
    registry.register_model(artifact)
    registry.promote_model("horizon_fixture")

    sessions = tuple(date.strftime("%Y%m%d") for date in pd.bdate_range("2010-01-04", "2026-07-31"))
    ParquetDataStore(raw_root).write(
        get_dataset_spec("trade_cal"),
        pd.DataFrame(
            {
                "exchange": "SSE",
                "cal_date": sessions,
                "is_open": 1,
                "pretrade_date": (None, *sessions[:-1]),
            }
        ),
    )
    session_file = tmp_path / "sessions.txt"
    session_file.write_text("\n".join(sessions), encoding="utf-8")

    missing = missing_horizons or set()
    label_rows = [
        {
            "trade_date": trade_date,
            "horizon": horizon,
            "is_label_available": True,
            "future_excess_ret": 0.01,
        }
        for horizon in (5, 10, 20, 60)
        if horizon not in missing
        for trade_date in ("20220110", "20260112")
    ]
    label_path = processed_root / "labels_forward" / "year=fixture" / "data.parquet"
    label_path.parent.mkdir(parents=True)
    pd.DataFrame(label_rows).to_parquet(label_path, index=False)

    universe_manifest = processed_root / "universe_daily" / "_manifest.json"
    atomic_write_json(
        universe_manifest,
        {
            "artifact_name": "universe_daily",
            "git_commit": "fixture",
            "config_hash": "fixture-config",
            "canonical_artifact": {"min_date": "20100104", "max_date": "20260717"},
        },
    )
    atomic_write_json(
        processed_root / "features_daily" / "_manifest.json",
        {"artifact_name": "features_daily", "feature_hash": feature_list_hash(("f1", "f2"))},
    )
    research_features = ("f1", "f3")
    feature_provenance = _write_governed_provenance(reports_root, research_features)
    feature_set = json.loads(feature_provenance.read_text(encoding="utf-8"))
    fold_dir = reports_root / "walk_forward" / "walk_forward_fixture"
    folds_file = fold_dir / "folds.json"
    atomic_write_json(
        folds_file,
        {
            "schema_version": 4,
            "run_id": "walk_forward_fixture",
            "folds": [
                {
                    "fold_id": "fold_0001_202201",
                    "train_start": "20100104",
                    "train_end": "20181231",
                    "validation_start": "20200102",
                    "validation_end": "20210930",
                    "evaluation_start": "20220104",
                    "evaluation_end": "20220131",
                    "purge_sessions": gap,
                    "embargo_sessions": gap,
                    "feature_hash": feature_list_hash(research_features),
                },
                {
                    "fold_id": "fold_0002_202601",
                    "train_start": "20100104",
                    "train_end": "20231229",
                    "validation_start": "20240401",
                    "validation_end": "20250930",
                    "evaluation_start": "20260105",
                    "evaluation_end": "20260130",
                    "purge_sessions": gap,
                    "embargo_sessions": gap,
                    "feature_hash": feature_list_hash(research_features),
                },
                {
                    "fold_id": "fold_0003_202607",
                    "train_start": "20100104",
                    "train_end": "20240131",
                    "validation_start": "20240501",
                    "validation_end": "20251031",
                    "evaluation_start": "20260701",
                    "evaluation_end": "20260731",
                    "purge_sessions": gap,
                    "embargo_sessions": gap,
                    "feature_hash": feature_list_hash(research_features),
                },
            ],
        },
    )
    fold_manifest = fold_dir / "manifest.json"
    atomic_write_json(
        fold_manifest,
        {
            "schema_version": 4,
            "artifact_name": "purged_walk_forward_plan",
            "run_id": "walk_forward_fixture",
            "feature_authority": "governed_feature_set",
            "feature_set_id": feature_set["feature_set_id"]
            if "feature_set_id" in feature_set
            else FeatureSetProvenance.model_validate(feature_set).feature_set_id,
            "feature_hash": feature_list_hash(research_features),
            "feature_set_hash": feature_list_hash(research_features),
            "feature_provenance_locator": str(feature_provenance.relative_to(reports_root)),
            "feature_provenance_hash": _hash(feature_provenance),
            "policy": {"purge_sessions": gap, "embargo_sessions": gap},
            "outputs": {"folds": str(folds_file.resolve())},
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project_name: horizon-fixture\n", encoding="utf-8")
    settings = AppSettings.model_validate({})
    return (
        MultiHorizonExperimentPlanner(
            raw_root=raw_root,
            models_root=models_root,
            processed_root=processed_root,
            reports_root=reports_root,
            settings=settings,
            config_path=config_path,
        ),
        {
            "config": config_path,
            "fold_manifest": fold_manifest,
            "folds": folds_file,
            "metrics": artifact / "metrics.json",
            "sessions": session_file,
            "universe_manifest": universe_manifest,
            "feature_provenance": feature_provenance,
        },
    )


def _write_governed_provenance(reports_root: Path, features: tuple[str, ...]) -> Path:
    diagnostics = reports_root / "feature_diagnostics" / "fixture"
    diagnostics.mkdir(parents=True)
    diagnostics_manifest = diagnostics / "manifest.json"
    recommendation = diagnostics / "recommended_features.json"
    atomic_write_json(diagnostics_manifest, {"artifact_name": "feature_diagnostics"})
    atomic_write_json(recommendation, {"recommended_features": list(features)})
    provenance = FeatureSetProvenance(
        schema_version=2,
        artifact_name="feature_set_provenance",
        feature_set_name="fixture",
        feature_set_version="v2",
        provenance_status="GOVERNED",
        features=features,
        feature_list_hash=feature_list_hash(features),
        selection_policy="fixture",
        selection_policy_version="1",
        selection_start="20100101",
        selection_end="20191231",
        source_diagnostics_run_id="fixture",
        source_diagnostics_manifest_locator="feature_diagnostics/fixture/manifest.json",
        source_diagnostics_manifest_hash=_hash(diagnostics_manifest),
        source_recommendation_locator="feature_diagnostics/fixture/recommended_features.json",
        source_recommendation_hash=_hash(recommendation),
        source_feature_universe_hash="fixture-universe",
        created_at="2026-08-09T00:00:00+00:00",
        created_by="pytest",
    )
    path = reports_root / "feature_selection" / provenance.feature_set_id / "feature_set.json"
    path.parent.mkdir(parents=True)
    atomic_write_json(path, provenance.model_dump(mode="json"))
    return path


def _write_model_artifact(models_root: Path) -> Path:
    model_id = "horizon_fixture"
    artifact = models_root / model_id
    artifact.mkdir(parents=True)
    features = ("f1", "f2")
    digest = feature_list_hash(features)
    (artifact / "model.txt").write_text("tree\n", encoding="utf-8")
    atomic_write_json(
        artifact / "feature_list.json",
        {"features": list(features), "feature_hash": digest},
    )
    atomic_write_json(
        artifact / "metrics.json",
        {"validation": {"rank_ic": 0.02}, "test": {"rank_ic": 0.01}},
    )
    atomic_write_json(
        artifact / "manifest.json",
        {
            "artifact_name": "lightgbm_ranker_baseline",
            "experiment_id": model_id,
            "completed_at": "2026-07-20T00:00:00+00:00",
            "feature_list_hash": digest,
            "train_start": "20100101",
            "train_end": "20191231",
        },
    )
    return artifact


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
