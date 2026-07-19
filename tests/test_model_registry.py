from __future__ import annotations

import json
from pathlib import Path

import pytest

from ashare_quant.cli import main
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.orchestration.lock import production_lock


def _write_artifact(
    models_root: Path,
    model_id: str,
    *,
    features: tuple[str, ...] = ("ret_1d", "volatility_20d"),
    include_model: bool = True,
    include_test_metrics: bool = True,
) -> Path:
    artifact = models_root / model_id
    artifact.mkdir(parents=True)
    digest = feature_list_hash(features)
    if include_model:
        (artifact / "model.txt").write_text("tree\n", encoding="utf-8")
    (artifact / "feature_list.json").write_text(
        json.dumps({"features": list(features), "feature_hash": digest}), encoding="utf-8"
    )
    metrics: dict[str, object] = {
        "validation": {"rank_ic": 0.031},
        "feature_importance": [],
    }
    if include_test_metrics:
        metrics["test"] = {"rank_ic": 0.021, "sharpe": 1.25}
    (artifact / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (artifact / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_name": "lightgbm_ranker_baseline",
                "experiment_id": model_id,
                "completed_at": "2026-07-19T12:00:00+00:00",
                "git_commit": "abc123",
                "config_hash": "config123",
                "feature_list_hash": digest,
                "train_start": "20100101",
                "train_end": "20191231",
            }
        ),
        encoding="utf-8",
    )
    return artifact


def test_register_model_persists_candidate_and_history(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    artifact = _write_artifact(models_root, "experiment_a")

    registered = ModelRegistry(models_root).register_model(artifact)
    reloaded = ModelRegistry(models_root).list_models()

    assert registered.status == "candidate"
    assert registered.training_date_range == {"start": "20100101", "end": "20191231"}
    assert registered.validation_metrics["rank_ic"] == 0.031
    assert registered.test_metrics["rank_ic"] == 0.021
    assert reloaded == (registered,)
    assert len(list((models_root / "registry_history").glob("register_*.json"))) == 1


def test_duplicate_registration_is_rejected_without_overwrite(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    artifact = _write_artifact(models_root, "experiment_a")
    registry = ModelRegistry(models_root)
    registry.register_model(artifact)
    before = (models_root / "registry.json").read_bytes()

    with pytest.raises(DataValidationError, match="already registered"):
        registry.register_model(artifact)

    assert (models_root / "registry.json").read_bytes() == before


def test_promotion_keeps_one_champion_per_model_type(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    registry = ModelRegistry(models_root)
    registry.register_model(_write_artifact(models_root, "experiment_a"))
    registry.register_model(_write_artifact(models_root, "experiment_b"))

    registry.promote_model("experiment_a")
    promoted = registry.promote_model("experiment_b")
    records = {record.model_id: record for record in registry.list_models()}

    assert promoted.status == "champion"
    assert registry.get_champion("lightgbm_ranker") == promoted
    assert records["experiment_a"].status == "candidate"
    assert records["experiment_b"].status == "champion"


@pytest.mark.parametrize(
    ("include_model", "include_test_metrics", "message"),
    [
        (False, True, "required artifact file is missing: model.txt"),
        (True, False, "does not contain non-empty test metrics"),
    ],
)
def test_promotion_explains_artifact_validation_failure(
    tmp_path: Path,
    include_model: bool,
    include_test_metrics: bool,
    message: str,
) -> None:
    models_root = tmp_path / "models"
    artifact = _write_artifact(
        models_root,
        "experiment_a",
        include_model=include_model,
        include_test_metrics=include_test_metrics,
    )
    registry = ModelRegistry(models_root)
    registry.register_model(artifact)

    with pytest.raises(DataValidationError, match=message):
        registry.promote_model("experiment_a")

    assert registry.list_models()[0].status == "candidate"


def test_promotion_rejects_feature_hash_changed_after_registration(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    artifact = _write_artifact(models_root, "experiment_a")
    registry = ModelRegistry(models_root)
    registry.register_model(artifact)
    (artifact / "feature_list.json").write_text(
        json.dumps({"features": ["ret_1d"], "feature_hash": feature_list_hash(("ret_1d",))}),
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="feature hash differs from registered value"):
        registry.promote_model("experiment_a")


def test_retired_models_remain_queryable_and_champion_is_cleared(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    registry = ModelRegistry(models_root)
    registry.register_model(_write_artifact(models_root, "experiment_a"))
    registry.promote_model("experiment_a")

    retired = registry.retire_model("experiment_a")

    assert retired.status == "retired"
    assert registry.get_champion("lightgbm_ranker") is None
    assert registry.list_models() == (retired,)
    history = list((models_root / "registry_history").glob("retire_*.json"))
    assert len(history) == 1
    payload = json.loads(history[0].read_text(encoding="utf-8"))
    assert payload["old_champion"] == "experiment_a"
    assert payload["new_champion"] is None


def test_models_lifecycle_cli_output_and_exit_codes(tmp_path: Path, capsys) -> None:
    models_root = tmp_path / "models"
    registry = ModelRegistry(models_root)
    registry.register_model(_write_artifact(models_root, "experiment_a"))
    common = [
        "--config",
        "config/default.yaml",
        "models",
        "--output-root",
        str(models_root),
    ]

    assert main([*common, "list"]) == 0
    listed = capsys.readouterr().out
    assert "model_id\tstatus\tcreated_time\ttest_rank_ic\ttest_sharpe\tfeature_count" in listed
    assert "experiment_a\tcandidate" in listed
    assert "0.021000\t1.250000\t2" in listed

    assert main([*common, "champion"]) == 0
    assert "model_champion: none" in capsys.readouterr().out

    assert main([*common, "promote", "experiment_a"]) == 0
    assert "model_promoted: model_id=experiment_a" in capsys.readouterr().out
    assert main([*common, "champion"]) == 0
    assert "model_champion: model_id=experiment_a" in capsys.readouterr().out

    assert main([*common, "retire", "experiment_a"]) == 0
    assert "model_retired: model_id=experiment_a status=retired" in capsys.readouterr().out
    assert main([*common, "promote", "missing"]) == 2
    assert "model_id is not registered: missing" in capsys.readouterr().err


def test_models_cli_reports_concurrent_registry_operation(tmp_path: Path, capsys) -> None:
    models_root = tmp_path / "models"
    registry = ModelRegistry(models_root)
    registry.register_model(_write_artifact(models_root, "experiment_a"))

    with production_lock(models_root / ".registry.lock", command="test owner"):
        exit_code = main(
            [
                "--config",
                "config/default.yaml",
                "models",
                "--output-root",
                str(models_root),
                "promote",
                "experiment_a",
            ]
        )

    assert exit_code == 2
    assert "another production run is active" in capsys.readouterr().err
