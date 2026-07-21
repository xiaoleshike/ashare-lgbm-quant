from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.cli import main
from ashare_quant.config.settings import AppSettings
from ashare_quant.data.datasets import get_dataset_spec
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.storage import ParquetDataStore
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.models.walk_forward import (
    PurgedWalkForwardPlanner,
    WalkForwardPlanResult,
)


def test_expanding_plan_has_strict_chronology_and_leakage_boundaries(
    tmp_path: Path,
) -> None:
    planner, sessions = _planner_fixture(tmp_path)

    result = planner.build(
        start_date="20200104",
        end_date="20200630",
        scheme="expanding",
    )
    folds = _read_folds(result.output_dir)
    positions = {date: index for index, date in enumerate(sessions)}

    assert len(folds) >= 2
    assert {fold["train_start"] for fold in folds} == {sessions[0]}
    assert [fold["train_end"] for fold in folds] == sorted(fold["train_end"] for fold in folds)
    for fold in folds:
        assert fold["train_end"] < fold["validation_start"]
        assert fold["validation_end"] < fold["evaluation_start"]
        assert positions[fold["validation_start"]] - positions[fold["train_end"]] - 1 == 3
        assert positions[fold["evaluation_start"]] - positions[fold["validation_end"]] - 1 == 3
        assert positions[fold["train_end"]] + 3 < positions[fold["validation_start"]]
        assert positions[fold["validation_end"]] + 3 < positions[fold["evaluation_start"]]


def test_rolling_plan_uses_fixed_trading_session_window(tmp_path: Path) -> None:
    planner, _ = _planner_fixture(tmp_path)

    result = planner.build(
        start_date="20200104",
        end_date="20200630",
        scheme="rolling",
    )
    folds = _read_folds(result.output_dir)

    assert len(folds) >= 2
    assert {fold["train_sessions"] for fold in folds} == {20}
    assert len({fold["train_start"] for fold in folds}) == len(folds)


def test_non_trading_request_boundaries_map_only_to_open_sessions(tmp_path: Path) -> None:
    planner, sessions = _planner_fixture(tmp_path)

    result = planner.build(
        start_date="20200104",  # Saturday
        end_date="20200628",  # Sunday
        scheme="expanding",
    )

    for fold in _read_folds(result.output_dir):
        for field in (
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "evaluation_start",
            "evaluation_end",
        ):
            assert fold[field] in sessions
            assert "20200104" <= fold[field] <= "20200628"


def test_walk_forward_gaps_are_horizon_agnostic_boundaries(tmp_path: Path) -> None:
    planner, _ = _planner_fixture(tmp_path)

    result = planner.build(
        start_date="20200104",
        end_date="20200630",
        scheme="expanding",
        purge_days=2,
        embargo_days=2,
    )

    folds = _read_folds(result.output_dir)
    assert all(fold["purge_sessions"] == 2 for fold in folds)
    assert all(fold["embargo_sessions"] == 2 for fold in folds)
    assert all("label_horizon" not in fold for fold in folds)
    assert all("label_exit_lag_sessions" not in fold for fold in folds)


def test_plan_manifest_records_fold_model_and_calendar_provenance(tmp_path: Path) -> None:
    planner, sessions = _planner_fixture(tmp_path)

    result = planner.build(
        start_date="20200104",
        end_date="20200630",
        scheme="expanding",
    )
    manifest = json.loads((result.output_dir / "manifest.json").read_text(encoding="utf-8"))
    folds = _read_folds(result.output_dir)

    assert {path.name for path in result.output_dir.iterdir()} == {"folds.json", "manifest.json"}
    assert manifest["artifact_name"] == "purged_walk_forward_plan"
    assert manifest["model_id"] == "walk_forward_fixture"
    assert manifest["feature_hash"] == feature_list_hash(("f1", "f2"))
    assert manifest["fold_count"] == len(folds)
    assert manifest["trade_calendar"]["session_count"] == len(sessions)
    assert manifest["leakage_contract"]["labels_loaded"] is False
    assert manifest["leakage_contract"]["model_fitted"] is False
    assert manifest["leakage_contract"]["fold_boundaries_are_horizon_agnostic"] is True
    required = {
        "fold_id",
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "evaluation_start",
        "evaluation_end",
        "purge_sessions",
        "embargo_sessions",
        "feature_hash",
        "model_id",
    }
    assert all(required <= set(fold) for fold in folds)


def test_plan_needs_no_labels_features_or_model_training(tmp_path: Path) -> None:
    planner, _ = _planner_fixture(tmp_path)
    assert not (tmp_path / "processed").exists()

    result = planner.build(
        start_date="20200104",
        end_date="20200630",
        scheme="expanding",
    )

    assert result.fold_count > 0
    assert not (tmp_path / "processed").exists()


def test_models_walk_forward_cli_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "reports" / "walk_forward" / "run"

    def succeed(self: object, **kwargs: object) -> WalkForwardPlanResult:
        return WalkForwardPlanResult(
            run_id="run",
            output_dir=output,
            fold_count=4,
            scheme="expanding",
            model_id="fixture",
        )

    monkeypatch.setattr(PurgedWalkForwardPlanner, "build", succeed)
    command = [
        "--config",
        "config/default.yaml",
        "models",
        "walk-forward-plan",
        "--start-date",
        "20100101",
        "--end-date",
        "20260717",
        "--scheme",
        "expanding",
    ]
    assert main(command) == 0
    assert "walk_forward_plan: run_id=run" in capsys.readouterr().out

    def fail(self: object, **kwargs: object) -> WalkForwardPlanResult:
        raise DataValidationError("calendar unavailable")

    monkeypatch.setattr(PurgedWalkForwardPlanner, "build", fail)
    assert main(command) == 2
    assert "calendar unavailable" in capsys.readouterr().err


def _planner_fixture(tmp_path: Path) -> tuple[PurgedWalkForwardPlanner, tuple[str, ...]]:
    raw_root = tmp_path / "raw"
    models_root = tmp_path / "models"
    reports_root = tmp_path / "reports"
    sessions = tuple(date.strftime("%Y%m%d") for date in pd.bdate_range("2020-01-06", "2020-06-26"))
    calendar = pd.DataFrame(
        {
            "exchange": "SSE",
            "cal_date": sessions,
            "is_open": 1,
            "pretrade_date": (None, *sessions[:-1]),
        }
    )
    ParquetDataStore(raw_root).write(get_dataset_spec("trade_cal"), calendar)
    artifact = _write_model_artifact(models_root)
    registry = ModelRegistry(models_root)
    registry.register_model(artifact)
    registry.promote_model("walk_forward_fixture")
    settings = AppSettings.model_validate(
        {
            "ranker": {
                "label_horizon": 2,
                "walk_forward": {
                    "annual_sessions": 10,
                    "minimum_training_years": 2,
                    "rolling_window_years": 2,
                    "validation_sessions": 10,
                    "purge_days": 3,
                    "embargo_days": 3,
                },
            }
        }
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("ranker: fixture\n", encoding="utf-8")
    return (
        PurgedWalkForwardPlanner(
            raw_root=raw_root,
            models_root=models_root,
            reports_root=reports_root,
            settings=settings,
            config_path=config_path,
        ),
        sessions,
    )


def _write_model_artifact(models_root: Path) -> Path:
    model_id = "walk_forward_fixture"
    artifact = models_root / model_id
    artifact.mkdir(parents=True)
    features = ("f1", "f2")
    digest = feature_list_hash(features)
    (artifact / "model.txt").write_text("tree\n", encoding="utf-8")
    (artifact / "feature_list.json").write_text(
        json.dumps({"features": list(features), "feature_hash": digest}), encoding="utf-8"
    )
    (artifact / "metrics.json").write_text(
        json.dumps({"validation": {"rank_ic": 0.02}, "test": {"rank_ic": 0.01}}),
        encoding="utf-8",
    )
    (artifact / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_name": "lightgbm_ranker_baseline",
                "experiment_id": model_id,
                "completed_at": "2026-07-20T00:00:00+00:00",
                "feature_list_hash": digest,
                "train_start": "20100101",
                "train_end": "20191231",
            }
        ),
        encoding="utf-8",
    )
    return artifact


def _read_folds(output_dir: Path) -> list[dict[str, object]]:
    payload = json.loads((output_dir / "folds.json").read_text(encoding="utf-8"))
    return payload["folds"]
