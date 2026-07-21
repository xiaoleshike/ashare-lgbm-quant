from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ashare_quant.cli import main
from ashare_quant.config.settings import AppSettings, RankerSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.challenger import ChallengerTrainer, ChallengerTrainingResult
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.ranker_data import RankerDataLoader, RankerDataset
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.utils.manifest import atomic_write_json


class _FakeBooster:
    def save_model(self, path: str) -> None:
        Path(path).write_text("deterministic challenger model\n", encoding="utf-8")

    def feature_importance(self, importance_type: str) -> np.ndarray:
        if importance_type == "gain":
            return np.asarray([2.0, 1.0])
        return np.asarray([2, 1])


class _FakeRanker:
    booster_ = _FakeBooster()

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features["f1"].to_numpy(dtype=float)


def test_challenger_reads_horizon_manifest_and_registers_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer, sources = _trainer_fixture(tmp_path)
    loaded_ranges: list[tuple[str, str, int]] = []

    def fake_load(
        self: RankerDataLoader,
        start_date: str,
        end_date: str,
        feature_names: tuple[str, ...],
        relevance_grades: int,
    ) -> RankerDataset:
        del relevance_grades
        loaded_ranges.append((start_date, end_date, self.horizon))
        return _ranker_dataset(feature_names, start_date)

    monkeypatch.setattr("ashare_quant.models.challenger.RankerDataLoader.load", fake_load)
    monkeypatch.setattr("ashare_quant.models.challenger.fit_ranker", lambda *_args: _FakeRanker())

    result = trainer.train(
        experiment_id="experiment_c_h5",
        experiment_manifest=sources["horizon_manifest"],
    )[0]

    assert loaded_ranges == [
        ("20100104", "20191231", 5),
        ("20200102", "20211231", 5),
    ]
    assert set(path.name for path in result.output_dir.iterdir()) == {
        "model.txt",
        "feature_list.json",
        "metrics.json",
        "manifest.json",
    }
    registered = ModelRegistry(sources["models_root"]).list_models()
    challenger = next(model for model in registered if model.model_id == result.model_id)
    champion = ModelRegistry(sources["models_root"]).get_champion("lightgbm_ranker")
    assert challenger.status == "candidate"
    assert champion is not None and champion.model_id == "source_champion"
    manifest = json.loads((result.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["horizon"] == 5
    assert manifest["holding_period"] == 5
    assert manifest["label_name"] == "future_excess_ret_5d"
    assert manifest["train_dates"] == {"start": "20100104", "end": "20191231"}
    assert manifest["validation_dates"] == {"start": "20200102", "end": "20211231"}
    assert manifest["training_rows"] == 20
    assert manifest["validation_rows"] == 20
    assert manifest["isolation_contract"]["final_test_labels_loaded"] is False
    metrics = json.loads((result.output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test"] == {}


def test_challenger_rejects_label_horizon_mismatch(tmp_path: Path) -> None:
    trainer, sources = _trainer_fixture(tmp_path)
    payload = _read_json(sources["horizon_manifest"])
    payload["experiments"][0]["label_name"] = "future_excess_ret_10d"
    atomic_write_json(sources["horizon_manifest"], payload)

    with pytest.raises(DataValidationError, match="label_name does not match horizon"):
        trainer.train(
            experiment_id="experiment_c_h5",
            experiment_manifest=sources["horizon_manifest"],
        )


def test_challenger_rejects_pre_v2_fold_schema(tmp_path: Path) -> None:
    trainer, sources = _trainer_fixture(tmp_path)
    fold_manifest = _read_json(sources["fold_manifest"])
    fold_manifest["schema_version"] = 1
    atomic_write_json(sources["fold_manifest"], fold_manifest)
    horizon = _read_json(sources["horizon_manifest"])
    horizon["folds_manifest_hash"] = _hash(sources["fold_manifest"])
    horizon["experiments"][0]["folds_manifest_hash"] = _hash(sources["fold_manifest"])
    atomic_write_json(sources["horizon_manifest"], horizon)

    with pytest.raises(DataValidationError, match="schema version 2"):
        trainer.train(
            experiment_id="experiment_c_h5",
            experiment_manifest=sources["horizon_manifest"],
        )


def test_challenger_rejects_selection_fold_overlapping_final_test(tmp_path: Path) -> None:
    trainer, sources = _trainer_fixture(tmp_path)
    folds = _read_json(sources["folds"])
    folds["folds"][0]["validation_end"] = "20230102"
    folds["folds"][0]["evaluation_start"] = "20230403"
    folds["folds"][0]["evaluation_end"] = "20231229"
    atomic_write_json(sources["folds"], folds)
    _refresh_fold_hashes(sources)

    with pytest.raises(DataValidationError, match="final-test period"):
        trainer.train(
            experiment_id="experiment_c_h5",
            experiment_manifest=sources["horizon_manifest"],
        )


def test_challenger_artifact_is_immutable_and_manifest_is_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer, sources = _trainer_fixture(tmp_path)
    monkeypatch.setattr(
        "ashare_quant.models.challenger.RankerDataLoader.load",
        lambda self, start_date, end_date, feature_names, relevance_grades: _ranker_dataset(
            feature_names, start_date
        ),
    )
    monkeypatch.setattr("ashare_quant.models.challenger.fit_ranker", lambda *_args: _FakeRanker())
    result = trainer.train(
        experiment_id="experiment_c_h5",
        experiment_manifest=sources["horizon_manifest"],
    )[0]
    before = (result.output_dir / "manifest.json").read_bytes()
    persisted = json.loads(before)
    plan = _read_json(sources["horizon_manifest"])
    folds = _read_json(sources["folds"])["folds"]
    source_model = next(
        model
        for model in ModelRegistry(sources["models_root"]).list_models()
        if model.model_id == "source_champion"
    )
    rebuilt = trainer._manifest(
        model_id=result.model_id,
        plan_path=sources["horizon_manifest"],
        plan=plan,
        experiment=plan["experiments"][0],
        fold=folds[0],
        source_model=source_model,
        features=("f1", "f2"),
        training_rows=20,
        validation_rows=20,
    )

    with pytest.raises(DataValidationError, match="immutable challenger artifact already exists"):
        trainer.train(
            experiment_id="experiment_c_h5",
            experiment_manifest=sources["horizon_manifest"],
        )

    assert (result.output_dir / "manifest.json").read_bytes() == before
    assert rebuilt == persisted


def test_train_challenger_cli_routes_single_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "models" / "challengers" / "experiment_c_h5_fixture"

    def fake_train(self: object, **kwargs: Any) -> tuple[ChallengerTrainingResult, ...]:
        assert kwargs["experiment_id"] == "experiment_c_h5"
        assert kwargs["all_horizons"] is False
        return (
            ChallengerTrainingResult(
                model_id="experiment_c_h5_fixture",
                experiment_id="experiment_c_h5_fixture",
                horizon=5,
                output_dir=output,
                training_rows=20,
                validation_rows=20,
                validation_rank_ic=0.1,
            ),
        )

    monkeypatch.setattr(ChallengerTrainer, "train", fake_train)
    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "models",
            "--processed-root",
            str(tmp_path / "processed"),
            "--output-root",
            str(tmp_path / "models"),
            "--reports-root",
            str(tmp_path / "reports"),
            "train-challenger",
            "--experiment-id",
            "experiment_c_h5",
        ]
    )

    assert exit_code == 0
    assert "challenger_trained: model_id=experiment_c_h5_fixture" in capsys.readouterr().out


def _trainer_fixture(tmp_path: Path) -> tuple[ChallengerTrainer, dict[str, Path]]:
    models_root = tmp_path / "models"
    processed_root = tmp_path / "processed"
    reports_root = tmp_path / "reports"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project_name: challenger-fixture\n", encoding="utf-8")
    features = ("f1", "f2")
    feature_hash = feature_list_hash(features)
    source_artifact = models_root / "source_champion"
    source_artifact.mkdir(parents=True)
    (source_artifact / "model.txt").write_text("source model\n", encoding="utf-8")
    atomic_write_json(
        source_artifact / "feature_list.json",
        {"features": list(features), "feature_hash": feature_hash},
    )
    atomic_write_json(
        source_artifact / "metrics.json",
        {"validation": {"rank_ic": 0.02}, "test": {"rank_ic": 0.01}},
    )
    atomic_write_json(
        source_artifact / "manifest.json",
        {
            "artifact_name": "lightgbm_ranker_baseline",
            "experiment_id": "source_champion",
            "completed_at": "2026-07-20T00:00:00+00:00",
            "git_commit": "fixture",
            "config_hash": "fixture",
            "feature_list_hash": feature_hash,
            "train_start": "20100104",
            "train_end": "20191231",
        },
    )
    registry = ModelRegistry(models_root)
    registry.register_model(source_artifact)
    registry.promote_model("source_champion")

    features_manifest = processed_root / "features_daily" / "_manifest.json"
    universe_manifest = processed_root / "universe_daily" / "_manifest.json"
    atomic_write_json(features_manifest, {"artifact_name": "features_daily"})
    atomic_write_json(universe_manifest, {"artifact_name": "universe_daily"})
    pd.DataFrame({"trade_date": ["20200102"], "ts_code": ["000001.SZ"]}).to_parquet(
        processed_root / "features_daily" / "data.parquet", index=False
    )
    labels_dir = processed_root / "labels_forward"
    labels_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "trade_date": ["20200102"],
            "ts_code": ["000001.SZ"],
            "horizon": [5],
            "future_excess_ret": [0.01],
            "is_label_available": [True],
        }
    ).to_parquet(labels_dir / "data.parquet", index=False)
    pd.DataFrame(
        {
            "trade_date": ["20200102"],
            "ts_code": ["000001.SZ"],
            "in_model_universe": [True],
        }
    ).to_parquet(processed_root / "universe_daily" / "data.parquet", index=False)

    fold_dir = reports_root / "walk_forward" / "fixture"
    folds = fold_dir / "folds.json"
    atomic_write_json(
        folds,
        {
            "schema_version": 2,
            "folds": [
                {
                    "fold_id": "fold_selection",
                    "train_start": "20100104",
                    "train_end": "20191231",
                    "validation_start": "20200102",
                    "validation_end": "20211231",
                    "evaluation_start": "20220104",
                    "evaluation_end": "20221230",
                    "purge_sessions": 61,
                    "embargo_sessions": 61,
                },
                {
                    "fold_id": "fold_test",
                    "train_start": "20100104",
                    "train_end": "20211231",
                    "validation_start": "20220104",
                    "validation_end": "20221230",
                    "evaluation_start": "20230103",
                    "evaluation_end": "20231229",
                    "purge_sessions": 61,
                    "embargo_sessions": 61,
                },
            ],
        },
    )
    fold_manifest = fold_dir / "manifest.json"
    atomic_write_json(
        fold_manifest,
        {
            "schema_version": 2,
            "artifact_name": "purged_walk_forward_plan",
            "feature_hash": feature_hash,
            "policy": {"purge_sessions": 61, "embargo_sessions": 61},
            "outputs": {"folds": str(folds.resolve())},
        },
    )
    horizon_dir = reports_root / "horizon_experiments" / "fixture"
    horizon_manifest = horizon_dir / "experiment_manifest.json"
    atomic_write_json(
        horizon_manifest,
        {
            "schema_version": 2,
            "artifact_name": "multi_horizon_experiment_plan",
            "plan_identity_hash": "a" * 64,
            "created_time": "2026-07-20T00:00:00+00:00",
            "source_model_id": "source_champion",
            "feature_hash": feature_hash,
            "universe_hash": _hash(universe_manifest),
            "config_hash": _hash(config_path),
            "folds_manifest": str(fold_manifest.resolve()),
            "folds_manifest_hash": _hash(fold_manifest),
            "folds_hash": _hash(folds),
            "selection_period": {
                "start_date": "20150101",
                "end_date": "20221231",
                "purpose": "challenger_comparison_only",
            },
            "final_test_period": {
                "start_date": "20230101",
                "end_date": "20260710",
                "purpose": "one_time_final_evaluation_only",
                "may_select_model": False,
            },
            "experiments": [
                {
                    "experiment_id": "h5_fixture",
                    "name": "h5",
                    "horizon": 5,
                    "holding_period": 5,
                    "execution_rule": "next_open",
                    "label_name": "future_excess_ret_5d",
                    "label_maturity_sessions": 6,
                    "required_purge_sessions": 6,
                    "required_embargo_sessions": 6,
                    "feature_hash": feature_hash,
                    "universe_hash": _hash(universe_manifest),
                    "config_hash": _hash(config_path),
                    "folds_manifest": str(fold_manifest.resolve()),
                    "folds_manifest_hash": _hash(fold_manifest),
                    "selection_period": {
                        "start_date": "20150101",
                        "end_date": "20221231",
                        "folds": [
                            {
                                "fold_id": "fold_selection",
                                "evaluation_start": "20220104",
                                "evaluation_end": "20221230",
                            }
                        ],
                        "may_select_model": True,
                    },
                    "final_test_period": {
                        "start_date": "20230101",
                        "end_date": "20260710",
                        "folds": [
                            {
                                "fold_id": "fold_test",
                                "evaluation_start": "20230103",
                                "evaluation_end": "20231229",
                            }
                        ],
                        "may_select_model": False,
                    },
                }
            ],
        },
    )
    settings = AppSettings(
        ranker=RankerSettings(
            n_estimators=3,
            num_leaves=3,
            min_child_samples=2,
            feature_fraction=1.0,
            bagging_fraction=1.0,
            minimum_group_size=3,
        )
    )
    return (
        ChallengerTrainer(
            processed_root=processed_root,
            models_root=models_root,
            reports_root=reports_root,
            settings=settings,
            config_path=config_path,
        ),
        {
            "config": config_path,
            "features_manifest": features_manifest,
            "universe_manifest": universe_manifest,
            "fold_manifest": fold_manifest,
            "folds": folds,
            "horizon_manifest": horizon_manifest,
            "models_root": models_root,
        },
    )


def _ranker_dataset(features: tuple[str, ...], trade_date: str) -> RankerDataset:
    frame = pd.DataFrame(
        {
            "trade_date": [trade_date] * 20,
            "ts_code": [f"{index:06d}.SZ" for index in range(20)],
            "f1": np.arange(20, dtype=np.float32),
            "f2": np.arange(20, dtype=np.float32) / 2,
            "future_excess_ret_5d": np.arange(20, dtype=float) / 100,
            "relevance": np.arange(20, dtype=np.int32) % 5,
        }
    )
    return RankerDataset(frame=frame, feature_names=features)


def _refresh_fold_hashes(sources: dict[str, Path]) -> None:
    fold_manifest = _read_json(sources["fold_manifest"])
    atomic_write_json(sources["fold_manifest"], fold_manifest)
    horizon = _read_json(sources["horizon_manifest"])
    horizon["folds_manifest_hash"] = _hash(sources["fold_manifest"])
    horizon["folds_hash"] = _hash(sources["folds"])
    horizon["experiments"][0]["folds_manifest_hash"] = _hash(sources["fold_manifest"])
    atomic_write_json(sources["horizon_manifest"], horizon)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
