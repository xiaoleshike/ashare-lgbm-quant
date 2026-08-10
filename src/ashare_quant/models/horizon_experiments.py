"""Read-only multi-horizon challenger experiment planning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import pandas as pd

from ashare_quant.config.settings import AppSettings, HorizonExperimentSettings
from ashare_quant.data.datasets import get_dataset_spec
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.storage import ParquetDataStore
from ashare_quant.models.feature_provenance import (
    feature_provenance_hash,
    validate_governed_feature_set,
)
from ashare_quant.models.registry import ModelRegistry, RegisteredModel
from ashare_quant.models.research_policy import enforce_research_window, load_research_policy
from ashare_quant.models.temporal_isolation import required_temporal_gap_sessions
from ashare_quant.utils.manifest import atomic_write_json, config_hash, current_git_info

HORIZON_PLAN_SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class HorizonExperimentPlanResult:
    """Published multi-horizon experiment plan identity."""

    run_id: str
    output_dir: Path
    experiment_count: int
    source_model_id: str
    folds_manifest: Path


class MultiHorizonExperimentPlanner:
    """Bind isolated label horizons to one pre-existing safe fold plan."""

    def __init__(
        self,
        *,
        raw_root: Path,
        models_root: Path,
        processed_root: Path,
        reports_root: Path,
        settings: AppSettings,
        config_path: Path,
        research_policy_path: Path = Path("config/research_policy.yaml"),
    ) -> None:
        self.raw_root = raw_root
        self.models_root = models_root
        self.processed_root = processed_root
        self.reports_root = reports_root
        self.settings = settings
        self.config_path = config_path
        self.research_policy_path = research_policy_path

    def build(self, *, folds_manifest: Path | None = None) -> HorizonExperimentPlanResult:
        """Create a plan using label availability metadata without loading target values."""

        champion = self._champion()
        experiments = self.settings.models.horizon_experiments
        maximum_horizon = max(experiment.horizon for experiment in experiments)
        fold_path, fold_manifest, fold_payload = self._resolve_folds_manifest(
            folds_manifest,
            required_gap=required_temporal_gap_sessions(maximum_horizon),
        )
        feature_hash = str(fold_manifest["feature_set_hash"])
        feature_set_id = str(fold_manifest["feature_set_id"])
        feature_provenance_sha256 = str(fold_manifest["feature_provenance_hash"])
        sessions = _load_open_sessions(self.raw_root)
        selection_period = self.settings.models.selection_period
        final_test_period = self.settings.models.final_test_period
        research_policy = load_research_policy(self.research_policy_path)
        enforce_research_window(
            research_policy,
            consumer="horizon_selection",
            start_date=selection_period.start_date,
            end_date=selection_period.end_date,
        )
        enforce_research_window(
            research_policy,
            consumer="walk_forward_evaluation",
            start_date=final_test_period.start_date,
            end_date=final_test_period.end_date,
        )
        label_availability, labels_fingerprint = _load_label_availability(
            self.processed_root,
            tuple(experiment.horizon for experiment in experiments),
            selection_start=selection_period.start_date,
            selection_end=selection_period.end_date,
            test_start=final_test_period.start_date,
            test_end=final_test_period.end_date,
        )
        universe_manifest_path = self.processed_root / "universe_daily" / "_manifest.json"
        universe_manifest = _load_json(universe_manifest_path, "universe manifest")
        universe_hash = _file_hash(universe_manifest_path)
        features_manifest_path = self.processed_root / "features_daily" / "_manifest.json"
        _load_json(features_manifest_path, "features manifest")
        features_manifest_hash = _file_hash(features_manifest_path)
        fold_hash = _file_hash(fold_path)
        folds_file = _folds_file(fold_path, fold_manifest)
        folds_hash = _file_hash(folds_file)
        resolved_config_hash = config_hash(self.config_path)
        if resolved_config_hash is None:
            raise DataValidationError(f"configuration file is missing: {self.config_path}")
        git = current_git_info()
        identity_payload = {
            "schema_version": HORIZON_PLAN_SCHEMA_VERSION,
            "source_model_id": champion.model_id,
            "model_type": champion.model_type,
            "feature_hash": feature_hash,
            "feature_set_id": feature_set_id,
            "feature_provenance_hash": feature_provenance_sha256,
            "universe_hash": universe_hash,
            "features_manifest_hash": features_manifest_hash,
            "config_hash": resolved_config_hash,
            "folds_manifest_hash": fold_hash,
            "folds_hash": folds_hash,
            "labels_fingerprint": labels_fingerprint,
            "git_commit": git["commit"],
            "research_policy_hash": research_policy.policy_hash,
            "experiments": [_experiment_config(experiment) for experiment in experiments],
            "selection_period": selection_period.model_dump(mode="json"),
            "final_test_period": final_test_period.model_dump(mode="json"),
            "maturity_cutoffs": {
                str(experiment.horizon): _maximum_mature_date(sessions, experiment.horizon)
                for experiment in experiments
            },
        }
        identity_hash = _payload_hash(identity_payload)
        run_id = f"horizon_plan_{identity_hash[:16]}"
        output_dir = self.reports_root / "horizon_experiments" / run_id
        output_path = output_dir / "experiment_manifest.json"
        if output_path.exists():
            existing = _load_json(output_path, "horizon experiment manifest")
            if existing.get("plan_identity_hash") != identity_hash:
                raise DataValidationError(f"horizon plan identity collision: {output_path}")
            return HorizonExperimentPlanResult(
                run_id=run_id,
                output_dir=output_dir,
                experiment_count=len(experiments),
                source_model_id=champion.model_id,
                folds_manifest=fold_path,
            )

        created_time = datetime.now(UTC).isoformat(timespec="seconds")
        experiment_records = [
            _experiment_record(
                experiment,
                identity_hash=identity_hash,
                created_time=created_time,
                feature_hash=feature_hash,
                universe_hash=universe_hash,
                config_hash_value=resolved_config_hash,
                model_type=champion.model_type,
                folds_manifest=fold_path,
                folds_manifest_hash=fold_hash,
                folds=fold_payload["folds"],
                sessions=sessions,
                selection_period=selection_period.model_dump(mode="json"),
                final_test_period=final_test_period.model_dump(mode="json"),
                label_availability=label_availability[experiment.horizon],
            )
            for experiment in experiments
        ]
        experiment_ids = [str(record["experiment_id"]) for record in experiment_records]
        if len(experiment_ids) != len(set(experiment_ids)):
            raise DataValidationError("multi-horizon experiment IDs are not unique")
        manifest: dict[str, Any] = {
            "schema_version": HORIZON_PLAN_SCHEMA_VERSION,
            "artifact_name": "multi_horizon_experiment_plan",
            "run_id": run_id,
            "plan_identity_hash": identity_hash,
            "created_time": created_time,
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "config_path": str(self.config_path),
            "config_hash": resolved_config_hash,
            "source_model_id": champion.model_id,
            "source_model_status": champion.status,
            "reference_champion_model_id": champion.model_id,
            "reference_champion_feature_hash": champion.feature_hash,
            "model_type": champion.model_type,
            "feature_authority": "governed_feature_set",
            "feature_set_id": feature_set_id,
            "feature_hash": feature_hash,
            "feature_set_hash": feature_hash,
            "feature_provenance_locator": fold_manifest["feature_provenance_locator"],
            "feature_provenance_hash": feature_provenance_sha256,
            "universe_hash": universe_hash,
            "universe_manifest": str(universe_manifest_path.resolve()),
            "universe_manifest_identity": {
                "artifact_name": universe_manifest.get("artifact_name"),
                "git_commit": universe_manifest.get("git_commit"),
                "config_hash": universe_manifest.get("config_hash"),
            },
            "features_manifest": str(features_manifest_path.resolve()),
            "features_manifest_hash": features_manifest_hash,
            "folds_manifest": str(fold_path.resolve()),
            "folds_manifest_hash": fold_hash,
            "folds_hash": folds_hash,
            "fold_count": len(fold_payload["folds"]),
            "labels_fingerprint": labels_fingerprint,
            "selection_period": {
                **selection_period.model_dump(mode="json"),
                "purpose": "challenger_comparison_only",
            },
            "final_test_period": {
                **final_test_period.model_dump(mode="json"),
                "purpose": "historical_holdout_evaluation_only",
                "classification": research_policy.historical_holdout.classification,
                "may_select_model": False,
            },
            "research_policy_path": str(self.research_policy_path),
            "research_policy_hash": research_policy.policy_hash,
            "prospective_lockbox_start": research_policy.prospective_lockbox.start_date,
            "experiments": experiment_records,
            "isolation_contract": {
                "model_trained": False,
                "champion_modified": False,
                "label_values_loaded": False,
                "label_availability_metadata_checked": True,
                "future_return_values_loaded": False,
                "test_results_used": False,
                "mixed_horizon_training": False,
                "folds_regenerated": False,
            },
        }
        atomic_write_json(output_path, manifest)
        return HorizonExperimentPlanResult(
            run_id=run_id,
            output_dir=output_dir,
            experiment_count=len(experiment_records),
            source_model_id=champion.model_id,
            folds_manifest=fold_path,
        )

    def _champion(self) -> RegisteredModel:
        champion = ModelRegistry(self.models_root).get_champion("lightgbm_ranker")
        if champion is None:
            raise DataValidationError("no lightgbm_ranker champion is registered")
        return champion

    def _resolve_folds_manifest(
        self,
        requested: Path | None,
        *,
        required_gap: int,
    ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        if requested is not None:
            path = requested / "manifest.json" if requested.is_dir() else requested
            manifest, folds = _validate_folds_manifest(
                path,
                required_gap=required_gap,
            )
            return path, manifest, folds

        candidates = sorted(
            (self.reports_root / "walk_forward").glob("*/manifest.json"), reverse=True
        )
        failures: list[str] = []
        for path in candidates:
            try:
                manifest, folds = _validate_folds_manifest(
                    path,
                    required_gap=required_gap,
                )
            except DataValidationError as error:
                failures.append(f"{path}: {error}")
                continue
            return path, manifest, folds
        detail = "" if not failures else f"; checked={len(failures)} incompatible plans"
        raise DataValidationError(
            "no compatible walk-forward manifest found; generate or specify an existing plan "
            f"with purge_sessions and embargo_sessions >= {required_gap}{detail}"
        )


def _validate_folds_manifest(
    path: Path,
    *,
    required_gap: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json(path, "walk-forward manifest")
    if manifest.get("artifact_name") != "purged_walk_forward_plan":
        raise DataValidationError(f"not a purged walk-forward manifest: {path}")
    if _required_int(manifest, "schema_version") != 4:
        raise DataValidationError(
            "WALK_FORWARD_FEATURE_AUTHORITY_INVALID: schema-v4 governed plan required"
        )
    if manifest.get("feature_authority") != "governed_feature_set":
        raise DataValidationError("WALK_FORWARD_FEATURE_AUTHORITY_INVALID")
    reports_root = path.resolve().parents[2]
    locator = manifest.get("feature_provenance_locator")
    if not isinstance(locator, str) or not locator:
        raise DataValidationError("WALK_FORWARD_FEATURE_PROVENANCE_MISMATCH: locator missing")
    provenance_path = reports_root / locator
    if feature_provenance_hash(provenance_path) != manifest.get("feature_provenance_hash"):
        raise DataValidationError("WALK_FORWARD_FEATURE_PROVENANCE_MISMATCH: hash changed")
    provenance = validate_governed_feature_set(provenance_path, reports_root=reports_root)
    if (
        provenance.feature_set_id != manifest.get("feature_set_id")
        or provenance.feature_list_hash != manifest.get("feature_set_hash")
        or manifest.get("feature_hash") != manifest.get("feature_set_hash")
    ):
        raise DataValidationError("WALK_FORWARD_FEATURE_PROVENANCE_MISMATCH: identity changed")
    policy = manifest.get("policy")
    if not isinstance(policy, dict):
        raise DataValidationError("walk-forward manifest has no policy")
    fixed_maturity_fields = {"label_horizon", "label_exit_lag_sessions"} & set(policy)
    if fixed_maturity_fields:
        raise DataValidationError(
            "walk-forward policy contains fixed-horizon maturity fields: "
            f"{sorted(fixed_maturity_fields)}"
        )
    purge = _required_int(policy, "purge_sessions")
    embargo = _required_int(policy, "embargo_sessions")
    if purge < required_gap or embargo < required_gap:
        raise DataValidationError(
            "walk-forward boundaries are unsafe for configured horizons: "
            f"purge={purge}, embargo={embargo}, required={required_gap}"
        )
    folds_file = _folds_file(path, manifest)
    folds_payload = _load_json(folds_file, "walk-forward folds")
    if _required_int(folds_payload, "schema_version") != 4:
        raise DataValidationError(
            "WALK_FORWARD_FEATURE_AUTHORITY_INVALID: folds schema-v4 required"
        )
    folds = folds_payload.get("folds")
    if not isinstance(folds, list) or not folds:
        raise DataValidationError("walk-forward folds are missing or empty")
    fold_ids: list[str] = []
    for fold in folds:
        if not isinstance(fold, dict):
            raise DataValidationError("walk-forward fold must be an object")
        fold_id = str(fold.get("fold_id", ""))
        fold_ids.append(fold_id)
        if fold.get("feature_hash") != manifest.get("feature_set_hash"):
            raise DataValidationError("WALK_FORWARD_FEATURE_PROVENANCE_MISMATCH: fold feature")
        fixed_fold_fields = {"label_horizon", "label_exit_lag_sessions"} & set(fold)
        if fixed_fold_fields:
            raise DataValidationError(f"fold contains fixed-horizon maturity fields: {fold_id}")
        if _required_int(fold, "purge_sessions") < required_gap:
            raise DataValidationError(f"fold purge is unsafe: {fold_id}")
        if _required_int(fold, "embargo_sessions") < required_gap:
            raise DataValidationError(f"fold embargo is unsafe: {fold_id}")
        if not (
            str(fold.get("train_end", "")) < str(fold.get("validation_start", ""))
            and str(fold.get("validation_end", "")) < str(fold.get("evaluation_start", ""))
        ):
            raise DataValidationError(f"fold chronology is invalid: {fold_id}")
    if any(not fold_id for fold_id in fold_ids) or len(fold_ids) != len(set(fold_ids)):
        raise DataValidationError("walk-forward fold IDs must be non-empty and unique")
    return manifest, folds_payload


def _folds_file(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    outputs = manifest.get("outputs")
    configured = outputs.get("folds") if isinstance(outputs, dict) else None
    if isinstance(configured, str) and configured:
        path = Path(configured)
        if path.is_absolute():
            return path
        local_path = manifest_path.parent / path
        if local_path.exists():
            return local_path
        # Schema-v2 plans previously stored a repository-root-relative path.
        for parent in manifest_path.parents:
            legacy_path = parent / path
            if legacy_path.exists():
                return legacy_path
        return local_path
    return manifest_path.parent / "folds.json"


def _experiment_config(experiment: HorizonExperimentSettings) -> dict[str, Any]:
    return {
        "name": experiment.name,
        "horizon": experiment.horizon,
        "holding_days": experiment.holding_days,
        "execution_rule": experiment.execution_rule,
    }


def _experiment_record(
    experiment: HorizonExperimentSettings,
    *,
    identity_hash: str,
    created_time: str,
    feature_hash: str,
    universe_hash: str,
    config_hash_value: str,
    model_type: str,
    folds_manifest: Path,
    folds_manifest_hash: str,
    folds: list[dict[str, Any]],
    sessions: tuple[str, ...],
    selection_period: dict[str, str],
    final_test_period: dict[str, str],
    label_availability: dict[str, Any],
) -> dict[str, Any]:
    maturity_sessions = required_temporal_gap_sessions(experiment.horizon)
    maximum_mature_date = _maximum_mature_date(sessions, experiment.horizon)
    selection_folds = _clip_folds(
        folds,
        start_date=selection_period["start_date"],
        end_date=selection_period["end_date"],
        maximum_mature_date=maximum_mature_date,
    )
    final_test_folds = _clip_folds(
        folds,
        start_date=final_test_period["start_date"],
        end_date=final_test_period["end_date"],
        maximum_mature_date=maximum_mature_date,
    )
    if not selection_folds:
        raise DataValidationError(
            f"horizon {experiment.horizon} has no mature selection-period folds"
        )
    if not final_test_folds:
        raise DataValidationError(f"horizon {experiment.horizon} has no mature final-test folds")
    experiment_identity = _payload_hash(
        {
            "plan_identity_hash": identity_hash,
            "name": experiment.name,
            "horizon": experiment.horizon,
        }
    )
    return {
        "experiment_id": f"{experiment.name}_{experiment_identity[:16]}",
        "name": experiment.name,
        "horizon": experiment.horizon,
        "holding_period": experiment.holding_days,
        "execution_rule": experiment.execution_rule,
        "label_name": f"future_excess_ret_{experiment.horizon}d",
        "label_maturity_sessions": maturity_sessions,
        "required_purge_sessions": maturity_sessions,
        "required_embargo_sessions": maturity_sessions,
        "maximum_mature_evaluation_date": maximum_mature_date,
        "feature_hash": feature_hash,
        "universe_hash": universe_hash,
        "config_hash": config_hash_value,
        "model_type": model_type,
        "created_time": created_time,
        "folds_manifest": str(folds_manifest.resolve()),
        "folds_manifest_hash": folds_manifest_hash,
        "selection_period": {
            **selection_period,
            "folds": selection_folds,
            "may_select_model": True,
        },
        "final_test_period": {
            **final_test_period,
            "folds": final_test_folds,
            "may_select_model": False,
        },
        "label_availability": label_availability,
    }


def _clip_folds(
    folds: list[dict[str, Any]],
    *,
    start_date: str,
    end_date: str,
    maximum_mature_date: str,
) -> list[dict[str, str]]:
    clipped: list[dict[str, str]] = []
    effective_end = min(end_date, maximum_mature_date)
    for fold in folds:
        evaluation_start = max(str(fold["evaluation_start"]), start_date)
        evaluation_end = min(str(fold["evaluation_end"]), effective_end)
        if evaluation_start > evaluation_end:
            continue
        clipped.append(
            {
                "fold_id": str(fold["fold_id"]),
                "evaluation_start": evaluation_start,
                "evaluation_end": evaluation_end,
            }
        )
    return clipped


def _load_open_sessions(raw_root: Path) -> tuple[str, ...]:
    frame = ParquetDataStore(raw_root).read_dataset(get_dataset_spec("trade_cal"))
    required = {"cal_date", "is_open"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"trade_cal is missing required columns: {missing}")
    open_rows = frame.loc[pd.to_numeric(frame["is_open"], errors="coerce") == 1, "cal_date"]
    sessions = tuple(sorted(open_rows.astype(str).unique().tolist()))
    if not sessions:
        raise DataValidationError("trade_cal contains no open sessions")
    return sessions


def _maximum_mature_date(sessions: tuple[str, ...], horizon: int) -> str:
    maturity_sessions = required_temporal_gap_sessions(horizon)
    if len(sessions) <= maturity_sessions:
        raise DataValidationError(
            f"trade_cal has insufficient sessions for horizon {horizon} maturity"
        )
    return sessions[-maturity_sessions - 1]


def _load_label_availability(
    processed_root: Path,
    horizons: tuple[int, ...],
    *,
    selection_start: str,
    selection_end: str,
    test_start: str,
    test_end: str,
) -> tuple[dict[int, dict[str, Any]], str]:
    dataset_dir = processed_root / "labels_forward"
    files = sorted(dataset_dir.glob("**/*.parquet"))
    if not files:
        raise DataValidationError(f"labels_forward does not exist: {dataset_dir}")
    glob = str(dataset_dir / "**/*.parquet")
    connection = duckdb.connect(":memory:")
    try:
        frame = connection.execute(
            """
            SELECT
              CAST(horizon AS BIGINT) AS horizon,
              COUNT(*) FILTER (
                WHERE is_label_available AND future_excess_ret IS NOT NULL
              ) AS available_rows,
              COUNT(*) FILTER (
                WHERE is_label_available AND future_excess_ret IS NOT NULL
                  AND trade_date >= ? AND trade_date <= ?
              ) AS selection_available_rows,
              COUNT(*) FILTER (
                WHERE is_label_available AND future_excess_ret IS NOT NULL
                  AND trade_date >= ? AND trade_date <= ?
              ) AS final_test_available_rows,
              MIN(trade_date) FILTER (
                WHERE is_label_available AND future_excess_ret IS NOT NULL
              ) AS minimum_available_date,
              MAX(trade_date) FILTER (
                WHERE is_label_available AND future_excess_ret IS NOT NULL
              ) AS maximum_available_date
            FROM read_parquet(?)
            GROUP BY horizon
            """,
            [selection_start, selection_end, test_start, test_end, glob],
        ).fetchdf()
    except (duckdb.Error, OSError) as error:
        raise DataValidationError(f"cannot inspect labels_forward availability: {error}") from error
    finally:
        connection.close()
    indexed = {_as_int(row.horizon): row for row in frame.itertuples(index=False)}
    availability: dict[int, dict[str, Any]] = {}
    for horizon in horizons:
        row = indexed.get(horizon)
        if row is None or _as_int(row.available_rows) == 0:
            raise DataValidationError(f"required label future_excess_ret_{horizon}d is unavailable")
        if _as_int(row.selection_available_rows) == 0:
            raise DataValidationError(
                f"required label future_excess_ret_{horizon}d has no selection-period rows"
            )
        if _as_int(row.final_test_available_rows) == 0:
            raise DataValidationError(
                f"required label future_excess_ret_{horizon}d has no final-test rows"
            )
        availability[horizon] = {
            "available_rows": _as_int(row.available_rows),
            "selection_available_rows": _as_int(row.selection_available_rows),
            "final_test_available_rows": _as_int(row.final_test_available_rows),
            "minimum_available_date": str(row.minimum_available_date),
            "maximum_available_date": str(row.maximum_available_date),
        }
    return availability, dataset_fingerprint(files, dataset_dir)


def dataset_fingerprint(files: list[Path], root: Path) -> str:
    """Return the deterministic inventory identity used by horizon plans."""

    entries = [
        {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in files
    ]
    return _payload_hash({"files": entries})


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"{description} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"cannot read {description}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"{description} must be a JSON object: {path}")
    return payload


def _required_int(payload: dict[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"walk-forward {field} must be an integer")
    return value


def _file_hash(path: Path) -> str:
    if not path.is_file():
        raise DataValidationError(f"manifest source does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()
