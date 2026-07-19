"""Immutable artifact registration and explicit champion lifecycle management."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.orchestration.lock import production_lock
from ashare_quant.utils.manifest import atomic_write_json, current_git_info

REGISTRY_SCHEMA_VERSION = 1
HISTORY_SCHEMA_VERSION = 1

type ModelStatus = Literal["candidate", "champion", "retired"]


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    """One immutable model artifact reference and its mutable lifecycle status."""

    model_id: str
    experiment_id: str
    model_type: str
    feature_hash: str
    feature_count: int
    training_date_range: dict[str, str]
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    git_commit: str | None
    config_hash: str | None
    creation_time: str
    artifact_path: str
    status: ModelStatus

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable registry record."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RegisteredModel:
        """Validate and deserialize one persisted registry record."""

        try:
            status = cast(ModelStatus, payload["status"])
            if status not in {"candidate", "champion", "retired"}:
                raise ValueError(f"unsupported model status: {status}")
            return cls(
                model_id=str(payload["model_id"]),
                experiment_id=str(payload["experiment_id"]),
                model_type=str(payload["model_type"]),
                feature_hash=str(payload["feature_hash"]),
                feature_count=int(payload["feature_count"]),
                training_date_range={
                    str(key): str(value)
                    for key, value in dict(payload["training_date_range"]).items()
                },
                validation_metrics=dict(payload["validation_metrics"]),
                test_metrics=dict(payload["test_metrics"]),
                git_commit=_optional_string(payload.get("git_commit")),
                config_hash=_optional_string(payload.get("config_hash")),
                creation_time=str(payload["creation_time"]),
                artifact_path=str(payload["artifact_path"]),
                status=status,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise DataValidationError(f"invalid model registry record: {error}") from error


class ModelRegistry:
    """Manage a single-host JSON registry without modifying model artifacts."""

    def __init__(self, models_root: Path) -> None:
        self.models_root = models_root
        self.registry_path = models_root / "registry.json"
        self.history_root = models_root / "registry_history"
        self.lock_path = models_root / ".registry.lock"

    def register_model(
        self,
        artifact_path: Path,
        *,
        model_id: str | None = None,
        model_type: str | None = None,
        operator_command: str = "register_model",
    ) -> RegisteredModel:
        """Register an existing artifact as a candidate without changing it."""

        artifact = artifact_path.resolve()
        manifest = _load_required_json(artifact / "manifest.json", "manifest")
        features = _load_required_json(artifact / "feature_list.json", "feature list")
        metrics = _load_required_json(artifact / "metrics.json", "metrics")
        feature_names = _feature_names(features, artifact)
        resolved_model_id = model_id or str(manifest.get("experiment_id") or artifact.name)
        _validate_identifier(resolved_model_id, "model_id")
        experiment_id = str(manifest.get("experiment_id") or artifact.name)
        resolved_model_type = model_type or _infer_model_type(manifest)
        _validate_identifier(resolved_model_type, "model_type")
        validation_metrics = metrics.get("validation")
        test_metrics = metrics.get("test")
        record = RegisteredModel(
            model_id=resolved_model_id,
            experiment_id=experiment_id,
            model_type=resolved_model_type,
            feature_hash=feature_list_hash(feature_names),
            feature_count=len(feature_names),
            training_date_range=_training_range(manifest),
            validation_metrics=(
                dict(validation_metrics) if isinstance(validation_metrics, dict) else {}
            ),
            test_metrics=dict(test_metrics) if isinstance(test_metrics, dict) else {},
            git_commit=_optional_string(manifest.get("git_commit")),
            config_hash=_optional_string(manifest.get("config_hash")),
            creation_time=_creation_time(manifest, artifact),
            artifact_path=str(artifact),
            status="candidate",
        )
        with production_lock(self.lock_path, command=operator_command):
            records = self._load_records()
            if any(existing.model_id == record.model_id for existing in records):
                raise DataValidationError(
                    f"model_id is already registered and cannot be overwritten: {record.model_id}"
                )
            records.append(record)
            self._persist(records)
            self._write_history(
                "register",
                operator_command,
                record.model_id,
                old_champion=None,
                new_champion=None,
            )
        return record

    def promote_model(
        self,
        model_id: str,
        *,
        operator_command: str | None = None,
    ) -> RegisteredModel:
        """Explicitly promote one validated candidate and demote its prior champion."""

        command = operator_command or f"ashare-quant models promote {model_id}"
        with production_lock(self.lock_path, command=command):
            records = self._load_records()
            selected = _find_model(records, model_id)
            if selected.status == "retired":
                raise DataValidationError(f"retired model cannot be promoted: {model_id}")
            self._validate_for_promotion(selected)
            old_champion = next(
                (
                    record
                    for record in records
                    if record.model_type == selected.model_type and record.status == "champion"
                ),
                None,
            )
            if old_champion is not None and old_champion.model_id == selected.model_id:
                return selected
            updated: list[RegisteredModel] = []
            for record in records:
                if record.model_id == selected.model_id:
                    updated.append(replace(record, status="champion"))
                elif record.model_type == selected.model_type and record.status == "champion":
                    updated.append(replace(record, status="candidate"))
                else:
                    updated.append(record)
            promoted = _find_model(updated, model_id)
            self._persist(updated)
            self._write_history(
                "promote",
                command,
                model_id,
                old_champion=None if old_champion is None else old_champion.model_id,
                new_champion=model_id,
            )
            return promoted

    def retire_model(
        self,
        model_id: str,
        *,
        operator_command: str | None = None,
    ) -> RegisteredModel:
        """Retire a registered model while retaining its record and artifact reference."""

        command = operator_command or f"ashare-quant models retire {model_id}"
        with production_lock(self.lock_path, command=command):
            records = self._load_records()
            selected = _find_model(records, model_id)
            if selected.status == "retired":
                return selected
            retired = replace(selected, status="retired")
            updated = [retired if record.model_id == model_id else record for record in records]
            self._persist(updated)
            self._write_history(
                "retire",
                command,
                model_id,
                old_champion=model_id if selected.status == "champion" else None,
                new_champion=None,
            )
            return retired

    def get_champion(self, model_type: str | None = None) -> RegisteredModel | None:
        """Return the champion for one type, or the sole champion across all types."""

        champions = [
            record
            for record in self._load_records()
            if record.status == "champion"
            and (model_type is None or record.model_type == model_type)
        ]
        if not champions:
            return None
        if len(champions) > 1 and model_type is None:
            raise DataValidationError(
                "multiple model types have champions; specify model_type explicitly"
            )
        return champions[0]

    def list_models(self) -> tuple[RegisteredModel, ...]:
        """Return all models, including retired records, in deterministic order."""

        return tuple(
            sorted(self._load_records(), key=lambda record: (record.creation_time, record.model_id))
        )

    def _validate_for_promotion(self, record: RegisteredModel) -> None:
        artifact = Path(record.artifact_path)
        failures: list[str] = []
        if not artifact.is_dir():
            failures.append(f"artifact directory does not exist: {artifact}")
        required_files = ("model.txt", "feature_list.json", "manifest.json", "metrics.json")
        for filename in required_files:
            if not (artifact / filename).is_file():
                failures.append(f"required artifact file is missing: {filename}")
        if failures:
            raise DataValidationError("model promotion validation failed: " + "; ".join(failures))

        feature_payload = _load_required_json(artifact / "feature_list.json", "feature list")
        manifest = _load_required_json(artifact / "manifest.json", "manifest")
        metrics = _load_required_json(artifact / "metrics.json", "metrics")
        computed_hash = feature_list_hash(_feature_names(feature_payload, artifact))
        declared_feature_hash = _optional_string(feature_payload.get("feature_hash"))
        manifest_feature_hash = _optional_string(manifest.get("feature_list_hash"))
        if computed_hash != record.feature_hash:
            failures.append(
                "feature hash differs from registered value: "
                f"{computed_hash} != {record.feature_hash}"
            )
        if declared_feature_hash is not None and declared_feature_hash != computed_hash:
            failures.append(
                "feature_list.json feature_hash does not match its ordered feature list"
            )
        if manifest_feature_hash is not None and manifest_feature_hash != computed_hash:
            failures.append("manifest feature_list_hash does not match feature_list.json")
        test_metrics = metrics.get("test")
        if not isinstance(test_metrics, dict) or not test_metrics:
            failures.append("metrics.json does not contain non-empty test metrics")
        if failures:
            raise DataValidationError("model promotion validation failed: " + "; ".join(failures))

    def _load_records(self) -> list[RegisteredModel]:
        if not self.registry_path.exists():
            return []
        payload = _load_required_json(self.registry_path, "model registry")
        if payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
            raise DataValidationError(
                f"unsupported model registry schema: {payload.get('schema_version')}"
            )
        raw_models = payload.get("models")
        if not isinstance(raw_models, list) or not all(
            isinstance(item, dict) for item in raw_models
        ):
            raise DataValidationError("model registry `models` must be an array of objects")
        records = [RegisteredModel.from_dict(item) for item in raw_models]
        ids = [record.model_id for record in records]
        if len(ids) != len(set(ids)):
            raise DataValidationError("model registry contains duplicate model_id values")
        champion_types = [record.model_type for record in records if record.status == "champion"]
        if len(champion_types) != len(set(champion_types)):
            raise DataValidationError("model registry contains multiple champions for a model_type")
        return records

    def _persist(self, records: list[RegisteredModel]) -> None:
        self.models_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.registry_path,
            {
                "schema_version": REGISTRY_SCHEMA_VERSION,
                "updated_at": _utc_now(),
                "models": [record.to_dict() for record in records],
            },
        )

    def _write_history(
        self,
        operation: str,
        operator_command: str,
        model_id: str,
        *,
        old_champion: str | None,
        new_champion: str | None,
    ) -> None:
        timestamp = datetime.now(UTC)
        git_info = current_git_info()
        filename = f"{operation}_{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}.json"
        atomic_write_json(
            self.history_root / filename,
            {
                "schema_version": HISTORY_SCHEMA_VERSION,
                "operation": operation,
                "operator_command": operator_command,
                "model_id": model_id,
                "old_champion": old_champion,
                "new_champion": new_champion,
                "git_commit": git_info["commit"],
                "git_dirty": git_info["dirty"],
                "timestamp": timestamp.isoformat(),
            },
        )


def _load_required_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"{description} JSON does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid {description} JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"{description} JSON must contain an object: {path}")
    return payload


def _feature_names(payload: dict[str, Any], artifact: Path) -> tuple[str, ...]:
    raw = payload.get("features")
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) for item in raw):
        raise DataValidationError(
            f"feature_list.json must contain a non-empty string array `features`: {artifact}"
        )
    names = tuple(str(item) for item in raw)
    if len(names) != len(set(names)):
        raise DataValidationError(f"feature_list.json contains duplicate features: {artifact}")
    return names


def _training_range(manifest: dict[str, Any]) -> dict[str, str]:
    start = manifest.get("train_start", manifest.get("training_start"))
    end = manifest.get("train_end", manifest.get("training_end"))
    if not isinstance(start, str) or not isinstance(end, str):
        raise DataValidationError(
            "model manifest must contain train_start/train_end or training_start/training_end"
        )
    return {"start": start, "end": end}


def _creation_time(manifest: dict[str, Any], artifact: Path) -> str:
    value = manifest.get("completed_at", manifest.get("creation_time"))
    if isinstance(value, str) and value:
        return value
    return datetime.fromtimestamp(artifact.stat().st_mtime, tz=UTC).isoformat()


def _infer_model_type(manifest: dict[str, Any]) -> str:
    artifact_name = str(manifest.get("artifact_name", ""))
    if "ranker" in artifact_name.lower():
        return "lightgbm_ranker"
    raise DataValidationError(
        "model_type cannot be inferred from manifest; pass model_type explicitly"
    )


def _find_model(records: list[RegisteredModel], model_id: str) -> RegisteredModel:
    try:
        return next(record for record in records if record.model_id == model_id)
    except StopIteration as error:
        raise DataValidationError(f"model_id is not registered: {model_id}") from error


def _validate_identifier(value: str, label: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise DataValidationError(f"{label} must be a non-empty simple identifier: {value}")


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
