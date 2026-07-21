"""Purged chronological fold planning without model fitting or label access."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ashare_quant.config.settings import AppSettings, WalkForwardPlanSettings
from ashare_quant.data.datasets import get_dataset_spec
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.storage import ParquetDataStore
from ashare_quant.models.registry import ModelRegistry, RegisteredModel
from ashare_quant.utils.manifest import (
    atomic_write_json,
    config_hash,
    current_git_info,
)

PLAN_SCHEMA_VERSION = 2
type WalkForwardScheme = Literal["expanding", "rolling"]


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One leakage-controlled train, validation, and evaluation split."""

    fold_id: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    evaluation_start: str
    evaluation_end: str
    purge_sessions: int
    embargo_sessions: int
    feature_hash: str
    model_id: str
    scheme: WalkForwardScheme
    train_sessions: int
    validation_sessions: int
    evaluation_sessions: int

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON representation used as the fold manifest."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class WalkForwardPlanResult:
    """Published walk-forward planning result."""

    run_id: str
    output_dir: Path
    fold_count: int
    scheme: WalkForwardScheme
    model_id: str


class PurgedWalkForwardPlanner:
    """Build monthly experiment folds from the authoritative trading calendar."""

    def __init__(
        self,
        *,
        raw_root: Path,
        models_root: Path,
        reports_root: Path,
        settings: AppSettings,
        config_path: Path,
    ) -> None:
        self.raw_root = raw_root
        self.models_root = models_root
        self.reports_root = reports_root
        self.settings = settings
        self.config_path = config_path

    def build(
        self,
        *,
        start_date: str,
        end_date: str,
        scheme: WalkForwardScheme,
        model_id: str | None = None,
        purge_days: int | None = None,
        embargo_days: int | None = None,
        rolling_years: int | None = None,
    ) -> WalkForwardPlanResult:
        """Publish a read-only walk-forward plan without loading labels or fitting a model."""

        _validate_date_range(start_date, end_date)
        model = self._resolve_model(model_id)
        policy = self.settings.ranker.walk_forward
        purge = policy.purge_days if purge_days is None else purge_days
        embargo = policy.embargo_days if embargo_days is None else embargo_days
        rolling_window_years = (
            policy.rolling_window_years if rolling_years is None else rolling_years
        )
        _validate_runtime_policy(policy, purge, embargo, rolling_window_years)
        if scheme not in {"expanding", "rolling"}:
            raise DataValidationError(f"unsupported walk-forward scheme: {scheme}")

        sessions = self._load_open_sessions(start_date, end_date)
        folds = _build_folds(
            sessions,
            scheme=scheme,
            model=model,
            policy=policy,
            purge_sessions=purge,
            embargo_sessions=embargo,
            rolling_window_years=rolling_window_years,
        )
        if not folds:
            required = (
                policy.minimum_training_years * policy.annual_sessions
                + policy.validation_sessions
                + purge
                + embargo
                + 1
            )
            raise DataValidationError(
                "date range cannot produce a walk-forward fold: "
                f"available_sessions={len(sessions)} required_at_least={required}"
            )

        run_id = _run_id(
            start_date=start_date,
            end_date=end_date,
            scheme=scheme,
            model_id=model.model_id,
        )
        output_dir = self.reports_root / "walk_forward" / run_id
        fold_payload: dict[str, Any] = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "run_id": run_id,
            "folds": [fold.to_dict() for fold in folds],
        }
        git = current_git_info()
        manifest: dict[str, Any] = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "artifact_name": "purged_walk_forward_plan",
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "config_path": str(self.config_path),
            "config_hash": config_hash(self.config_path),
            "requested_start_date": start_date,
            "requested_end_date": end_date,
            "scheme": scheme,
            "fold_count": len(folds),
            "model_id": model.model_id,
            "model_status": model.status,
            "model_artifact_path": model.artifact_path,
            "feature_hash": model.feature_hash,
            "policy": {
                "annual_sessions": policy.annual_sessions,
                "minimum_training_years": policy.minimum_training_years,
                "rolling_window_years": rolling_window_years,
                "validation_sessions": policy.validation_sessions,
                "purge_sessions": purge,
                "embargo_sessions": embargo,
                "evaluation_frequency": policy.evaluation_frequency,
            },
            "trade_calendar": {
                "source": str(self.raw_root / "trade_cal"),
                "session_count": len(sessions),
                "minimum_session": sessions[0],
                "maximum_session": sessions[-1],
            },
            "leakage_contract": {
                "calendar_source": "trade_cal open sessions",
                "labels_loaded": False,
                "model_fitted": False,
                "fold_boundaries_are_horizon_agnostic": True,
                "label_maturity_must_be_validated_by_horizon_plan": True,
            },
            "outputs": {
                "folds": str(output_dir / "folds.json"),
                "manifest": str(output_dir / "manifest.json"),
            },
        }
        atomic_write_json(output_dir / "folds.json", fold_payload)
        atomic_write_json(output_dir / "manifest.json", manifest)
        return WalkForwardPlanResult(
            run_id=run_id,
            output_dir=output_dir,
            fold_count=len(folds),
            scheme=scheme,
            model_id=model.model_id,
        )

    def _resolve_model(self, model_id: str | None) -> RegisteredModel:
        registry = ModelRegistry(self.models_root)
        if model_id is None:
            champion = registry.get_champion("lightgbm_ranker")
            if champion is None:
                raise DataValidationError(
                    "no lightgbm_ranker champion is registered; pass --model-id or promote a model"
                )
            return champion
        selected = next(
            (record for record in registry.list_models() if record.model_id == model_id),
            None,
        )
        if selected is None:
            raise DataValidationError(f"model_id is not registered: {model_id}")
        return selected

    def _load_open_sessions(self, start_date: str, end_date: str) -> tuple[str, ...]:
        store = ParquetDataStore(self.raw_root)
        spec = get_dataset_spec("trade_cal")
        frame = store.read_dataset(spec, start_date, end_date)
        required = {"cal_date", "is_open"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise DataValidationError(f"trade_cal is missing required columns: {missing}")
        open_rows = frame.loc[pd.to_numeric(frame["is_open"], errors="coerce") == 1].copy()
        open_rows["cal_date"] = open_rows["cal_date"].astype(str)
        dates = tuple(sorted(open_rows["cal_date"].unique().tolist()))
        if not dates:
            raise DataValidationError(
                f"trade_cal has no open sessions between {start_date} and {end_date}"
            )
        return dates


def _build_folds(
    sessions: tuple[str, ...],
    *,
    scheme: WalkForwardScheme,
    model: RegisteredModel,
    policy: WalkForwardPlanSettings,
    purge_sessions: int,
    embargo_sessions: int,
    rolling_window_years: int,
) -> tuple[WalkForwardFold, ...]:
    session_index = {date: index for index, date in enumerate(sessions)}
    months: dict[str, list[str]] = {}
    for date in sessions:
        months.setdefault(date[:6], []).append(date)
    minimum_training_sessions = policy.minimum_training_years * policy.annual_sessions
    rolling_window_sessions = rolling_window_years * policy.annual_sessions
    folds: list[WalkForwardFold] = []
    for month_dates in months.values():
        evaluation_start_index = session_index[month_dates[0]]
        validation_end_index = evaluation_start_index - embargo_sessions - 1
        validation_start_index = validation_end_index - policy.validation_sessions + 1
        train_end_index = validation_start_index - purge_sessions - 1
        if train_end_index < 0 or validation_start_index < 0:
            continue
        if scheme == "expanding":
            train_start_index = 0
            if train_end_index + 1 < minimum_training_sessions:
                continue
        else:
            train_start_index = train_end_index - rolling_window_sessions + 1
            if train_start_index < 0:
                continue
        fold_number = len(folds) + 1
        fold = WalkForwardFold(
            fold_id=f"fold_{fold_number:04d}_{month_dates[0][:6]}",
            train_start=sessions[train_start_index],
            train_end=sessions[train_end_index],
            validation_start=sessions[validation_start_index],
            validation_end=sessions[validation_end_index],
            evaluation_start=month_dates[0],
            evaluation_end=month_dates[-1],
            purge_sessions=purge_sessions,
            embargo_sessions=embargo_sessions,
            feature_hash=model.feature_hash,
            model_id=model.model_id,
            scheme=scheme,
            train_sessions=train_end_index - train_start_index + 1,
            validation_sessions=policy.validation_sessions,
            evaluation_sessions=len(month_dates),
        )
        _validate_fold(fold, session_index)
        folds.append(fold)
    return tuple(folds)


def _validate_fold(fold: WalkForwardFold, session_index: dict[str, int]) -> None:
    train_start = session_index[fold.train_start]
    train_end = session_index[fold.train_end]
    validation_start = session_index[fold.validation_start]
    validation_end = session_index[fold.validation_end]
    evaluation_start = session_index[fold.evaluation_start]
    evaluation_end = session_index[fold.evaluation_end]
    if not (
        train_start
        <= train_end
        < validation_start
        <= validation_end
        < evaluation_start
        <= evaluation_end
    ):
        raise DataValidationError(f"walk-forward fold chronology is invalid: {fold.fold_id}")
    if validation_start - train_end - 1 != fold.purge_sessions:
        raise DataValidationError(f"walk-forward purge boundary is invalid: {fold.fold_id}")
    if evaluation_start - validation_end - 1 != fold.embargo_sessions:
        raise DataValidationError(f"walk-forward embargo boundary is invalid: {fold.fold_id}")


def _validate_runtime_policy(
    policy: WalkForwardPlanSettings,
    purge_days: int,
    embargo_days: int,
    rolling_years: int,
) -> None:
    if purge_days < 0:
        raise DataValidationError("purge_days must not be negative")
    if embargo_days < 0:
        raise DataValidationError("embargo_days must not be negative")
    if rolling_years <= 0:
        raise DataValidationError("rolling_years must be positive")


def _validate_date_range(start_date: str, end_date: str) -> None:
    for name, value in (("start_date", start_date), ("end_date", end_date)):
        try:
            parsed = datetime.strptime(value, "%Y%m%d")
        except ValueError as error:
            raise DataValidationError(f"{name} must be YYYYMMDD: {value}") from error
        if parsed.strftime("%Y%m%d") != value:
            raise DataValidationError(f"{name} must be YYYYMMDD: {value}")
    if start_date > end_date:
        raise DataValidationError("start_date must not be after end_date")


def _run_id(*, start_date: str, end_date: str, scheme: str, model_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    identity = json.dumps(
        {
            "start_date": start_date,
            "end_date": end_date,
            "scheme": scheme,
            "model_id": model_id,
        },
        sort_keys=True,
    ).encode()
    digest = hashlib.sha256(identity).hexdigest()[:8]
    return f"walk_forward_{scheme}_{timestamp}_{digest}"
