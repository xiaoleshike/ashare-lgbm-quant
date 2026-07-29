"""Maturity-gated prospective performance observation orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ashare_quant.data.datasets import get_dataset_spec
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.storage import ParquetDataStore
from ashare_quant.labels.storage import LabelStore
from ashare_quant.models.shadow.storage import canonical_payload_hash
from ashare_quant.monitoring.performance_observation.maturity import (
    maturity_dates,
    open_sessions,
    require_observation_session,
)
from ashare_quant.monitoring.performance_observation.metrics import (
    build_performance_metrics,
)
from ashare_quant.monitoring.performance_observation.schemas import (
    OBSERVATION_COLUMNS,
    OBSERVATION_KEY,
    SUPPORTED_HORIZONS,
    PerformanceObservationResult,
)
from ashare_quant.monitoring.performance_observation.storage import (
    load_observation_history,
    logical_observation_hash,
    observation_content_hash,
    publish_observation_artifact,
    read_observation_artifact,
)
from ashare_quant.monitoring.performance_observation.validation import (
    load_shadow_sources,
)
from ashare_quant.utils.manifest import config_hash, current_git_info, read_manifest

type DataFrame = pd.DataFrame


class PerformanceObservationService:
    """Join mature prospective predictions to immutable realized labels."""

    def __init__(
        self,
        *,
        raw_root: Path,
        processed_root: Path,
        reports_root: Path,
        config_path: Path,
        runs_root: Path = Path("runs"),
    ) -> None:
        self.raw_store = ParquetDataStore(raw_root)
        self.label_store = LabelStore(processed_root)
        self.reports_root = reports_root
        self.config_path = config_path
        self.runs_root = runs_root

    def run(self, observation_as_of: str) -> PerformanceObservationResult:
        """Publish only observations that mature for the first time by this cutoff."""

        _validate_date(observation_as_of)
        calendar = self.raw_store.read_dataset(get_dataset_spec("trade_cal"))
        sessions = open_sessions(calendar)
        require_observation_session(sessions, observation_as_of)
        maturity_sessions = tuple(date for date in sessions if date <= observation_as_of)
        shadow, shadow_manifests, shadow_hashes = load_shadow_sources(
            reports_root=self.reports_root,
            runs_root=self.runs_root,
            observation_as_of=observation_as_of,
        )
        if shadow.empty:
            raise DataValidationError("no prospective shadow predictions are available")

        output_root = self.reports_root / "performance_observation"
        output_dir = output_root / observation_as_of
        history, history_hashes = load_observation_history(
            output_root,
            before_or_on=observation_as_of,
            exclude_date=observation_as_of,
        )
        candidates = _expand_horizons(shadow)
        candidates = _attach_maturity(
            candidates,
            sessions=maturity_sessions,
            observation_as_of=observation_as_of,
        )
        mature_signal_dates = set(candidates["signal_date"].astype(str))
        shadow_hashes = {
            signal_date: digest
            for signal_date, digest in shadow_hashes.items()
            if signal_date in mature_signal_dates
        }
        shadow_manifests = [
            manifest
            for manifest in shadow_manifests
            if str(manifest.get("source_signal_date")) in mature_signal_dates
        ]
        labels = self._load_labels(candidates)
        built = _join_labels(candidates, labels, observation_as_of)
        new_rows = _enforce_append_only(built, history)
        new_rows = (
            new_rows.loc[:, list(OBSERVATION_COLUMNS)]
            .sort_values(list(OBSERVATION_KEY), kind="mergesort")
            .reset_index(drop=True)
        )
        combined = pd.concat([history, new_rows], ignore_index=True)
        metrics = build_performance_metrics(combined)
        observation_hash = logical_observation_hash(new_rows)
        manifest = self._manifest(
            observation_as_of=observation_as_of,
            observations=new_rows,
            observation_hash=observation_hash,
            metrics=metrics,
            shadow_manifests=shadow_manifests,
            shadow_hashes=shadow_hashes,
            history_hashes=history_hashes,
            calendar=maturity_sessions,
            labels=labels,
        )

        existing = read_observation_artifact(output_dir)
        if existing is not None:
            existing_rows, existing_manifest = existing
            if (
                existing_manifest.get("observation_hash") != observation_hash
                or existing_manifest.get("source_identity_hash") != manifest["source_identity_hash"]
            ):
                raise DataValidationError(
                    "existing performance observation has different immutable content"
                )
            return PerformanceObservationResult(
                observation_as_of,
                len(existing_rows),
                int(existing_rows["label_status"].eq("available").sum()),
                output_dir,
                output_dir / "manifest.json",
                idempotent=True,
            )
        if output_dir.exists():
            raise DataValidationError(
                f"incomplete performance observation output exists: {output_dir}"
            )
        publish_observation_artifact(
            output_dir=output_dir,
            observations=new_rows,
            metrics=metrics,
            manifest=manifest,
        )
        return PerformanceObservationResult(
            observation_as_of,
            len(new_rows),
            int(new_rows["label_status"].eq("available").sum()),
            output_dir,
            output_dir / "manifest.json",
        )

    def _load_labels(self, candidates: DataFrame) -> DataFrame:
        if candidates.empty:
            return self.label_store.read("99999999", "99999999")
        start = str(candidates["signal_date"].min())
        end = str(candidates["signal_date"].max())
        horizons = sorted(pd.to_numeric(candidates["horizon"], errors="raise").astype(int).unique())
        labels = pd.concat(
            [self.label_store.read(start, end, horizon=int(horizon)) for horizon in horizons],
            ignore_index=True,
        )
        keys = candidates.loc[:, ["signal_date", "ts_code", "horizon"]].drop_duplicates()
        keys = keys.rename(columns={"signal_date": "trade_date"})
        keys["trade_date"] = keys["trade_date"].astype(str)
        keys["ts_code"] = keys["ts_code"].astype(str)
        keys["horizon"] = pd.to_numeric(keys["horizon"], errors="raise").astype(int)
        labels["trade_date"] = labels["trade_date"].astype(str)
        labels["ts_code"] = labels["ts_code"].astype(str)
        labels["horizon"] = pd.to_numeric(labels["horizon"], errors="raise").astype(int)
        labels = labels.merge(
            keys,
            on=["trade_date", "ts_code", "horizon"],
            how="inner",
            validate="one_to_one",
        )
        if labels.empty:
            raise DataValidationError(
                f"mature labels are missing for prospective signals {start}..{end}"
            )
        return labels

    def _manifest(
        self,
        *,
        observation_as_of: str,
        observations: DataFrame,
        observation_hash: str,
        metrics: dict[str, Any],
        shadow_manifests: list[dict[str, Any]],
        shadow_hashes: dict[str, str],
        history_hashes: dict[str, str],
        calendar: tuple[str, ...],
        labels: DataFrame,
    ) -> dict[str, Any]:
        label_manifest = read_manifest(self.label_store.dataset_dir)
        label_manifest_identity = (
            {
                "schema_version": label_manifest.get("schema_version"),
                "artifact_name": label_manifest.get("artifact_name"),
            }
            if isinstance(label_manifest, dict)
            else None
        )
        source_identity = {
            "observation_as_of": observation_as_of,
            "shadow_prediction_hashes": shadow_hashes,
            "history_observation_hashes": history_hashes,
            "calendar_hash": canonical_payload_hash(calendar),
            "label_hash": _label_hash(labels),
            "label_manifest_identity": label_manifest_identity,
            "config_hash": config_hash(self.config_path),
        }
        git = current_git_info()
        return {
            "schema_version": 1,
            "artifact_name": "performance_observation",
            "observation_as_of": observation_as_of,
            "observation_hash": observation_hash,
            "source_identity_hash": canonical_payload_hash(source_identity),
            "row_count": len(observations),
            "available_rows": int(observations["label_status"].eq("available").sum()),
            "model_ids": sorted(observations["model_id"].astype(str).unique()),
            "model_lineage": _model_lineage(shadow_manifests),
            "horizons": sorted(
                pd.to_numeric(observations["horizon"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            ),
            "maturity_through": observation_as_of,
            "shadow_prediction_hashes": shadow_hashes,
            "shadow_run_ids": sorted(
                {str(manifest.get("shadow_run_id")) for manifest in shadow_manifests}
            ),
            "production_run_ids": sorted(
                {str(manifest.get("production_run_id")) for manifest in shadow_manifests}
            ),
            "history_observation_hashes": history_hashes,
            "label_hash": source_identity["label_hash"],
            "label_manifest_identity": label_manifest_identity,
            "calendar_hash": source_identity["calendar_hash"],
            "metrics_summary": {
                "available_rows": metrics["available_rows"],
                "model_horizon_groups": len(metrics["models"]),
            },
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "config_hash": config_hash(self.config_path),
            "generated_at": datetime.now(UTC).isoformat(),
            "access_policy": "prospective_production",
            "contracts": {
                "labels_used_only_after_maturity": True,
                "historical_predictions_used": False,
                "inference_called": False,
                "backtest_called": False,
                "paper_trading_called": False,
                "registry_modified": False,
            },
        }


def _model_lineage(shadow_manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve model identity without requiring downstream shadow-artifact reads."""

    lineage: dict[str, dict[str, Any]] = {}
    for manifest in shadow_manifests:
        models = manifest.get("models")
        if not isinstance(models, list):
            continue
        for raw in models:
            if not isinstance(raw, dict):
                continue
            model_id = str(raw.get("model_id", ""))
            if not model_id:
                continue
            record = {
                "model_id": model_id,
                "model_role": str(raw.get("model_role", "")),
                "feature_hash": str(raw.get("feature_hash") or manifest.get("feature_hash") or ""),
                "universe_hash": str(
                    raw.get("universe_hash") or manifest.get("universe_hash") or ""
                ),
                "source_models": sorted(str(value) for value in raw.get("source_models", [])),
                "fusion_method": raw.get("fusion_method"),
            }
            previous = lineage.get(model_id)
            if previous is not None and previous != record:
                raise DataValidationError(f"shadow model lineage changed: {model_id}")
            lineage[model_id] = record
    return [lineage[key] for key in sorted(lineage)]


def _expand_horizons(shadow: DataFrame) -> DataFrame:
    frames: list[DataFrame] = []
    for _, row in shadow.iterrows():
        role = str(row["model_role"])
        if role == "multi_horizon_ensemble":
            horizons = SUPPORTED_HORIZONS
        else:
            value = pd.to_numeric(pd.Series([row["native_horizon"]]), errors="coerce").iloc[0]
            if pd.isna(value):
                raise DataValidationError(f"shadow model lacks native horizon: {role}")
            horizons = (int(value),)
        for horizon in horizons:
            record = row.to_dict()
            record["signal_date"] = str(row["trade_date"])
            record["horizon"] = horizon
            frames.append(pd.DataFrame([record]))
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(list(OBSERVATION_KEY)).any():
        raise DataValidationError("expanded shadow predictions contain duplicate identities")
    return result


def _attach_maturity(
    frame: DataFrame,
    *,
    sessions: tuple[str, ...],
    observation_as_of: str,
) -> DataFrame:
    records: list[dict[str, object]] = []
    for (signal_date, horizon), group in frame.groupby(["signal_date", "horizon"], sort=True):
        resolved_signal_date = str(signal_date)
        resolved_horizon = int(str(horizon))
        try:
            entry_date, exit_date = maturity_dates(
                sessions,
                resolved_signal_date,
                resolved_horizon,
            )
        except DataValidationError as error:
            if "cannot determine maturity" in str(error):
                continue
            raise
        if exit_date > observation_as_of:
            continue
        working = group.copy()
        working["expected_entry_date"] = entry_date
        working["expected_exit_date"] = exit_date
        records.extend(
            {str(column): value for column, value in record.items()}
            for record in working.to_dict("records")
        )
    if not records:
        return frame.iloc[:0].assign(
            expected_entry_date=pd.Series(dtype=str),
            expected_exit_date=pd.Series(dtype=str),
        )
    return pd.DataFrame.from_records(records)


def _join_labels(
    candidates: DataFrame,
    labels: DataFrame,
    observation_as_of: str,
) -> DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=list(OBSERVATION_COLUMNS))
    selected_labels = labels.loc[
        :,
        [
            "trade_date",
            "ts_code",
            "horizon",
            "entry_date",
            "exit_date",
            "future_excess_ret",
            "is_label_available",
            "label_unavailable_reason",
        ],
    ].copy()
    selected_labels = selected_labels.rename(columns={"trade_date": "signal_date"})
    for column in ("signal_date", "ts_code", "entry_date", "exit_date"):
        selected_labels[column] = selected_labels[column].astype(str)
    selected_labels["horizon"] = pd.to_numeric(selected_labels["horizon"], errors="raise").astype(
        int
    )
    if selected_labels.duplicated(["signal_date", "ts_code", "horizon"]).any():
        raise DataValidationError("labels_forward contains duplicate observation keys")
    merged = candidates.merge(
        selected_labels,
        on=["signal_date", "ts_code", "horizon"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if merged["_merge"].ne("both").any():
        missing = int(merged["_merge"].ne("both").sum())
        raise DataValidationError(f"mature prospective labels are missing; rows={missing}")
    if (
        merged["entry_date"].ne(merged["expected_entry_date"])
        | merged["exit_date"].ne(merged["expected_exit_date"])
    ).any():
        raise DataValidationError("label entry/exit dates violate trade_cal maturity contract")
    available = merged["is_label_available"].fillna(False).astype(bool)
    returns = pd.to_numeric(merged["future_excess_ret"], errors="coerce")
    if returns.loc[available].isna().any():
        raise DataValidationError("available mature labels contain null future_excess_ret")
    if returns.loc[~available].notna().any():
        raise DataValidationError("unavailable mature labels contain future_excess_ret")
    reason = merged["label_unavailable_reason"].fillna("").astype(str)
    if reason.loc[~available].eq("").any():
        raise DataValidationError("unavailable mature labels lack a reason")
    merged["label_status"] = np.where(available, "available", reason)
    merged["future_excess_ret"] = returns
    merged["observation_as_of"] = observation_as_of
    merged["observation_id"] = merged.apply(
        lambda row: canonical_payload_hash(
            {
                "model_id": str(row["model_id"]),
                "signal_date": str(row["signal_date"]),
                "ts_code": str(row["ts_code"]),
                "horizon": int(row["horizon"]),
            }
        ),
        axis=1,
    )
    return merged


def _enforce_append_only(current: DataFrame, history: DataFrame) -> DataFrame:
    if history.empty:
        return current
    historical = {
        tuple(row[column] for column in OBSERVATION_KEY): observation_content_hash(row)
        for _, row in history.iterrows()
    }
    keep: list[bool] = []
    for _, row in current.iterrows():
        key = tuple(row[column] for column in OBSERVATION_KEY)
        previous = historical.get(key)
        if previous is None:
            keep.append(True)
            continue
        if previous != observation_content_hash(row):
            raise DataValidationError(f"append-only observation content changed for identity={key}")
        keep.append(False)
    return current.loc[keep].copy()


def _label_hash(labels: DataFrame) -> str:
    columns = [
        "trade_date",
        "ts_code",
        "horizon",
        "entry_date",
        "exit_date",
        "future_excess_ret",
        "is_label_available",
        "label_unavailable_reason",
    ]
    ordered = labels.loc[:, columns].sort_values(
        ["trade_date", "horizon", "ts_code"], kind="mergesort"
    )
    normalized = ordered.astype(object).where(ordered.notna(), None)
    return canonical_payload_hash(normalized.to_dict("records"))


def _validate_date(value: str) -> None:
    if len(value) != 8 or not value.isdigit():
        raise DataValidationError(f"performance observation as_of must use YYYYMMDD: {value}")
