"""Point-in-time candidate filtering over production model predictions."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from ashare_quant.config.settings import CandidateSelectionSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.utils.manifest import atomic_write_json, config_hash, current_git_info

type DataFrame = pd.DataFrame

CANDIDATE_COLUMNS = (
    "rank",
    "ts_code",
    "prediction_score",
    "selection_reason",
    "trade_date",
    "model_id",
)


@dataclass(frozen=True, slots=True)
class CandidateSelectionResult:
    """Published candidate selection result for one signal date."""

    as_of: str
    model_id: str
    prediction_count: int
    candidate_count: int
    filtered_counts: dict[str, int]
    output_path: Path
    candidates: DataFrame


class CandidateSelector:
    """Apply configured signal-date eligibility and data-quality filters."""

    def __init__(
        self,
        *,
        raw_root: Path,
        processed_root: Path,
        reports_root: Path,
        config_path: Path,
        settings: CandidateSelectionSettings,
    ) -> None:
        self.raw_root = raw_root
        self.processed_root = processed_root
        self.reports_root = reports_root
        self.config_path = config_path
        self.settings = settings

    def select(self, as_of: str) -> CandidateSelectionResult:
        """Filter and rank one existing production prediction artifact."""

        report_dir = self.reports_root / as_of
        prediction_manifest = _load_json(report_dir / "manifest.json", "prediction manifest")
        summary = _load_json(report_dir / "summary.json", "prediction summary")
        predictions = _load_predictions(report_dir / "predictions.parquet", as_of)
        model_id, feature_hash = _validate_prediction_identity(
            predictions, prediction_manifest, as_of
        )
        inputs = _load_filter_inputs(self.raw_root, self.processed_root, as_of)
        candidates, filtered_counts = _apply_filters(predictions, inputs, self.settings)
        self._publish(
            report_dir,
            as_of,
            model_id,
            feature_hash,
            predictions,
            candidates,
            filtered_counts,
            prediction_manifest,
            summary,
        )
        return CandidateSelectionResult(
            as_of=as_of,
            model_id=model_id,
            prediction_count=len(predictions),
            candidate_count=len(candidates),
            filtered_counts=filtered_counts,
            output_path=report_dir / "candidates.csv",
            candidates=candidates,
        )

    def _publish(
        self,
        report_dir: Path,
        as_of: str,
        model_id: str,
        feature_hash: str,
        predictions: DataFrame,
        candidates: DataFrame,
        filtered_counts: dict[str, int],
        prediction_manifest: dict[str, Any],
        prediction_summary: dict[str, Any],
    ) -> None:
        generated_time = datetime.now(UTC).isoformat()
        git_info = current_git_info()
        summary = dict(prediction_summary)
        summary.update(
            {
                "prediction_count": len(predictions),
                "candidate_count": len(candidates),
                "filtered_counts": filtered_counts,
            }
        )
        manifest = {
            "schema_version": 1,
            "artifact_name": "production_candidates",
            "as_of": as_of,
            "model_id": model_id,
            "prediction_manifest": prediction_manifest,
            "feature_hash": feature_hash,
            "config_path": str(self.config_path),
            "config_hash": config_hash(self.config_path),
            "filtering_rules": self.settings.model_dump(mode="json"),
            "prediction_count": len(predictions),
            "candidate_count": len(candidates),
            "filtered_counts": filtered_counts,
            "generated_time": generated_time,
            "git_commit": git_info["commit"],
            "git_dirty": git_info["dirty"],
        }
        report_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=report_dir) as temporary:
            staging = Path(temporary)
            candidates.to_csv(staging / "candidates.csv", index=False)
            candidates.to_csv(staging / "candidate.csv", index=False)
            atomic_write_json(staging / "summary.json", summary)
            atomic_write_json(staging / "candidates_manifest.json", manifest)
            for filename in ("candidates.csv", "candidate.csv", "summary.json"):
                os.replace(staging / filename, report_dir / filename)
            os.replace(
                staging / "candidates_manifest.json",
                report_dir / "candidates_manifest.json",
            )


def _load_predictions(path: Path, as_of: str) -> DataFrame:
    if not path.is_file():
        raise DataValidationError(f"production predictions do not exist: {path}")
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as error:
        raise DataValidationError(f"cannot read production predictions {path}: {error}") from error
    required = {"trade_date", "ts_code", "prediction_score", "model_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"production predictions are missing columns: {missing}")
    frame = frame.loc[:, ["trade_date", "ts_code", "prediction_score", "model_id"]].copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["ts_code"] = frame["ts_code"].astype(str)
    if frame.empty:
        raise DataValidationError(f"production predictions are empty for {as_of}")
    if not frame["trade_date"].eq(as_of).all():
        raise DataValidationError(f"production predictions contain dates other than {as_of}")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise DataValidationError(f"production predictions contain duplicate keys for {as_of}")
    frame["prediction_score"] = pd.to_numeric(frame["prediction_score"], errors="coerce")
    if not np.isfinite(frame["prediction_score"].to_numpy(dtype=float)).all():
        raise DataValidationError("production predictions contain non-finite scores")
    return frame.sort_values(
        ["prediction_score", "ts_code"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def _validate_prediction_identity(
    predictions: DataFrame, manifest: dict[str, Any], as_of: str
) -> tuple[str, str]:
    model_ids = predictions["model_id"].dropna().astype(str).unique().tolist()
    if len(model_ids) != 1:
        raise DataValidationError(f"predictions must contain exactly one model_id: {model_ids}")
    model_id = model_ids[0]
    if manifest.get("artifact_name") != "production_predictions":
        raise DataValidationError("prediction manifest has an unexpected artifact_name")
    if str(manifest.get("as_of")) != as_of:
        raise DataValidationError("prediction manifest as_of does not match requested date")
    if str(manifest.get("model_id")) != model_id:
        raise DataValidationError("prediction manifest model_id does not match predictions")
    feature_hash = manifest.get("feature_hash")
    if not isinstance(feature_hash, str) or not feature_hash:
        raise DataValidationError("prediction manifest lacks feature_hash")
    return model_id, feature_hash


def _load_filter_inputs(raw_root: Path, processed_root: Path, as_of: str) -> DataFrame:
    universe = _read_date_partition(
        processed_root,
        "universe_daily",
        as_of,
        (
            "ts_code",
            "is_st",
            "is_suspended",
            "in_model_universe",
            "is_low_liquidity",
            "list_days",
        ),
    )
    daily = _read_date_partition(
        raw_root,
        "daily",
        as_of,
        ("ts_code", "open", "high", "low", "close", "amount"),
    )
    daily_basic = _read_date_partition(
        raw_root,
        "daily_basic",
        as_of,
        ("ts_code", "turnover_rate", "total_mv"),
    )
    limits = _read_date_partition(
        raw_root, "stk_limit", as_of, ("ts_code", "up_limit", "down_limit")
    )
    for name, frame in (
        ("universe_daily", universe),
        ("daily", daily),
        ("daily_basic", daily_basic),
        ("stk_limit", limits),
    ):
        if frame.duplicated("ts_code").any():
            raise DataValidationError(f"{name} contains duplicate ts_code rows for {as_of}")
        frame[f"has_{name}"] = True
    merged = universe.merge(daily, on="ts_code", how="outer", validate="one_to_one")
    merged = merged.merge(daily_basic, on="ts_code", how="outer", validate="one_to_one")
    return merged.merge(limits, on="ts_code", how="outer", validate="one_to_one")


def _read_date_partition(
    root: Path,
    dataset: str,
    as_of: str,
    columns: tuple[str, ...],
) -> DataFrame:
    dataset_dir = root / dataset
    if not list(dataset_dir.glob("**/*.parquet")):
        raise DataValidationError(f"required candidate input dataset is missing: {dataset}")
    glob = dataset_dir / "**" / "*.parquet"
    selected = ", ".join(f'"{column}"' for column in columns)
    query = f"""
        SELECT {selected}
        FROM read_parquet('{glob.as_posix()}', hive_partitioning=false)
        WHERE CAST(trade_date AS VARCHAR) = ?
        ORDER BY ts_code
    """  # noqa: S608 -- fixed identifiers and configured local Parquet path
    try:
        with duckdb.connect() as connection:
            frame = connection.execute(query, [as_of]).fetch_df()
    except duckdb.Error as error:
        raise DataValidationError(
            f"cannot load candidate input {dataset} for {as_of}: {error}"
        ) from error
    if frame.empty:
        raise DataValidationError(f"candidate input {dataset} has no rows for {as_of}")
    frame["ts_code"] = frame["ts_code"].astype(str)
    return frame


def _apply_filters(
    predictions: DataFrame,
    inputs: DataFrame,
    settings: CandidateSelectionSettings,
) -> tuple[DataFrame, dict[str, int]]:
    working = predictions.merge(inputs, on="ts_code", how="left", validate="one_to_one")
    reasons = pd.Series(pd.NA, index=working.index, dtype="string")

    def exclude(mask: pd.Series, reason: str) -> None:
        selected = reasons.isna() & mask.fillna(False).astype(bool)
        reasons.loc[selected] = reason

    exclude(working["has_universe_daily"].isna(), "missing_universe")
    if settings.require_model_universe:
        exclude(~_boolean(working, "in_model_universe"), "not_in_model_universe")
    codes = working["ts_code"].astype(str).str.upper()
    if settings.exclude_bj_market:
        exclude(codes.str.endswith(".BJ"), "bj_market")
    if settings.exclude_star_market:
        exclude(codes.str.match(r"^688\d{3}\.SH$"), "star_market")
    if settings.exclude_chinext_market:
        exclude(codes.str.match(r"^300\d{3}\.SZ$"), "chinext_market")
    if settings.exclude_st:
        exclude(_boolean(working, "is_st"), "st")
    if settings.exclude_suspended:
        exclude(_boolean(working, "is_suspended"), "suspended")
    if settings.exclude_low_liquidity:
        exclude(_boolean(working, "is_low_liquidity"), "insufficient_liquidity")
    list_days = pd.to_numeric(working["list_days"], errors="coerce")
    exclude(list_days.isna() | list_days.lt(settings.min_list_trading_days), "newly_listed")
    if settings.require_daily_row:
        exclude(working["has_daily"].isna(), "missing_daily")
    if settings.require_daily_basic_row:
        exclude(working["has_daily_basic"].isna(), "missing_daily_basic")
    if settings.require_stk_limit_row:
        exclude(working["has_stk_limit"].isna(), "missing_stk_limit")
    if settings.require_valid_ohlc:
        exclude(~_valid_ohlc(working), "invalid_ohlc")
    if settings.min_total_mv is not None:
        total_mv = pd.to_numeric(working["total_mv"], errors="coerce")
        exclude(
            total_mv.isna() | total_mv.lt(settings.min_total_mv),
            "insufficient_market_cap",
        )
    if settings.min_daily_amount is not None:
        amount = pd.to_numeric(working["amount"], errors="coerce")
        exclude(
            amount.isna() | amount.lt(settings.min_daily_amount),
            "insufficient_daily_amount",
        )
    if settings.min_turnover_rate is not None:
        turnover = pd.to_numeric(working["turnover_rate"], errors="coerce")
        exclude(
            turnover.isna() | turnover.lt(settings.min_turnover_rate),
            "insufficient_turnover",
        )
    if settings.max_turnover_rate is not None:
        turnover = pd.to_numeric(working["turnover_rate"], errors="coerce")
        exclude(
            turnover.isna() | turnover.gt(settings.max_turnover_rate),
            "excessive_turnover",
        )
    if settings.require_valid_price_limits:
        exclude(
            ~_valid_price_limits(working, settings.price_limit_tolerance),
            "abnormal_price_limits",
        )

    filtered_counts = {
        str(reason): int(count) for reason, count in reasons.dropna().value_counts().items()
    }
    eligible = working.loc[
        reasons.isna(), ["ts_code", "prediction_score", "trade_date", "model_id"]
    ]
    eligible = eligible.sort_values(
        ["prediction_score", "ts_code"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    if len(eligible) > settings.max_candidates:
        filtered_counts["below_candidate_cutoff"] = len(eligible) - settings.max_candidates
        eligible = eligible.iloc[: settings.max_candidates].copy()
    eligible.insert(0, "rank", np.arange(1, len(eligible) + 1, dtype=int))
    eligible.insert(3, "selection_reason", "passed_configured_filters")
    candidates = eligible.loc[:, list(CANDIDATE_COLUMNS)].reset_index(drop=True)
    return candidates, dict(sorted(filtered_counts.items()))


def _boolean(frame: DataFrame, column: str) -> pd.Series:
    return frame[column].fillna(False).astype(bool)


def _valid_ohlc(frame: DataFrame) -> pd.Series:
    values = {
        name: pd.to_numeric(frame[name], errors="coerce")
        for name in ("open", "high", "low", "close")
    }
    return (
        values["open"].gt(0)
        & values["high"].gt(0)
        & values["low"].gt(0)
        & values["close"].gt(0)
        & values["high"].ge(values["low"])
        & values["high"].ge(values["open"])
        & values["high"].ge(values["close"])
        & values["low"].le(values["open"])
        & values["low"].le(values["close"])
    )


def _valid_price_limits(frame: DataFrame, tolerance: float) -> pd.Series:
    close = pd.to_numeric(frame["close"], errors="coerce")
    up_limit = pd.to_numeric(frame["up_limit"], errors="coerce")
    down_limit = pd.to_numeric(frame["down_limit"], errors="coerce")
    return (
        close.gt(0)
        & up_limit.gt(0)
        & down_limit.gt(0)
        & up_limit.gt(down_limit)
        & close.le(up_limit + tolerance)
        & close.ge(down_limit - tolerance)
    )


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"{description} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid {description} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"{description} must contain a JSON object: {path}")
    return payload
