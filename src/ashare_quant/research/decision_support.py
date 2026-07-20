"""Read-only, rules-based investment decision support for human review."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from ashare_quant.config.settings import DecisionSupportSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.utils.manifest import atomic_write_json

type DataFrame = pd.DataFrame

_SAFE_COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class DecisionSupportResult:
    """Published human-review decision-support artifacts."""

    as_of: str
    model_id: str
    candidate_count: int
    json_path: Path
    markdown_path: Path


class InvestmentDecisionSupport:
    """Describe same-session candidates without changing ranking or creating orders."""

    def __init__(
        self,
        *,
        raw_root: Path,
        processed_root: Path,
        reports_root: Path,
        settings: DecisionSupportSettings,
    ) -> None:
        self.raw_root = raw_root
        self.processed_root = processed_root
        self.reports_root = reports_root
        self.settings = settings

    def generate(self, as_of: str) -> DecisionSupportResult:
        """Generate deterministic JSON and Markdown from existing research artifacts."""

        report_dir = self.reports_root / as_of
        predictions = _load_predictions(report_dir / "predictions.parquet", as_of)
        candidates = _load_candidates(report_dir / "candidates.csv", as_of)
        explanation_payload = _load_json(report_dir / "explanations.json", "explanations")
        model_id, feature_hash, explanations = _validate_identity(
            predictions, candidates, explanation_payload, as_of, self.settings.score_tolerance
        )
        feature_names = self._feature_names()
        codes = tuple(candidates["ts_code"].astype(str))
        features = _read_as_of(
            self.processed_root,
            "features_daily",
            as_of,
            codes,
            feature_names,
        )
        daily_basic = _read_as_of(
            self.raw_root,
            "daily_basic",
            as_of,
            codes,
            ("turnover_rate", "total_mv"),
        )
        merged = candidates.merge(features, on=["trade_date", "ts_code"], validate="one_to_one")
        merged = merged.merge(
            daily_basic, on=["trade_date", "ts_code"], how="left", validate="one_to_one"
        )
        records = [
            self._stock_record(row, explanations[str(row["ts_code"])])
            for _, row in merged.sort_values(["rank", "ts_code"], kind="mergesort").iterrows()
        ]
        payload = {
            "schema_version": 1,
            "artifact_name": "daily_investment_decision_support",
            "as_of": as_of,
            "model_id": model_id,
            "feature_hash": feature_hash,
            "candidate_count": len(records),
            "rules": self.settings.model_dump(mode="json"),
            "disclaimer": (
                "Human-review decision support only. This artifact does not change model ranking, "
                "predict prices, create orders, set positions, or define fixed exit levels."
            ),
            "stocks": records,
        }
        json_path = report_dir / "decision.json"
        markdown_path = report_dir / "decision_report.md"
        _publish(json_path, markdown_path, payload, _render_markdown(payload))
        return DecisionSupportResult(as_of, model_id, len(records), json_path, markdown_path)

    def _feature_names(self) -> tuple[str, ...]:
        names = (
            self.settings.gap_feature,
            self.settings.ma20_feature,
            self.settings.amount_ratio_feature,
            self.settings.liquidity_feature,
            self.settings.short_return_feature,
            self.settings.volatility_feature,
        )
        invalid = sorted({name for name in names if _SAFE_COLUMN.fullmatch(name) is None})
        if invalid:
            raise DataValidationError(f"decision support contains invalid feature names: {invalid}")
        return tuple(dict.fromkeys(names))

    def _stock_record(
        self,
        row: pd.Series[Any],
        explanation: dict[str, Any],
    ) -> dict[str, Any]:
        values = {
            "open_gap": _optional_float(row.get(self.settings.gap_feature)),
            "ma20_ratio": _optional_float(row.get(self.settings.ma20_feature)),
            "amount_ratio_20d": _optional_float(row.get(self.settings.amount_ratio_feature)),
            "amihud_20d": _optional_float(row.get(self.settings.liquidity_feature)),
            "short_return": _optional_float(row.get(self.settings.short_return_feature)),
            "realized_volatility": _optional_float(row.get(self.settings.volatility_feature)),
            "turnover_rate": _optional_float(row.get("turnover_rate")),
            "total_mv": _optional_float(row.get("total_mv")),
        }
        technical = _technical_state(values)
        return {
            "ts_code": str(row["ts_code"]),
            "model_rank": int(explanation["model_rank"]),
            "candidate_rank": int(row["rank"]),
            "prediction_score": float(row["prediction_score"]),
            "signal_strength": str(explanation["signal_strength"]),
            "confidence": str(explanation["confidence"]),
            "positive_contributions": explanation["positive_contributions"],
            "negative_contributions": explanation["negative_contributions"],
            "technical_state": technical,
            "watch_entry_conditions": self._watch_conditions(values),
            "risk_observations": self._risk_observations(values),
        }

    def _watch_conditions(self, values: dict[str, float | None]) -> list[dict[str, object]]:
        return [
            _condition(
                "open_gap_within_range",
                values["open_gap"],
                lambda value: abs(value) <= self.settings.maximum_abs_open_gap,
                f"开盘缺口绝对值不高于 {self.settings.maximum_abs_open_gap:.2%}",
            ),
            _condition(
                "at_or_above_ma20",
                values["ma20_ratio"],
                lambda value: value >= self.settings.minimum_ma20_ratio,
                f"相对MA20不低于 {self.settings.minimum_ma20_ratio:.2%}",
            ),
            _condition(
                "amount_activity_sufficient",
                values["amount_ratio_20d"],
                lambda value: value >= self.settings.minimum_amount_ratio,
                f"成交额不低于20日均值的 {self.settings.minimum_amount_ratio:.2f} 倍",
            ),
            _condition(
                "turnover_sufficient",
                values["turnover_rate"],
                lambda value: value >= self.settings.minimum_turnover_rate,
                f"换手率不低于 {self.settings.minimum_turnover_rate:.2f}%",
            ),
            _condition(
                "liquidity_pressure_acceptable",
                values["amihud_20d"],
                lambda value: value <= self.settings.maximum_amihud,
                f"Amihud流动性压力不高于 {self.settings.maximum_amihud:.8g}",
            ),
        ]

    def _risk_observations(self, values: dict[str, float | None]) -> list[dict[str, object]]:
        risks: list[dict[str, object]] = []
        _append_risk(
            risks,
            "short_return_elevated",
            values["short_return"],
            lambda value: value >= self.settings.excessive_short_return,
            "短期涨幅较高",
        )
        _append_risk(
            risks,
            "volatility_elevated",
            values["realized_volatility"],
            lambda value: value >= self.settings.elevated_volatility,
            "近期波动增加",
        )
        _append_risk(
            risks,
            "amount_activity_declining",
            values["amount_ratio_20d"],
            lambda value: value <= self.settings.liquidity_decline_amount_ratio,
            "成交额相对20日均值下降",
        )
        _append_risk(
            risks,
            "liquidity_pressure_high",
            values["amihud_20d"],
            lambda value: value > self.settings.maximum_amihud,
            "流动性压力偏高",
        )
        return risks


def _load_predictions(path: Path, as_of: str) -> DataFrame:
    if not path.is_file():
        raise DataValidationError(f"predictions are missing: {path}")
    frame = pd.read_parquet(path)
    required = {"trade_date", "ts_code", "prediction_score", "model_id"}
    _require_columns(frame, required, "predictions")
    if frame.empty or set(frame["trade_date"].astype(str)) != {as_of}:
        raise DataValidationError("predictions are empty or contain a date other than --as-of")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise DataValidationError("predictions contain duplicate keys")
    return frame.assign(
        trade_date=frame["trade_date"].astype(str), ts_code=frame["ts_code"].astype(str)
    )


def _load_candidates(path: Path, as_of: str) -> DataFrame:
    if not path.is_file():
        raise DataValidationError(f"candidates are missing: {path}")
    frame = pd.read_csv(path, dtype={"trade_date": str, "ts_code": str, "model_id": str})
    required = {"rank", "trade_date", "ts_code", "prediction_score", "model_id"}
    _require_columns(frame, required, "candidates")
    if frame.empty or set(frame["trade_date"].astype(str)) != {as_of}:
        raise DataValidationError("candidates are empty or contain a date other than --as-of")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise DataValidationError("candidates contain duplicate keys")
    return frame


def _validate_identity(
    predictions: DataFrame,
    candidates: DataFrame,
    payload: dict[str, Any],
    as_of: str,
    tolerance: float,
) -> tuple[str, str, dict[str, dict[str, Any]]]:
    model_ids = set(predictions["model_id"].astype(str))
    if len(model_ids) != 1:
        raise DataValidationError("predictions must contain exactly one model_id")
    model_id = next(iter(model_ids))
    if set(candidates["model_id"].astype(str)) != {model_id}:
        raise DataValidationError("candidate model_id does not match predictions")
    if str(payload.get("as_of")) != as_of or str(payload.get("model_id")) != model_id:
        raise DataValidationError("explanation date or model_id does not match inputs")
    feature_hash = payload.get("feature_hash")
    if not isinstance(feature_hash, str) or not feature_hash:
        raise DataValidationError("explanations lack feature_hash")
    ranked_predictions = predictions.sort_values(
        ["prediction_score", "ts_code"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    ranked_predictions["model_rank"] = np.arange(1, len(ranked_predictions) + 1, dtype=int)
    prediction_scores = dict(
        zip(
            ranked_predictions["ts_code"].astype(str),
            ranked_predictions["prediction_score"].astype(float),
            strict=True,
        )
    )
    prediction_ranks = dict(
        zip(
            ranked_predictions["ts_code"].astype(str),
            ranked_predictions["model_rank"].astype(int),
            strict=True,
        )
    )
    explanations: dict[str, dict[str, Any]] = {}
    raw_stocks = payload.get("stocks")
    if not isinstance(raw_stocks, list):
        raise DataValidationError("explanations lack stock records")
    for item in raw_stocks:
        if isinstance(item, dict) and isinstance(item.get("ts_code"), str):
            explanations[str(item["ts_code"])] = item
    for _, candidate in candidates.iterrows():
        code = str(candidate["ts_code"])
        if code not in prediction_scores or code not in explanations:
            raise DataValidationError(f"candidate is missing prediction or explanation: {code}")
        scores = (
            float(candidate["prediction_score"]),
            prediction_scores[code],
            float(explanations[code]["prediction_score"]),
        )
        if max(scores) - min(scores) >= tolerance:
            raise DataValidationError(f"candidate prediction score identity mismatch: {code}")
        if int(explanations[code]["candidate_rank"]) != int(candidate["rank"]):
            raise DataValidationError(f"candidate rank differs from explanation: {code}")
        if int(explanations[code]["model_rank"]) != prediction_ranks[code]:
            raise DataValidationError(f"model rank differs from predictions: {code}")
    if set(explanations) != set(candidates["ts_code"].astype(str)):
        raise DataValidationError("explanation stock set differs from candidate stock set")
    return model_id, feature_hash, explanations


def _read_as_of(
    root: Path,
    dataset: str,
    as_of: str,
    codes: tuple[str, ...],
    columns: tuple[str, ...],
) -> DataFrame:
    dataset_dir = root / dataset
    if not list(dataset_dir.glob("**/*.parquet")):
        raise DataValidationError(f"required decision-support dataset is missing: {dataset}")
    invalid = sorted(column for column in columns if _SAFE_COLUMN.fullmatch(column) is None)
    if invalid:
        raise DataValidationError(f"invalid configured columns for {dataset}: {invalid}")
    glob = dataset_dir / "**" / "*.parquet"
    selected = ", ".join(f'CAST("{column}" AS DOUBLE) AS "{column}"' for column in columns)
    placeholders = ", ".join("?" for _ in codes)
    query = f"""
        SELECT CAST(trade_date AS VARCHAR) AS trade_date,
               CAST(ts_code AS VARCHAR) AS ts_code,
               {selected}
        FROM read_parquet('{glob.as_posix()}', hive_partitioning=false)
        WHERE CAST(trade_date AS VARCHAR) = ?
          AND CAST(ts_code AS VARCHAR) IN ({placeholders})
        ORDER BY ts_code
    """  # noqa: S608 -- validated columns and parameterized values
    try:
        with duckdb.connect() as connection:
            frame = connection.execute(query, [as_of, *codes]).fetch_df()
    except duckdb.Error as error:
        raise DataValidationError(f"cannot read {dataset} for {as_of}: {error}") from error
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise DataValidationError(f"{dataset} contains duplicate candidate rows")
    if dataset == "features_daily" and set(frame["ts_code"].astype(str)) != set(codes):
        raise DataValidationError("features_daily lacks one or more candidate rows")
    return frame


def _technical_state(values: dict[str, float | None]) -> dict[str, object]:
    ma20 = values["ma20_ratio"]
    amount = values["amount_ratio_20d"]
    return {
        **values,
        "ma20_status": (
            "unavailable" if ma20 is None else "above_or_equal" if ma20 >= 0 else "below"
        ),
        "amount_activity": (
            "unavailable" if amount is None else "above_average" if amount >= 1 else "below_average"
        ),
    }


def _condition(
    rule: str,
    value: float | None,
    predicate: Callable[[float], bool],
    description: str,
) -> dict[str, object]:
    status = "unavailable" if value is None else "met" if predicate(value) else "watch"
    return {"rule": rule, "status": status, "value": value, "description": description}


def _append_risk(
    risks: list[dict[str, object]],
    rule: str,
    value: float | None,
    predicate: Callable[[float], bool],
    description: str,
) -> None:
    if value is not None and predicate(value):
        risks.append({"rule": rule, "value": value, "description": description})


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Investment Decision Support Report",
        "",
        f"- Date: {payload['as_of']}",
        f"- Model ID: `{payload['model_id']}`",
        f"- Candidates: {payload['candidate_count']}",
        "",
        "> Human-review support only. This report does not change ranking, predict prices, "
        "create orders, set positions, or define stop-loss/take-profit levels.",
    ]
    stocks = payload["stocks"]
    assert isinstance(stocks, list)
    for stock in stocks:
        assert isinstance(stock, dict)
        lines.extend(
            [
                "",
                f"## {stock['candidate_rank']}. {stock['ts_code']}",
                "",
                f"- Model rank: {stock['model_rank']}",
                f"- Prediction score: {float(stock['prediction_score']):.10f}",
                f"- Signal strength: `{stock['signal_strength']}`",
                f"- Explanation history confidence: `{stock['confidence']}`",
                "",
                "### Main SHAP Contributions",
                "",
                *_render_contributions(stock["positive_contributions"]),
                "",
                "### Negative SHAP Factors",
                "",
                *_render_contributions(stock["negative_contributions"]),
                "",
                "### Current Technical State",
                "",
                *_render_technical(stock["technical_state"]),
                "",
                "### Watch Entry Conditions",
                "",
                *_render_observations(stock["watch_entry_conditions"], include_status=True),
                "",
                "### Risk Observations",
                "",
                *_render_observations(stock["risk_observations"], include_status=False),
            ]
        )
    return "\n".join(lines) + "\n"


def _render_contributions(value: object) -> list[str]:
    rows = value if isinstance(value, list) else []
    if not rows:
        return ["- None"]
    return [
        f"- `{row['feature']}`: {float(row['shap']):+.8f} - {row['description']}"
        for row in rows
        if isinstance(row, dict)
    ]


def _render_technical(value: object) -> list[str]:
    if not isinstance(value, dict):
        return ["- Unavailable"]
    return [f"- {key}: {_display(item)}" for key, item in value.items()]


def _render_observations(value: object, *, include_status: bool) -> list[str]:
    rows = value if isinstance(value, list) else []
    if not rows:
        return ["- No configured risk observation triggered."]
    rendered = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = f" [{row['status']}]" if include_status else ""
        rendered.append(f"- {row['description']}{status}; observed={_display(row.get('value'))}")
    return rendered


def _display(value: object) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _optional_float(value: object) -> float | None:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _require_columns(frame: DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"{name} lack required columns: {missing}")


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"{name} are missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"cannot read {name}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"{name} must contain a JSON object")
    return payload


def _publish(
    json_path: Path,
    markdown_path: Path,
    payload: dict[str, Any],
    markdown: str,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=json_path.parent) as temporary:
        staging = Path(temporary)
        staged_markdown = staging / markdown_path.name
        staged_markdown.write_text(markdown, encoding="utf-8")
        atomic_write_json(staging / json_path.name, payload)
        os.replace(staged_markdown, markdown_path)
        os.replace(staging / json_path.name, json_path)
