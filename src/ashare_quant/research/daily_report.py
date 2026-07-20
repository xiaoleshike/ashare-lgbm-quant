"""Deterministic daily research reporting over unchanged candidate rankings."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from ashare_quant.config.settings import DailyResearchReportSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.utils.manifest import atomic_write_json

type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class DailyReportResult:
    """Summary of one published daily quantitative research report."""

    as_of: str
    model_id: str
    candidate_count: int
    report_path: Path
    summary_path: Path
    warnings: tuple[str, ...]


class DailyResearchReportGenerator:
    """Describe model candidates without changing ranking or creating trades."""

    def __init__(
        self,
        *,
        raw_root: Path,
        processed_root: Path,
        reports_root: Path,
        settings: DailyResearchReportSettings,
    ) -> None:
        self.raw_root = raw_root
        self.processed_root = processed_root
        self.reports_root = reports_root
        self.settings = settings

    def generate(self, as_of: str) -> DailyReportResult:
        """Generate deterministic Markdown and JSON for one candidate date."""

        report_dir = self.reports_root / as_of
        candidates = _load_candidates(report_dir / "candidates.csv", as_of)
        predictions = _load_predictions(report_dir / "predictions.parquet", as_of)
        model_id = _validate_candidate_identity(candidates, predictions)
        warnings: list[str] = []
        codes = tuple(candidates["ts_code"].astype(str))
        universe = _read_optional_as_of(
            self.processed_root,
            "universe_daily",
            as_of,
            ("ts_code", "market", "industry", "is_limit_up"),
            warnings,
        )
        daily_basic = _read_optional_as_of(
            self.raw_root,
            "daily_basic",
            as_of,
            ("ts_code", "total_mv"),
            warnings,
        )
        daily_history = _read_optional_daily_history(
            self.raw_root,
            as_of,
            codes,
            self.settings.volatility_window,
            warnings,
        )
        enriched = _enrich_candidates(
            candidates,
            universe,
            daily_basic,
            daily_history,
            self.settings,
            warnings,
        )
        summary = _build_summary(
            as_of,
            model_id,
            len(predictions),
            enriched,
            self.settings,
            warnings,
        )
        markdown = _render_markdown(summary, enriched, self.settings)
        report_path = report_dir / "daily_report.md"
        summary_path = report_dir / "research_summary.json"
        _publish(report_dir, report_path, summary_path, markdown, summary)
        return DailyReportResult(
            as_of=as_of,
            model_id=model_id,
            candidate_count=len(candidates),
            report_path=report_path,
            summary_path=summary_path,
            warnings=tuple(warnings),
        )


def _load_candidates(path: Path, as_of: str) -> DataFrame:
    required = ("rank", "ts_code", "prediction_score", "trade_date", "model_id")
    if not path.is_file():
        raise DataValidationError(f"candidate report does not exist: {path}")
    try:
        frame = pd.read_csv(path, dtype={"trade_date": str, "ts_code": str, "model_id": str})
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise DataValidationError(f"cannot read candidate report {path}: {error}") from error
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise DataValidationError(f"candidate report is missing columns: {missing}")
    frame = frame.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["ts_code"] = frame["ts_code"].astype(str)
    if not frame.empty and not frame["trade_date"].eq(as_of).all():
        raise DataValidationError(f"candidate report contains dates other than {as_of}")
    if frame.duplicated("ts_code").any():
        raise DataValidationError(f"candidate report contains duplicate ts_code for {as_of}")
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame["prediction_score"] = pd.to_numeric(frame["prediction_score"], errors="coerce")
    if not frame.empty and (
        frame["rank"].isna().any()
        or not np.isfinite(frame["prediction_score"].to_numpy(dtype=float)).all()
    ):
        raise DataValidationError("candidate report contains invalid rank or prediction score")
    return frame.sort_values(["rank", "ts_code"], kind="mergesort").reset_index(drop=True)


def _load_predictions(path: Path, as_of: str) -> DataFrame:
    required = ("trade_date", "ts_code", "prediction_score", "model_id")
    if not path.is_file():
        raise DataValidationError(f"prediction artifact does not exist: {path}")
    try:
        frame = pd.read_parquet(path, columns=list(required))
    except (OSError, ValueError) as error:
        raise DataValidationError(f"cannot read prediction artifact {path}: {error}") from error
    if frame.empty:
        raise DataValidationError(f"prediction artifact is empty for {as_of}")
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["ts_code"] = frame["ts_code"].astype(str)
    if not frame["trade_date"].eq(as_of).all():
        raise DataValidationError(f"prediction artifact contains dates other than {as_of}")
    if frame.duplicated("ts_code").any():
        raise DataValidationError(f"prediction artifact contains duplicate ts_code for {as_of}")
    return frame


def _validate_candidate_identity(candidates: DataFrame, predictions: DataFrame) -> str:
    prediction_models = predictions["model_id"].dropna().astype(str).unique().tolist()
    if len(prediction_models) != 1:
        raise DataValidationError("prediction artifact must contain exactly one model_id")
    model_id = str(prediction_models[0])
    if candidates.empty:
        return model_id
    candidate_models = candidates["model_id"].dropna().astype(str).unique().tolist()
    if candidate_models != [model_id]:
        raise DataValidationError("candidate model_id does not match prediction artifact")
    source = predictions.set_index("ts_code")["prediction_score"].astype(float)
    missing = sorted(set(candidates["ts_code"]) - set(source.index))
    if missing:
        raise DataValidationError(f"candidates are absent from predictions: {missing}")
    candidate_scores = candidates.set_index("ts_code")["prediction_score"].astype(float)
    if not np.allclose(
        candidate_scores.to_numpy(),
        source.loc[candidate_scores.index].to_numpy(),
        rtol=0,
        atol=1e-12,
    ):
        raise DataValidationError("candidate scores differ from production predictions")
    return model_id


def _read_optional_as_of(
    root: Path,
    dataset: str,
    as_of: str,
    columns: tuple[str, ...],
    warnings: list[str],
) -> DataFrame:
    files = list((root / dataset).glob("**/*.parquet"))
    empty = pd.DataFrame(columns=list(columns))
    if not files:
        warnings.append(f"optional report input is missing: {dataset}")
        return empty
    glob = root / dataset / "**" / "*.parquet"
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
        warnings.append(f"optional report input is unreadable: {dataset}: {error}")
        return empty
    if frame.empty:
        warnings.append(f"optional report input has no rows for {as_of}: {dataset}")
        return empty
    frame["ts_code"] = frame["ts_code"].astype(str)
    if frame.duplicated("ts_code").any():
        warnings.append(f"optional report input has duplicate rows for {as_of}: {dataset}")
        frame = frame.drop_duplicates("ts_code", keep="last")
    return frame


def _read_optional_daily_history(
    raw_root: Path,
    as_of: str,
    codes: tuple[str, ...],
    window: int,
    warnings: list[str],
) -> DataFrame:
    columns = ["ts_code", "trade_date", "pct_chg", "amount"]
    if not codes:
        return pd.DataFrame(columns=columns)
    files = list((raw_root / "daily").glob("**/*.parquet"))
    if not files:
        warnings.append("optional report input is missing: daily")
        return pd.DataFrame(columns=columns)
    glob = raw_root / "daily" / "**" / "*.parquet"
    placeholders = ", ".join("?" for _ in codes)
    query = f"""
        SELECT CAST(ts_code AS VARCHAR) AS ts_code,
               CAST(trade_date AS VARCHAR) AS trade_date,
               CAST(pct_chg AS DOUBLE) AS pct_chg,
               CAST(amount AS DOUBLE) AS amount
        FROM read_parquet('{glob.as_posix()}', hive_partitioning=false)
        WHERE CAST(trade_date AS VARCHAR) <= ?
          AND CAST(ts_code AS VARCHAR) IN ({placeholders})
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY ts_code ORDER BY CAST(trade_date AS VARCHAR) DESC
        ) <= ?
        ORDER BY ts_code, trade_date
    """  # noqa: S608 -- parameterized values and configured local Parquet path
    try:
        with duckdb.connect() as connection:
            frame = connection.execute(query, [as_of, *codes, window]).fetch_df()
    except duckdb.Error as error:
        warnings.append(f"optional report input is unreadable: daily: {error}")
        return pd.DataFrame(columns=columns)
    if frame.empty:
        warnings.append(f"optional report input has no candidate history through {as_of}: daily")
    return frame


def _enrich_candidates(
    candidates: DataFrame,
    universe: DataFrame,
    daily_basic: DataFrame,
    daily_history: DataFrame,
    settings: DailyResearchReportSettings,
    warnings: list[str],
) -> DataFrame:
    enriched = candidates.copy()
    for frame in (universe, daily_basic):
        if not frame.empty:
            enriched = enriched.merge(frame, on="ts_code", how="left", validate="one_to_one")
    for column, default in (
        ("market", "Unknown"),
        ("industry", "Unknown"),
        ("is_limit_up", False),
        ("total_mv", np.nan),
    ):
        if column not in enriched:
            enriched[column] = default
    enriched["market"] = enriched["market"].fillna("Unknown").astype(str)
    enriched["industry"] = enriched["industry"].fillna("Unknown").astype(str)
    enriched["is_limit_up"] = enriched["is_limit_up"].fillna(False).astype(bool)
    enriched["board"] = enriched["ts_code"].map(_board_name)
    latest = daily_history.loc[
        daily_history["trade_date"]
        .astype(str)
        .eq(candidates["trade_date"].iloc[0] if not candidates.empty else "")
    ].copy()
    if not latest.empty:
        latest = latest.drop_duplicates("ts_code", keep="last")
        latest = latest.loc[:, ["ts_code", "pct_chg", "amount"]]
        enriched = enriched.merge(latest, on="ts_code", how="left", validate="one_to_one")
    else:
        enriched["pct_chg"] = np.nan
        enriched["amount"] = np.nan
    volatility = _trailing_volatility(daily_history, settings)
    enriched = enriched.merge(volatility, on="ts_code", how="left", validate="one_to_one")
    if not candidates.empty:
        missing_universe = int(enriched["market"].eq("Unknown").sum())
        missing_daily = int(enriched["pct_chg"].isna().sum())
        missing_mv = int(enriched["total_mv"].isna().sum())
        if missing_universe:
            warnings.append(f"candidate rows missing universe metadata: {missing_universe}")
        if missing_daily:
            warnings.append(f"candidate rows missing as-of daily data: {missing_daily}")
        if missing_mv:
            warnings.append(f"candidate rows missing market capitalization: {missing_mv}")
    enriched["risk_flags"] = enriched.apply(lambda row: _risk_flags(row, settings), axis=1)
    return enriched


def _trailing_volatility(
    daily_history: DataFrame, settings: DailyResearchReportSettings
) -> DataFrame:
    rows: list[dict[str, object]] = []
    if daily_history.empty:
        return pd.DataFrame(columns=["ts_code", "recent_volatility_pct", "volatility_observations"])
    for ts_code, group in daily_history.groupby("ts_code", sort=True):
        values = pd.to_numeric(group["pct_chg"], errors="coerce").dropna()
        volatility = (
            float(values.std(ddof=0))
            if len(values) >= settings.volatility_min_observations
            else np.nan
        )
        rows.append(
            {
                "ts_code": str(ts_code),
                "recent_volatility_pct": volatility,
                "volatility_observations": len(values),
            }
        )
    return pd.DataFrame(rows)


def _risk_flags(row: pd.Series, settings: DailyResearchReportSettings) -> list[str]:
    flags: list[str] = []
    pct_chg = _optional_float(row.get("pct_chg"))
    volatility = _optional_float(row.get("recent_volatility_pct"))
    amount = _optional_float(row.get("amount"))
    if pct_chg is None:
        flags.append("missing_daily_data")
    elif abs(pct_chg) >= settings.abnormal_return_abs_pct:
        flags.append("abnormal_recent_return")
    if volatility is not None and volatility >= settings.high_volatility_pct:
        flags.append("high_volatility")
    if amount is not None and amount < settings.low_liquidity_amount:
        flags.append("low_liquidity")
    if bool(row.get("is_limit_up", False)):
        flags.append("limit_up")
    if str(row.get("market", "Unknown")) == "Unknown":
        flags.append("missing_universe_metadata")
    if _optional_float(row.get("total_mv")) is None:
        flags.append("missing_market_cap")
    return flags


def _build_summary(
    as_of: str,
    model_id: str,
    prediction_count: int,
    enriched: DataFrame,
    settings: DailyResearchReportSettings,
    warnings: list[str],
) -> dict[str, Any]:
    risks = [
        {
            "rank": int(row["rank"]),
            "ts_code": str(row["ts_code"]),
            "flags": list(row["risk_flags"]),
            "pct_chg": _optional_float(row.get("pct_chg")),
            "recent_volatility_pct": _optional_float(row.get("recent_volatility_pct")),
            "amount": _optional_float(row.get("amount")),
        }
        for _, row in enriched.iterrows()
        if row["risk_flags"]
    ]
    return {
        "schema_version": 1,
        "artifact_name": "daily_quantitative_research_report",
        "as_of": as_of,
        "model_id": model_id,
        "prediction_count": prediction_count,
        "candidate_count": len(enriched),
        "top_candidate_count": min(len(enriched), settings.top_candidates),
        "statistics": {
            "board_distribution": _distribution(enriched.get("board")),
            "market_cap_distribution": _market_cap_distribution(enriched.get("total_mv")),
            "industry_distribution": _distribution(enriched.get("industry")),
        },
        "risk_flags": risks,
        "risk_thresholds": settings.model_dump(mode="json"),
        "warnings": list(dict.fromkeys(warnings)),
        "model_explanation": (
            "This report describes model ranking only. Risk flags and descriptive "
            "statistics do not modify candidate scores or ranking."
        ),
    }


def _render_markdown(
    summary: dict[str, Any], enriched: DataFrame, settings: DailyResearchReportSettings
) -> str:
    lines = [
        "# Daily Quantitative Research Report",
        "",
        "## Market Summary",
        "",
        f"- Date: {summary['as_of']}",
        f"- Candidate count: {summary['candidate_count']}",
        f"- Model ID: `{summary['model_id']}`",
        "",
        "## Candidate Ranking",
        "",
        "| Rank | TS Code | Score | Market | Industry |",
        "| ---: | --- | ---: | --- | --- |",
    ]
    for _, row in enriched.head(settings.top_candidates).iterrows():
        lines.append(
            f"| {int(row['rank'])} | `{_escape(row['ts_code'])}` | "
            f"{float(row['prediction_score']):.8f} | {_escape(row['market'])} | "
            f"{_escape(row['industry'])} |"
        )
    if enriched.empty:
        lines.append("| - | - | - | - | - |")
    lines.extend(["", "## Statistics", ""])
    statistics = summary["statistics"]
    lines.extend(_distribution_lines("Board distribution", statistics["board_distribution"]))
    lines.extend(
        _distribution_lines("Market cap distribution", statistics["market_cap_distribution"])
    )
    lines.extend(_distribution_lines("Industry distribution", statistics["industry_distribution"]))
    lines.extend(["", "## Risk Flags", ""])
    risks = summary["risk_flags"]
    if risks:
        lines.extend(
            [
                "| Rank | TS Code | Flags | Return % | Volatility % | Amount |",
                "| ---: | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for risk in risks:
            lines.append(
                f"| {risk['rank']} | `{risk['ts_code']}` | {', '.join(risk['flags'])} | "
                f"{_format_optional(risk['pct_chg'])} | "
                f"{_format_optional(risk['recent_volatility_pct'])} | "
                f"{_format_optional(risk['amount'], decimals=0)} |"
            )
    else:
        lines.append("No configured risk flags were detected.")
    lines.extend(
        [
            "",
            "## Model Explanation",
            "",
            str(summary["model_explanation"]),
            "",
            "Industry values are descriptive universe metadata and are not used to alter ranking.",
        ]
    )
    warnings = summary["warnings"]
    if warnings:
        lines.extend(["", "### Data Warnings", ""])
        lines.extend(f"- {_escape(warning)}" for warning in warnings)
    return "\n".join(lines) + "\n"


def _publish(
    report_dir: Path,
    report_path: Path,
    summary_path: Path,
    markdown: str,
    summary: dict[str, Any],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=report_dir) as temporary:
        staging = Path(temporary)
        staged_report = staging / report_path.name
        staged_report.write_text(markdown, encoding="utf-8")
        atomic_write_json(staging / summary_path.name, summary)
        os.replace(staged_report, report_path)
        os.replace(staging / summary_path.name, summary_path)


def _board_name(ts_code: object) -> str:
    code = str(ts_code).upper()
    if code.endswith(".BJ"):
        return "Beijing"
    if code.startswith("688") and code.endswith(".SH"):
        return "STAR"
    if code.startswith("300") and code.endswith(".SZ"):
        return "ChiNext"
    if code.endswith(".SH"):
        return "Shanghai Main"
    if code.endswith(".SZ"):
        return "Shenzhen Main"
    return "Other"


def _distribution(values: pd.Series | None) -> dict[str, int]:
    if values is None or values.empty:
        return {}
    normalized = values.fillna("Unknown").astype(str)
    counts = normalized.value_counts()
    return {key: int(counts[key]) for key in sorted(counts.index)}


def _market_cap_distribution(values: pd.Series | None) -> dict[str, int]:
    labels = ("below_5bn_cny", "5bn_to_10bn_cny", "10bn_to_30bn_cny", "at_least_30bn_cny")
    result = {label: 0 for label in labels}
    result["missing"] = 0
    if values is None:
        return result
    for value in pd.to_numeric(values, errors="coerce"):
        if pd.isna(value):
            result["missing"] += 1
        elif value < 500_000:
            result[labels[0]] += 1
        elif value < 1_000_000:
            result[labels[1]] += 1
        elif value < 3_000_000:
            result[labels[2]] += 1
        else:
            result[labels[3]] += 1
    return result


def _distribution_lines(title: str, values: dict[str, int]) -> list[str]:
    lines = [f"### {title}", ""]
    if not values:
        return [*lines, "- No data", ""]
    lines.extend(f"- {_escape(key)}: {value}" for key, value in values.items())
    lines.append("")
    return lines


def _optional_float(value: object) -> float | None:
    if not isinstance(value, (str, int, float, np.integer, np.floating)):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def _format_optional(value: object, *, decimals: int = 4) -> str:
    converted = _optional_float(value)
    return "-" if converted is None else f"{converted:.{decimals}f}"


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
