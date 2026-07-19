"""Read-only validation for persisted production feature matrices."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from ashare_quant.features.registry import FEATURE_REGISTRY
from ashare_quant.features.storage import FeatureStore


@dataclass(frozen=True, slots=True)
class FeatureValidationResult:
    """Validation outcome for one feature date range."""

    ok: bool
    rows: int
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class FeatureValidator:
    """Validate feature schema, keys, and finite values without changing data."""

    def __init__(self, store: FeatureStore) -> None:
        self._store = store

    def validate(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> FeatureValidationResult:
        """Validate an optional inclusive date range using DuckDB projection pushdown."""

        files = sorted(self._store.dataset_dir.glob("**/*.parquet"))
        if not files:
            return FeatureValidationResult(
                ok=False,
                rows=0,
                errors=("required features_daily dataset is missing",),
            )
        if start_date is not None and end_date is not None and start_date > end_date:
            return FeatureValidationResult(
                ok=False,
                rows=0,
                errors=("feature validation start_date must be <= end_date",),
            )

        glob = self._store.dataset_dir / "**" / "*.parquet"
        source = f"read_parquet('{glob.as_posix()}', union_by_name=true, hive_partitioning=false)"
        where, parameters = _date_filter(start_date, end_date)
        with duckdb.connect() as connection:
            schema_rows = connection.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()  # noqa: S608
            columns = {str(row[0]) for row in schema_rows}
            required = {"trade_date", "ts_code", *(spec.name for spec in FEATURE_REGISTRY)}
            missing = sorted(required - columns)
            if missing:
                return FeatureValidationResult(
                    ok=False,
                    rows=0,
                    errors=(f"features_daily is missing registered columns: {missing}",),
                )

            summary_query = f"""
                SELECT
                    COUNT(*) AS rows,
                    COUNT(*) - COUNT(DISTINCT (
                        CAST(trade_date AS VARCHAR), CAST(ts_code AS VARCHAR)
                    )) AS duplicate_rows
                FROM {source}
                {where}
            """  # noqa: S608 -- local configured Parquet path
            summary = connection.execute(summary_query, parameters).fetchone()
            if summary is None:
                return FeatureValidationResult(False, 0, errors=("feature query failed",))
            row_count = int(summary[0])
            duplicate_rows = int(summary[1])
            errors: list[str] = []
            if row_count == 0:
                errors.append("required features_daily date range is empty")
            if duplicate_rows:
                errors.append(
                    f"features_daily primary key is not unique; duplicate rows={duplicate_rows}"
                )

            if row_count:
                invalid_columns = _infinite_feature_columns(
                    connection,
                    source,
                    where,
                    parameters,
                )
                if invalid_columns:
                    errors.append(
                        f"features_daily contains infinite values in columns: {invalid_columns}"
                    )
        return FeatureValidationResult(
            ok=not errors,
            rows=row_count,
            errors=tuple(errors),
        )


def _date_filter(start_date: str | None, end_date: str | None) -> tuple[str, list[str]]:
    conditions: list[str] = []
    parameters: list[str] = []
    if start_date is not None:
        conditions.append("CAST(trade_date AS VARCHAR) >= ?")
        parameters.append(start_date)
    if end_date is not None:
        conditions.append("CAST(trade_date AS VARCHAR) <= ?")
        parameters.append(end_date)
    return (f"WHERE {' AND '.join(conditions)}" if conditions else "", parameters)


def _infinite_feature_columns(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    where: str,
    parameters: list[str],
) -> list[str]:
    feature_names = [spec.name for spec in FEATURE_REGISTRY]
    expressions = ", ".join(
        f'COUNT(*) FILTER (WHERE isinf(CAST("{name}" AS DOUBLE))) AS "{name}"'
        for name in feature_names
    )
    row = connection.execute(
        f"SELECT {expressions} FROM {source} {where}",  # noqa: S608
        parameters,
    ).fetchone()
    if row is None:
        return []
    return [name for name, count in zip(feature_names, row, strict=True) if int(count) > 0]
