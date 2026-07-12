"""Partitioned Parquet storage for canonical raw Tushare datasets."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ashare_quant.data.datasets import DATASET_SPECS, DatasetSpec
from ashare_quant.data.exceptions import DataValidationError

type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class DatasetStatus:
    """Summarize locally stored dataset state."""

    name: str
    exists: bool
    rows: int
    partitions: int
    min_date: str | None = None
    max_date: str | None = None
    snapshot_updated_at: str | None = None
    snapshot_age_days: int | None = None


class ParquetDataStore:
    """Read and write canonical raw datasets as partitioned Parquet."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def dataset_dir(self, spec: DatasetSpec) -> Path:
        """Return the root directory for one dataset."""

        return self.root / spec.name

    def write(self, spec: DatasetSpec, frame: DataFrame) -> int:
        """Idempotently merge and write a frame into its partition files."""

        if frame.empty:
            return 0
        self._validate_frame(spec, frame)

        normalized = self._normalize_dates(spec, frame.copy())
        rows_written = 0
        for partition_path, partition_frame in self._iter_partitions(spec, normalized):
            merged = self._merge_existing(spec, partition_path, partition_frame)
            self._atomic_write(partition_path, merged)
            rows_written += len(partition_frame)
        return rows_written

    def replace_snapshot(self, spec: DatasetSpec, frame: DataFrame) -> int:
        """Atomically replace a snapshot dataset with a complete non-empty frame."""

        if spec.partitioning != "snapshot":
            raise DataValidationError(f"{spec.name} is not a snapshot dataset")
        if frame.empty:
            raise DataValidationError(f"{spec.name} snapshot refresh returned no rows")
        self._validate_frame(spec, frame)
        path = self.dataset_dir(spec) / "snapshot=latest" / "data.parquet"
        normalized = self._normalize_dates(spec, frame.copy())
        merged = normalized.drop_duplicates(subset=list(spec.primary_key), keep="last")
        sort_columns = [column for column in (*spec.primary_key, spec.date_column or "") if column]
        merged = merged.sort_values(sort_columns).reset_index(drop=True)
        self._atomic_write(path, merged)
        return len(merged)

    def read_dataset(
        self,
        spec: DatasetSpec,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> DataFrame:
        """Read partition files for a dataset over an optional inclusive date range."""

        files = self._dataset_files_for_range(spec, start_date, end_date)
        if not files:
            return pd.DataFrame(columns=list(spec.required_columns))
        frame = pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)
        if spec.date_column is not None and spec.date_column in frame.columns:
            frame[spec.date_column] = frame[spec.date_column].astype(str)
            if start_date is not None:
                frame = frame[frame[spec.date_column] >= start_date]
            if end_date is not None:
                frame = frame[frame[spec.date_column] <= end_date]
        return frame.reset_index(drop=True)

    def status(self, spec: DatasetSpec) -> DatasetStatus:
        """Return row, partition, and date coverage summary."""

        files = sorted(self.dataset_dir(spec).glob("**/*.parquet"))
        if not files:
            return DatasetStatus(name=spec.name, exists=False, rows=0, partitions=0)
        frame = self.read_dataset(spec)
        min_date: str | None = None
        max_date: str | None = None
        snapshot_updated_at: str | None = None
        snapshot_age_days: int | None = None
        if spec.date_column is not None and spec.date_column in frame.columns and not frame.empty:
            dates = frame[spec.date_column].astype(str)
            min_date = str(dates.min())
            max_date = str(dates.max())
        if spec.partitioning == "snapshot":
            latest_mtime = max(file.stat().st_mtime for file in files)
            updated_at = datetime.fromtimestamp(latest_mtime, tz=UTC)
            snapshot_updated_at = updated_at.isoformat(timespec="seconds")
            snapshot_age_days = max(0, (datetime.now(UTC) - updated_at).days)
        return DatasetStatus(
            name=spec.name,
            exists=True,
            rows=len(frame),
            partitions=len(files),
            min_date=min_date,
            max_date=max_date,
            snapshot_updated_at=snapshot_updated_at,
            snapshot_age_days=snapshot_age_days,
        )

    def all_statuses(self) -> list[DatasetStatus]:
        """Return status for all configured datasets."""

        return [self.status(spec) for spec in DATASET_SPECS.values()]

    def max_date(self, spec: DatasetSpec) -> str | None:
        """Return the maximum stored date for an incremental dataset."""

        return self.status(spec).max_date

    def _validate_frame(self, spec: DatasetSpec, frame: DataFrame) -> None:
        missing = [column for column in spec.required_columns if column not in frame.columns]
        if missing:
            raise DataValidationError(f"{spec.name} is missing required columns: {missing}")
        missing_pk = [column for column in spec.primary_key if column not in frame.columns]
        if missing_pk:
            raise DataValidationError(f"{spec.name} is missing primary-key columns: {missing_pk}")

    def _normalize_dates(self, spec: DatasetSpec, frame: DataFrame) -> DataFrame:
        if spec.date_column is not None and spec.date_column in frame.columns:
            frame[spec.date_column] = frame[spec.date_column].astype(str)
        return frame

    def _iter_partitions(self, spec: DatasetSpec, frame: DataFrame) -> list[tuple[Path, DataFrame]]:
        if spec.partitioning == "snapshot":
            return [(self.dataset_dir(spec) / "snapshot=latest" / "data.parquet", frame)]

        if spec.date_column is None:
            raise DataValidationError(
                f"{spec.name} requires a date column for trade_month partitioning"
            )

        partitions: list[tuple[Path, DataFrame]] = []
        dates = frame[spec.date_column].astype(str)
        for year_month, partition_frame in frame.groupby(dates.str.slice(0, 6), sort=True):
            year = str(year_month)[:4]
            month = str(year_month)[4:6]
            path = self.dataset_dir(spec) / f"year={year}" / f"month={month}" / "data.parquet"
            partitions.append((path, partition_frame.copy()))
        return partitions

    def _dataset_files_for_range(
        self,
        spec: DatasetSpec,
        start_date: str | None,
        end_date: str | None,
    ) -> list[Path]:
        if spec.partitioning == "snapshot" or spec.date_column is None:
            return sorted(self.dataset_dir(spec).glob("**/*.parquet"))
        if start_date is None and end_date is None:
            return sorted(self.dataset_dir(spec).glob("**/*.parquet"))

        months = months_in_range(start_date, end_date)
        files: list[Path] = []
        for year_month in months:
            path = (
                self.dataset_dir(spec)
                / f"year={year_month[:4]}"
                / f"month={year_month[4:6]}"
                / "data.parquet"
            )
            if path.exists():
                files.append(path)
        return sorted(files)

    def _merge_existing(self, spec: DatasetSpec, path: Path, frame: DataFrame) -> DataFrame:
        if path.exists():
            existing = pd.read_parquet(path)
            merged = pd.concat([existing, frame], ignore_index=True)
        else:
            merged = frame.copy()
        merged = merged.drop_duplicates(subset=list(spec.primary_key), keep="last")
        sort_columns = [column for column in (*spec.primary_key, spec.date_column or "") if column]
        return merged.sort_values(sort_columns).reset_index(drop=True)

    @staticmethod
    def _atomic_write(path: Path, frame: DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as temp_dir:
            temp_path = Path(temp_dir) / "data.parquet"
            frame.to_parquet(temp_path, index=False)
            shutil.move(str(temp_path), path)


def months_in_range(start_date: str | None, end_date: str | None) -> list[str]:
    """Return YYYYMM partitions touched by an inclusive date range."""

    if start_date is None and end_date is None:
        return []
    start = pd.Period((start_date or end_date or "")[:6], freq="M")
    end = pd.Period((end_date or start_date or "")[:6], freq="M")
    if start > end:
        return []
    return [period.strftime("%Y%m") for period in pd.period_range(start, end, freq="M")]
