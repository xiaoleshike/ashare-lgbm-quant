"""Partitioned Parquet storage for executable forward-return labels."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError

LABEL_COLUMNS: tuple[str, ...] = (
    "trade_date",
    "ts_code",
    "horizon",
    "entry_date",
    "exit_date",
    "entry_price",
    "exit_price",
    "stock_forward_ret",
    "benchmark_forward_ret",
    "future_excess_ret",
    "future_rank_pct",
    "future_quantile",
    "is_label_available",
    "label_unavailable_reason",
)

type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class LabelStatus:
    """Summarize stored labels globally or for one date/horizon."""

    exists: bool
    rows: int
    partitions: int
    min_date: str | None = None
    max_date: str | None = None
    available: int = 0
    unavailable: int = 0


class LabelStore:
    """Read and write forward-return labels as partitioned Parquet."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def dataset_dir(self) -> Path:
        """Return the labels dataset root directory."""

        return self.root / "labels_forward"

    def write(self, frame: DataFrame) -> int:
        """Idempotently merge and write label rows partitioned by trade month."""

        if frame.empty:
            return 0
        self._validate_columns(frame)
        working = frame.copy()
        working["trade_date"] = working["trade_date"].astype(str)
        working["ts_code"] = working["ts_code"].astype(str)
        working["horizon"] = working["horizon"].astype(int)

        rows_written = 0
        dates = working["trade_date"].astype(str)
        for year_month, partition_frame in working.groupby(dates.str.slice(0, 6), sort=True):
            year = str(year_month)[:4]
            month = str(year_month)[4:6]
            path = self.dataset_dir / f"year={year}" / f"month={month}" / "data.parquet"
            merged = self._merge_existing(path, partition_frame.copy())
            self._atomic_write(path, merged)
            rows_written += len(partition_frame)
        return rows_written

    def read(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        horizon: int | None = None,
    ) -> DataFrame:
        """Read stored label rows, optionally constrained by date range and horizon."""

        files = sorted(self.dataset_dir.glob("**/*.parquet"))
        if not files:
            return pd.DataFrame(columns=list(LABEL_COLUMNS))
        frame = pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)
        frame["trade_date"] = frame["trade_date"].astype(str)
        if start_date is not None:
            frame = frame[frame["trade_date"] >= start_date]
        if end_date is not None:
            frame = frame[frame["trade_date"] <= end_date]
        if horizon is not None:
            frame = frame[frame["horizon"].astype(int) == horizon]
        return frame.reset_index(drop=True)

    def status(self, date: str | None = None, horizon: int | None = None) -> LabelStatus:
        """Return row counts and availability counts."""

        frame = self.read(date, date, horizon) if date is not None else self.read(horizon=horizon)
        files = sorted(self.dataset_dir.glob("**/*.parquet"))
        if frame.empty:
            return LabelStatus(exists=bool(files), rows=0, partitions=len(files))
        dates = frame["trade_date"].astype(str)
        available = frame["is_label_available"].astype(bool)
        return LabelStatus(
            exists=True,
            rows=len(frame),
            partitions=len(files),
            min_date=str(dates.min()),
            max_date=str(dates.max()),
            available=int(available.sum()),
            unavailable=int((~available).sum()),
        )

    @staticmethod
    def _validate_columns(frame: DataFrame) -> None:
        missing = [column for column in LABEL_COLUMNS if column not in frame.columns]
        if missing:
            raise DataValidationError(f"labels_forward is missing required columns: {missing}")
        if frame.duplicated(subset=["trade_date", "ts_code", "horizon"]).any():
            duplicate_count = int(
                frame.duplicated(subset=["trade_date", "ts_code", "horizon"]).sum()
            )
            raise DataValidationError(
                f"labels_forward primary key is not unique; duplicate rows={duplicate_count}"
            )

    @staticmethod
    def _merge_existing(path: Path, frame: DataFrame) -> DataFrame:
        if path.exists():
            existing = pd.read_parquet(path)
            merged = pd.concat([existing, frame], ignore_index=True)
        else:
            merged = frame.copy()
        merged = merged.drop_duplicates(subset=["trade_date", "ts_code", "horizon"], keep="last")
        return merged.sort_values(["trade_date", "horizon", "ts_code"]).reset_index(drop=True)

    @staticmethod
    def _atomic_write(path: Path, frame: DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as temp_dir:
            temp_path = Path(temp_dir) / "data.parquet"
            frame.to_parquet(temp_path, index=False)
            shutil.move(str(temp_path), path)
