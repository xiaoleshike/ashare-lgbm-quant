"""Partitioned Parquet storage for point-in-time universe outputs."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError

UNIVERSE_COLUMNS: tuple[str, ...] = (
    "trade_date",
    "ts_code",
    "name",
    "market",
    "exchange",
    "industry",
    "list_date",
    "delist_date",
    "list_days",
    "is_listed",
    "is_new_stock",
    "is_st",
    "is_suspended",
    "is_low_liquidity",
    "is_limit_up",
    "is_limit_down",
    "can_buy",
    "can_sell",
    "in_base_universe",
    "in_model_universe",
    "exclude_reason",
)


type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class UniverseStatus:
    """Summarize a stored universe table or one date."""

    exists: bool
    rows: int
    partitions: int
    min_date: str | None = None
    max_date: str | None = None
    in_base_universe: int = 0
    in_model_universe: int = 0
    can_buy: int = 0
    can_sell: int = 0


class UniverseStore:
    """Read and write daily universe outputs as partitioned Parquet."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def dataset_dir(self) -> Path:
        """Return the universe dataset root directory."""

        return self.root / "universe_daily"

    def write(self, frame: DataFrame) -> int:
        """Idempotently merge and write universe rows partitioned by trade month."""

        if frame.empty:
            return 0
        self._validate_columns(frame)
        working = frame.copy()
        working["trade_date"] = working["trade_date"].astype(str)
        working["ts_code"] = working["ts_code"].astype(str)

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

    def read(self, start_date: str | None = None, end_date: str | None = None) -> DataFrame:
        """Read stored universe rows, optionally constrained by inclusive dates."""

        files = sorted(self.dataset_dir.glob("**/*.parquet"))
        if not files:
            return pd.DataFrame(columns=list(UNIVERSE_COLUMNS))
        frame = pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)
        frame["trade_date"] = frame["trade_date"].astype(str)
        if start_date is not None:
            frame = frame[frame["trade_date"] >= start_date]
        if end_date is not None:
            frame = frame[frame["trade_date"] <= end_date]
        return frame.reset_index(drop=True)

    def status(self, date: str | None = None) -> UniverseStatus:
        """Return row counts and basic membership counts."""

        frame = self.read(date, date) if date is not None else self.read()
        files = sorted(self.dataset_dir.glob("**/*.parquet"))
        if frame.empty:
            return UniverseStatus(exists=bool(files), rows=0, partitions=len(files))
        dates = frame["trade_date"].astype(str)
        return UniverseStatus(
            exists=True,
            rows=len(frame),
            partitions=len(files),
            min_date=str(dates.min()),
            max_date=str(dates.max()),
            in_base_universe=int(frame["in_base_universe"].sum()),
            in_model_universe=int(frame["in_model_universe"].sum()),
            can_buy=int(frame["can_buy"].sum()),
            can_sell=int(frame["can_sell"].sum()),
        )

    @staticmethod
    def _validate_columns(frame: DataFrame) -> None:
        missing = [column for column in UNIVERSE_COLUMNS if column not in frame.columns]
        if missing:
            raise DataValidationError(f"universe_daily is missing required columns: {missing}")
        if frame.duplicated(subset=["trade_date", "ts_code"]).any():
            duplicate_count = int(frame.duplicated(subset=["trade_date", "ts_code"]).sum())
            raise DataValidationError(
                f"universe_daily primary key is not unique; duplicate rows={duplicate_count}"
            )

    @staticmethod
    def _merge_existing(path: Path, frame: DataFrame) -> DataFrame:
        if path.exists():
            existing = pd.read_parquet(path)
            merged = pd.concat([existing, frame], ignore_index=True)
        else:
            merged = frame.copy()
        merged = merged.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
        return merged.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    @staticmethod
    def _atomic_write(path: Path, frame: DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as temp_dir:
            temp_path = Path(temp_dir) / "data.parquet"
            frame.to_parquet(temp_path, index=False)
            shutil.move(str(temp_path), path)
