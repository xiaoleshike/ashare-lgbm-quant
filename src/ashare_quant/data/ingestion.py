"""Historical and incremental Tushare ingestion orchestration."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from ashare_quant.config import AppSettings
from ashare_quant.data.datasets import DEFAULT_DATASETS, DatasetSpec, get_dataset_spec
from ashare_quant.data.exceptions import (
    DataValidationError,
    TushareNoDataError,
    TusharePermissionError,
    TushareRequestError,
)
from ashare_quant.data.storage import ParquetDataStore
from ashare_quant.data.tushare_client import TushareClient, TushareClientConfig

LOGGER = logging.getLogger(__name__)
type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Outcome for one dataset download operation."""

    dataset: str
    rows_written: int = 0
    skipped: bool = False
    message: str = ""


@dataclass(frozen=True, slots=True)
class GapReport:
    """Missing trading-date coverage for one dataset."""

    dataset: str
    start_date: str
    end_date: str
    expected_dates: int
    missing_dates: tuple[str, ...] = ()
    missing_by_entity: dict[str, tuple[str, ...]] = field(default_factory=dict)
    excluded_before_inception_by_entity: dict[str, tuple[str, ...]] = field(default_factory=dict)
    skipped: bool = False
    message: str = ""

    @property
    def has_gaps(self) -> bool:
        """Return whether this report contains missing expected coverage."""

        return bool(self.missing_dates or self.missing_by_entity)

    @property
    def excluded_before_inception(self) -> int:
        """Return expected entity-date pairs excluded before configured inception."""

        return sum(len(dates) for dates in self.excluded_before_inception_by_entity.values())


class DataIngestionService:
    """Coordinate Tushare downloads, local storage, and resumable updates."""

    def __init__(
        self,
        settings: AppSettings,
        store: ParquetDataStore | None = None,
        client: TushareClient | None = None,
    ) -> None:
        self._settings = settings
        self._store = store or ParquetDataStore(settings.paths.parquet_store)
        self._client = client

    @property
    def store(self) -> ParquetDataStore:
        """Return the configured local data store."""

        return self._store

    def init(
        self,
        dataset_names: tuple[str, ...] = DEFAULT_DATASETS,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[DownloadResult]:
        """Run full historical initialization for selected datasets."""

        start = start_date or self._settings.data.default_start_date
        end = end_date or today_yyyymmdd()
        ordered_names = self._with_calendar_first(dataset_names)
        return [
            self._download_dataset(get_dataset_spec(name), start, end) for name in ordered_names
        ]

    def update(
        self,
        dataset_names: tuple[str, ...] = DEFAULT_DATASETS,
        end_date: str | None = None,
        refresh_snapshots: bool = False,
        repair_gaps: bool = False,
    ) -> list[DownloadResult]:
        """Run incremental update from local coverage where possible."""

        end = end_date or today_yyyymmdd()
        results: list[DownloadResult] = []
        for name in self._with_calendar_first(dataset_names):
            spec = get_dataset_spec(name)
            if repair_gaps:
                results.extend(
                    self.repair_gaps((name,), self._settings.data.default_start_date, end)
                )
            if spec.partitioning == "snapshot" and self._store.status(spec).exists:
                if self._should_skip_snapshot_refresh(spec, refresh_snapshots):
                    results.append(
                        DownloadResult(spec.name, skipped=True, message="snapshot already exists")
                    )
                    continue
                results.append(
                    self._refresh_snapshot(spec, self._settings.data.default_start_date, end)
                )
                continue
            if spec.date_column is None and self._store.status(spec).exists:
                results.append(
                    DownloadResult(spec.name, skipped=True, message="snapshot already exists")
                )
                continue

            max_date = self._store.max_date(spec)
            start = self._incremental_start_date(spec, max_date, end)
            if start > end:
                results.append(
                    DownloadResult(spec.name, skipped=True, message="already up to date")
                )
                continue
            results.append(self._download_dataset(spec, start, end))
        return results

    def scan_gaps(
        self,
        dataset_names: tuple[str, ...] = DEFAULT_DATASETS,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[GapReport]:
        """Report missing expected trading dates without downloading data."""

        start = start_date or self._settings.data.default_start_date
        end = end_date or today_yyyymmdd()
        return [
            self._scan_dataset_gaps(get_dataset_spec(name), start, end) for name in dataset_names
        ]

    def repair_gaps(
        self,
        dataset_names: tuple[str, ...] = DEFAULT_DATASETS,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[DownloadResult]:
        """Download only missing trading dates detected from trade_cal coverage."""

        start = start_date or self._settings.data.default_start_date
        end = end_date or today_yyyymmdd()
        results: list[DownloadResult] = []
        for name in dataset_names:
            spec = get_dataset_spec(name)
            report = self._scan_dataset_gaps(spec, start, end)
            if report.skipped:
                results.append(DownloadResult(spec.name, skipped=True, message=report.message))
                continue
            if not report.has_gaps:
                results.append(DownloadResult(spec.name, skipped=True, message="no gaps detected"))
                continue
            rows_written = self._repair_dataset_gaps(spec, report)
            results.append(
                DownloadResult(
                    spec.name,
                    rows_written=rows_written,
                    message=f"repaired gaps={len(report.missing_dates)}",
                )
            )
        return results

    def _incremental_start_date(
        self, spec: DatasetSpec, max_date: str | None, end_date: str
    ) -> str:
        if max_date is None:
            return self._settings.data.default_start_date
        if spec.fetch_mode == "finance_period_vip":
            end = datetime.strptime(end_date, "%Y%m%d")
            lookback = end - timedelta(days=self._settings.data.finance_revision_lookback_days)
            return max(
                self._settings.data.default_start_date,
                lookback.strftime("%Y%m%d"),
            )
        if spec.fetch_mode == "month":
            return next_month_start_yyyymmdd(max_date)
        if spec.revision_lookback_open_days <= 0:
            return next_yyyymmdd(max_date)
        lookback_dates = [
            date
            for date in self._open_dates(self._settings.data.default_start_date, end_date)
            if date <= max_date
        ]
        if not lookback_dates:
            return next_yyyymmdd(max_date)
        offset = min(spec.revision_lookback_open_days, len(lookback_dates))
        return lookback_dates[-offset]

    def _download_dataset(
        self, spec: DatasetSpec, start_date: str, end_date: str
    ) -> DownloadResult:
        rows_written = 0
        try:
            if spec.fetch_mode == "finance_period_vip":
                rows_written = self._download_finance(spec, start_date, end_date)
            elif spec.fetch_mode == "ts_code_date_range":
                rows_written = self._download_batched(spec, start_date, end_date)
            else:
                for frame in self._iter_fetch_frames(spec, start_date, end_date):
                    rows_written += self._store.write(spec, frame)
            LOGGER.info(
                "dataset stored",
                extra={
                    "dataset": spec.name,
                    "rows_written": rows_written,
                    "start": start_date,
                    "end": end_date,
                },
            )
            return DownloadResult(spec.name, rows_written=rows_written)
        except TusharePermissionError as error:
            LOGGER.warning("dataset skipped for permission error", extra={"dataset": spec.name})
            return DownloadResult(spec.name, skipped=True, message=str(error))

    def _scan_dataset_gaps(self, spec: DatasetSpec, start_date: str, end_date: str) -> GapReport:
        if not self._supports_gap_detection(spec):
            return GapReport(
                spec.name,
                start_date,
                end_date,
                expected_dates=0,
                skipped=True,
                message="gap detection is not defined for this dataset fetch mode",
            )

        open_dates = self._open_dates(start_date, end_date)
        if not open_dates:
            return GapReport(
                spec.name,
                start_date,
                end_date,
                expected_dates=0,
                skipped=True,
                message="no open trading dates found in trade_cal",
            )

        frame = self._store.read_dataset(spec, start_date, end_date)
        if spec.fetch_mode == "index_codes":
            return self._scan_index_gaps(spec, frame, open_dates, start_date, end_date)

        present_dates = set()
        if not frame.empty and spec.date_column is not None and spec.date_column in frame.columns:
            present_dates = set(frame[spec.date_column].dropna().astype(str).unique())
        missing = [
            date
            for date in open_dates
            if date not in present_dates and not self._has_allowed_empty_marker(spec, date)
        ]
        return GapReport(
            spec.name,
            start_date,
            end_date,
            expected_dates=len(open_dates),
            missing_dates=tuple(missing),
        )

    def _scan_index_gaps(
        self,
        spec: DatasetSpec,
        frame: DataFrame,
        open_dates: tuple[str, ...],
        start_date: str,
        end_date: str,
    ) -> GapReport:
        missing_by_code: dict[str, tuple[str, ...]] = {}
        excluded_by_code: dict[str, tuple[str, ...]] = {}
        expected_dates = 0
        if frame.empty or not {"ts_code", "trade_date"}.issubset(frame.columns):
            present: dict[str, set[str]] = {}
        else:
            work = frame[["ts_code", "trade_date"]].copy()
            work["ts_code"] = work["ts_code"].astype(str)
            work["trade_date"] = work["trade_date"].astype(str)
            present = {
                str(code): set(group["trade_date"].astype(str))
                for code, group in work.groupby("ts_code", sort=False)
            }
        for code in self._settings.data.index_codes:
            code_present = present.get(code, set())
            first_available_date = self._settings.data.index_first_available_dates.get(code)
            excluded_dates = tuple(
                date
                for date in open_dates
                if first_available_date is not None and date < first_available_date
            )
            if excluded_dates:
                excluded_by_code[code] = excluded_dates
            eligible_dates = tuple(
                date
                for date in open_dates
                if first_available_date is None or date >= first_available_date
            )
            expected_dates += len(eligible_dates)
            missing_dates = tuple(
                date
                for date in eligible_dates
                if date not in code_present and not self._has_allowed_empty_marker(spec, date, code)
            )
            if missing_dates:
                missing_by_code[code] = missing_dates
        union_missing = tuple(
            sorted({date for dates in missing_by_code.values() for date in dates})
        )
        return GapReport(
            spec.name,
            start_date,
            end_date,
            expected_dates=expected_dates,
            missing_dates=union_missing,
            missing_by_entity=missing_by_code,
            excluded_before_inception_by_entity=excluded_by_code,
        )

    def _repair_dataset_gaps(self, spec: DatasetSpec, report: GapReport) -> int:
        if spec.fetch_mode == "index_codes":
            return self._repair_index_gaps(spec, report)
        if spec.fetch_mode != "trade_date":
            return 0
        rows_written = 0
        params = dict(spec.params)
        fields = fields_param(spec)
        if fields is not None:
            params["fields"] = fields
        for trade_date in report.missing_dates:
            frame = self._query(spec, trade_date=trade_date, **params)
            if frame.empty:
                self._handle_empty_fetch(spec, trade_date)
                continue
            rows_written += self._store.write(spec, frame)
        return rows_written

    def _repair_index_gaps(self, spec: DatasetSpec, report: GapReport) -> int:
        rows_written = 0
        params = dict(spec.params)
        fields = fields_param(spec)
        if fields is not None:
            params["fields"] = fields
        for code, missing_dates in report.missing_by_entity.items():
            for trade_date in missing_dates:
                frame = self._query(
                    spec,
                    ts_code=code,
                    start_date=trade_date,
                    end_date=trade_date,
                    **params,
                )
                if frame.empty:
                    self._handle_empty_fetch(spec, trade_date, code)
                    continue
                rows_written += self._store.write(spec, frame)
        return rows_written

    @staticmethod
    def _supports_gap_detection(spec: DatasetSpec) -> bool:
        return (
            spec.uses_trade_calendar
            and spec.date_column is not None
            and spec.partitioning == "trade_month"
            and spec.fetch_mode in {"trade_date", "index_codes"}
        )

    def _should_skip_snapshot_refresh(self, spec: DatasetSpec, refresh_snapshots: bool) -> bool:
        """Return whether an existing snapshot should be preserved for this update."""

        if refresh_snapshots:
            return False
        policy = self._settings.data.snapshot_refresh_policy
        if policy == "always":
            return False
        if policy == "manual":
            return True
        status = self._store.status(spec)
        if status.snapshot_age_days is None:
            return True
        return status.snapshot_age_days < self._settings.data.snapshot_refresh_ttl_days

    def _refresh_snapshot(
        self, spec: DatasetSpec, start_date: str, end_date: str
    ) -> DownloadResult:
        """Fetch a complete snapshot and replace the local copy only when valid and non-empty."""

        try:
            frames = [
                frame
                for frame in self._iter_fetch_frames(spec, start_date, end_date)
                if not frame.empty
            ]
            if not frames:
                return DownloadResult(
                    spec.name,
                    skipped=True,
                    message="snapshot refresh returned no rows; existing snapshot preserved",
                )
            frame = pd.concat(frames, ignore_index=True).drop_duplicates()
            rows_written = self._store.replace_snapshot(spec, frame)
            LOGGER.info(
                "snapshot refreshed",
                extra={"dataset": spec.name, "rows_written": rows_written},
            )
            return DownloadResult(
                spec.name,
                rows_written=rows_written,
                message="snapshot refreshed",
            )
        except TushareRequestError as error:
            LOGGER.warning(
                "snapshot refresh failed; existing snapshot preserved",
                extra={"dataset": spec.name},
            )
            return DownloadResult(
                spec.name,
                skipped=True,
                message=f"{error}; existing snapshot preserved",
            )

    def _download_finance(self, spec: DatasetSpec, start_date: str, end_date: str) -> int:
        if spec.vip_endpoint is None:
            raise ValueError(f"{spec.name} requires a VIP endpoint")
        rows_written = 0
        try:
            for frame in self._iter_finance_vip_frames(spec, start_date, end_date):
                if not frame.empty:
                    rows_written += self._store.write(spec, frame.drop_duplicates())
            return rows_written
        except TusharePermissionError:
            LOGGER.warning(
                "VIP finance endpoint unavailable; falling back to per-stock requests",
                extra={"dataset": spec.name, "endpoint": spec.vip_endpoint},
            )
            return self._download_finance_fallback(spec, start_date, end_date)

    def _iter_finance_vip_frames(
        self, spec: DatasetSpec, start_date: str, end_date: str
    ) -> Iterator[DataFrame]:
        if spec.vip_endpoint is None:
            return
        page_size = self._settings.data.tushare_page_size
        for period in iter_report_periods(
            start_date,
            end_date,
            self._settings.data.finance_revision_lookback_days,
        ):
            offset = 0
            while True:
                frame = self._get_client().query(
                    spec.vip_endpoint,
                    period=period,
                    limit=page_size,
                    offset=offset,
                )
                filtered = self._filter_by_date(spec, frame, start_date, end_date)
                if not filtered.empty:
                    yield filtered
                if len(frame) < page_size:
                    break
                offset += page_size

    def _download_finance_fallback(self, spec: DatasetSpec, start_date: str, end_date: str) -> int:
        rows_written = 0
        frames: list[DataFrame] = []
        for ts_code in self._stock_codes(start_date, end_date):
            try:
                frame = self._query(
                    spec,
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                )
            except TushareNoDataError:
                LOGGER.info(
                    "dataset entity skipped because Tushare returned no data",
                    extra={"dataset": spec.name, "ts_code": ts_code},
                )
                continue
            if frame.empty:
                continue
            frames.append(self._filter_by_date(spec, frame, start_date, end_date))
            if len(frames) >= 100:
                rows_written += self._store.write(
                    spec, pd.concat(frames, ignore_index=True).drop_duplicates()
                )
                frames.clear()
        if frames:
            rows_written += self._store.write(
                spec, pd.concat(frames, ignore_index=True).drop_duplicates()
            )
        return rows_written

    def _download_batched(self, spec: DatasetSpec, start_date: str, end_date: str) -> int:
        rows_written = 0
        frames: list[DataFrame] = []
        batch_size = 100
        for frame in self._iter_fetch_frames(spec, start_date, end_date):
            if frame.empty:
                continue
            frames.append(frame)
            if len(frames) >= batch_size:
                rows_written += self._store.write(spec, pd.concat(frames, ignore_index=True))
                frames.clear()
        if frames:
            rows_written += self._store.write(spec, pd.concat(frames, ignore_index=True))
        return rows_written

    def _iter_fetch_frames(
        self, spec: DatasetSpec, start_date: str, end_date: str
    ) -> Iterator[DataFrame]:
        params = dict(spec.params)
        fields = fields_param(spec)
        if fields is not None:
            params["fields"] = fields

        if spec.fetch_mode == "snapshot":
            yield self._query(spec, **params)
            return
        if spec.fetch_mode == "stock_basic_statuses":
            for status in self._settings.data.stock_list_statuses:
                yield self._query(spec, list_status=status, **params)
            return
        if spec.fetch_mode == "date_range":
            for chunk_start, chunk_end in iter_year_chunks(
                start_date, end_date, self._settings.data.date_range_chunk_years
            ):
                yield self._query(spec, start_date=chunk_start, end_date=chunk_end, **params)
            return
        if spec.fetch_mode == "ts_code_date_range":
            for ts_code in self._stock_codes(start_date, end_date):
                yield from self._iter_ts_code_date_range_frames(
                    spec, ts_code, start_date, end_date, params
                )
            return
        if spec.fetch_mode == "finance_period_vip":
            yield from self._iter_finance_vip_frames(spec, start_date, end_date)
            return
        if spec.fetch_mode == "month":
            for month in iter_months(start_date, end_date):
                yield self._query(spec, month=month, **params)
            return
        if spec.fetch_mode == "ths_members_snapshot":
            concept_codes = self._concept_codes(start_date, end_date)
            for ts_code in concept_codes:
                try:
                    yield self._query(spec, ts_code=ts_code, **params)
                except TushareNoDataError:
                    LOGGER.info(
                        "dataset entity skipped because Tushare returned no data",
                        extra={"dataset": spec.name, "ts_code": ts_code},
                    )
            return
        if spec.fetch_mode == "trade_date":
            for trade_date in self._open_dates(start_date, end_date):
                frame = self._query(spec, trade_date=trade_date, **params)
                if frame.empty:
                    self._handle_empty_fetch(spec, trade_date)
                    continue
                yield frame
            return
        if spec.fetch_mode == "week_end_trade_date":
            for trade_date in self._period_end_open_dates(start_date, end_date, "week"):
                yield self._query(spec, trade_date=trade_date, **params)
            return
        if spec.fetch_mode == "month_end_trade_date":
            for trade_date in self._period_end_open_dates(start_date, end_date, "month"):
                yield self._query(spec, trade_date=trade_date, **params)
            return
        if spec.fetch_mode == "index_codes":
            for code in self._settings.data.index_codes:
                first_available_date = self._settings.data.index_first_available_dates.get(code)
                entity_start = max(start_date, first_available_date or start_date)
                if entity_start > end_date:
                    continue
                for chunk_start, chunk_end in iter_year_chunks(
                    entity_start, end_date, self._settings.data.date_range_chunk_years
                ):
                    frame = self._query(
                        spec,
                        ts_code=code,
                        start_date=chunk_start,
                        end_date=chunk_end,
                        **params,
                    )
                    if frame.empty:
                        for trade_date in self._open_dates(chunk_start, chunk_end):
                            self._handle_empty_fetch(spec, trade_date, code)
                        continue
                    yield frame
            return
        if spec.fetch_mode == "fund_market_snapshot":
            for market in self._settings.data.fund_markets:
                yield self._query(spec, market=market, **params)
            return
        if spec.fetch_mode == "hs_const_snapshot":
            for hs_type in self._settings.data.hs_types:
                yield self._query(spec, hs_type=hs_type, **params)
            return
        raise ValueError(f"Unsupported fetch mode: {spec.fetch_mode}")

    def _iter_ts_code_date_range_frames(
        self,
        spec: DatasetSpec,
        ts_code: str,
        start_date: str,
        end_date: str,
        params: dict[str, object],
    ) -> Iterator[DataFrame]:
        try:
            frame = self._query(
                spec,
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                **params,
            )
        except TushareNoDataError:
            LOGGER.info(
                "dataset entity range skipped because Tushare returned no data",
                extra={
                    "dataset": spec.name,
                    "ts_code": ts_code,
                    "start": start_date,
                    "end": end_date,
                },
            )
            return

        row_limit = spec.response_row_limit
        if row_limit is None or len(frame) < row_limit:
            filtered = self._filter_by_date(spec, frame, start_date, end_date)
            if not filtered.empty:
                yield filtered
            return

        open_dates = self._open_dates(start_date, end_date)
        if len(open_dates) <= 1:
            raise DataValidationError(
                f"{spec.name} returned {len(frame)} rows at its {row_limit}-row limit "
                f"for {ts_code} on {start_date}..{end_date}; completeness cannot be guaranteed"
            )
        split_index = len(open_dates) // 2
        left_end = open_dates[split_index - 1]
        right_start = open_dates[split_index]
        LOGGER.warning(
            "tushare response reached row limit; splitting date range",
            extra={
                "dataset": spec.name,
                "ts_code": ts_code,
                "start": start_date,
                "end": end_date,
                "rows": len(frame),
                "split_left_end": left_end,
                "split_right_start": right_start,
            },
        )
        yield from self._iter_ts_code_date_range_frames(spec, ts_code, start_date, left_end, params)
        yield from self._iter_ts_code_date_range_frames(
            spec, ts_code, right_start, end_date, params
        )

    def _query(self, spec: DatasetSpec, **params: object) -> DataFrame:
        return self._get_client().query(spec.endpoint, **params)

    @staticmethod
    def _filter_by_date(
        spec: DatasetSpec, frame: DataFrame, start_date: str, end_date: str
    ) -> DataFrame:
        if frame.empty or spec.date_column is None or spec.date_column not in frame.columns:
            return frame
        dates = frame[spec.date_column].astype(str)
        return frame.loc[(dates >= start_date) & (dates <= end_date)]

    def _handle_empty_fetch(
        self,
        spec: DatasetSpec,
        trade_date: str,
        entity: str | None = None,
    ) -> None:
        """Record or warn about an empty response for an expected trading date."""

        if spec.allow_empty_trading_days:
            self._store.mark_empty_result(spec, trade_date, entity)
            LOGGER.info(
                "empty dataset response recorded as complete",
                extra={"dataset": spec.name, "trade_date": trade_date, "entity": entity},
            )
            return
        LOGGER.warning(
            "empty dataset response leaves a repairable gap",
            extra={"dataset": spec.name, "trade_date": trade_date, "entity": entity},
        )

    def _has_allowed_empty_marker(
        self,
        spec: DatasetSpec,
        trade_date: str,
        entity: str | None = None,
    ) -> bool:
        return spec.allow_empty_trading_days and self._store.has_empty_result(
            spec, trade_date, entity
        )

    def _stock_codes(self, start_date: str, end_date: str) -> tuple[str, ...]:
        stock_spec = get_dataset_spec("stock_basic")
        stock_basic = self._store.read_dataset(stock_spec)
        if stock_basic.empty:
            for frame in self._iter_fetch_frames(stock_spec, start_date, end_date):
                self._store.write(stock_spec, frame)
            stock_basic = self._store.read_dataset(stock_spec)
        if stock_basic.empty or "ts_code" not in stock_basic.columns:
            return ()
        return tuple(stock_basic["ts_code"].dropna().astype(str).sort_values().unique())

    def _open_dates(self, start_date: str, end_date: str) -> tuple[str, ...]:
        calendar_spec = get_dataset_spec("trade_cal")
        calendar = self._store.read_dataset(calendar_spec)
        if calendar.empty:
            for frame in self._iter_fetch_frames(calendar_spec, start_date, end_date):
                self._store.write(calendar_spec, frame)
            calendar = self._store.read_dataset(calendar_spec)
        open_days = calendar.loc[
            (calendar["is_open"].astype(int) == 1)
            & (calendar["cal_date"].astype(str) >= start_date)
            & (calendar["cal_date"].astype(str) <= end_date),
            "cal_date",
        ].astype(str)
        return tuple(open_days.sort_values().unique())

    def _concept_codes(self, start_date: str, end_date: str) -> tuple[str, ...]:
        concept_spec = get_dataset_spec("concept")
        concepts = self._store.read_dataset(concept_spec)
        if concepts.empty:
            for frame in self._iter_fetch_frames(concept_spec, start_date, end_date):
                self._store.write(concept_spec, frame)
            concepts = self._store.read_dataset(concept_spec)
        if concepts.empty or "ts_code" not in concepts.columns:
            return ()
        return tuple(concepts["ts_code"].dropna().astype(str).sort_values().unique())

    def _period_end_open_dates(
        self, start_date: str, end_date: str, period: str
    ) -> tuple[str, ...]:
        open_dates = self._open_dates(start_date, end_date)
        if not open_dates:
            return ()
        frame = pd.DataFrame({"trade_date": list(open_dates)})
        dates = pd.to_datetime(frame["trade_date"], format="%Y%m%d")
        if period == "week":
            frame["period"] = dates.dt.strftime("%G%V")
        elif period == "month":
            frame["period"] = dates.dt.strftime("%Y%m")
        else:
            raise ValueError(f"Unsupported period: {period}")
        return tuple(frame.groupby("period", sort=True)["trade_date"].max().astype(str))

    def _get_client(self) -> TushareClient:
        if self._client is None:
            self._client = TushareClient(
                token=self._settings.tushare_token,
                config=TushareClientConfig(
                    retry_attempts=self._settings.data.retry_attempts,
                    rate_limit_per_minute=self._settings.data.rate_limit_per_minute,
                    request_interval_seconds=self._settings.data.request_interval_seconds,
                    backoff_base_seconds=self._settings.data.backoff_base_seconds,
                    backoff_max_seconds=self._settings.data.backoff_max_seconds,
                    endpoint_rate_limits_per_minute=dict(
                        self._settings.data.endpoint_rate_limits_per_minute
                    ),
                ),
            )
        return self._client

    @staticmethod
    def _with_calendar_first(dataset_names: tuple[str, ...]) -> tuple[str, ...]:
        names = list(dict.fromkeys(dataset_names))
        needs_calendar = any(get_dataset_spec(name).uses_trade_calendar for name in names)
        if needs_calendar and "trade_cal" not in names:
            names.insert(0, "trade_cal")
        if "trade_cal" in names:
            names.insert(0, names.pop(names.index("trade_cal")))
        return tuple(names)


def fields_param(spec: DatasetSpec) -> str | None:
    """Return a comma-delimited Tushare fields argument."""

    return ",".join(spec.fields) if spec.fields else None


def today_yyyymmdd() -> str:
    """Return today's date in Tushare YYYYMMDD format."""

    return datetime.now().strftime("%Y%m%d")


def next_yyyymmdd(date_value: str | None) -> str:
    """Return the calendar day after a YYYYMMDD date string."""

    if date_value is None:
        raise ValueError("date_value is required")
    parsed = datetime.strptime(date_value, "%Y%m%d")
    return (parsed + timedelta(days=1)).strftime("%Y%m%d")


def next_month_start_yyyymmdd(month_value: str | None) -> str:
    """Return the first calendar day after a YYYYMM month string."""

    if month_value is None:
        raise ValueError("month_value is required")
    parsed = datetime.strptime(f"{month_value[:6]}01", "%Y%m%d")
    next_month = (parsed.replace(day=28) + timedelta(days=4)).replace(day=1)
    return next_month.strftime("%Y%m%d")


def iter_months(start_date: str, end_date: str) -> Iterator[str]:
    """Yield inclusive YYYYMM values intersecting a YYYYMMDD date range."""

    current = datetime.strptime(start_date[:6] + "01", "%Y%m%d")
    end = datetime.strptime(end_date[:6] + "01", "%Y%m%d")
    while current <= end:
        yield current.strftime("%Y%m")
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)


def iter_year_chunks(
    start_date: str, end_date: str, years_per_chunk: int
) -> Iterator[tuple[str, str]]:
    """Yield inclusive YYYYMMDD date chunks, bounded by calendar years."""

    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    current = start
    while current <= end:
        chunk_end = min(
            current.replace(year=current.year + years_per_chunk) - timedelta(days=1), end
        )
        yield current.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")
        current = chunk_end + timedelta(days=1)


def iter_report_periods(start_date: str, end_date: str, report_lookback_days: int) -> Iterator[str]:
    """Yield quarter-end report periods that may be announced in a date window."""

    start = datetime.strptime(start_date, "%Y%m%d") - timedelta(days=report_lookback_days)
    end = datetime.strptime(end_date, "%Y%m%d")
    for year in range(start.year, end.year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            period = datetime(year, month, day)
            if start <= period <= end:
                yield period.strftime("%Y%m%d")


def build_store(root: str | Path | None, settings: AppSettings) -> ParquetDataStore:
    """Build a Parquet store from a CLI override or configured path."""

    return ParquetDataStore(Path(root) if root else settings.paths.parquet_store)
