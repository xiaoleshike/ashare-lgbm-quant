"""Versioned security identity canonicalization at raw-to-research boundaries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ashare_quant.data.datasets import get_dataset_spec
from ashare_quant.data.exceptions import DataValidationError

type DataFrame = pd.DataFrame

SECURITY_IDENTITY_DATASETS = frozenset(
    {
        "stock_basic",
        "daily",
        "adj_factor",
        "daily_basic",
        "suspend_d",
        "stk_limit",
        "namechange",
        "income",
        "balancesheet",
        "cashflow",
        "fina_indicator",
        "forecast",
        "express",
    }
)

_COLLISION_KEY_OVERRIDES: dict[str, tuple[str, ...]] = {
    # S and R on one canonical security/date are contradictory lifecycle evidence.
    "suspend_d": ("ts_code", "trade_date"),
}


@dataclass(frozen=True, slots=True)
class SecurityAlias:
    """One explicit, optionally effective-dated source-code alias."""

    source_code: str
    canonical_code: str
    exchange: str
    effective_from: str | None
    effective_to: str | None
    mapping_source: str

    def applies(self, as_of_date: str | None) -> bool:
        """Return whether this alias applies at the supplied point in time."""

        if as_of_date is None:
            return self.effective_from is None and self.effective_to is None
        return (self.effective_from is None or self.effective_from <= as_of_date) and (
            self.effective_to is None or as_of_date <= self.effective_to
        )


@dataclass(frozen=True, slots=True)
class SecurityIdentityScanResult:
    """Read-only cross-source alias consistency summary."""

    mapping_version: str
    mapping_hash: str
    configured_aliases: int
    observed_alias_rows: int
    observed_aliases: int
    unresolved_mismatches: tuple[str, ...]
    same_source_multi_event_keys: int = 0
    collisions: int = 0

    @property
    def ok(self) -> bool:
        """Return whether all observed explicit aliases resolve across sources."""

        return not self.unresolved_mismatches and self.collisions == 0


class SecurityIdentityResolver:
    """Resolve explicit source aliases to stable canonical security identities."""

    def __init__(
        self,
        *,
        mapping_version: str,
        mapping_hash: str,
        aliases: tuple[SecurityAlias, ...],
        mapping_path: Path | None = None,
    ) -> None:
        self.mapping_version = mapping_version
        self.mapping_hash = mapping_hash
        self.mapping_path = mapping_path
        self._aliases = aliases
        self._by_source: dict[str, tuple[SecurityAlias, ...]] = {}
        for source_code in sorted({alias.source_code for alias in aliases}):
            self._by_source[source_code] = tuple(
                alias for alias in aliases if alias.source_code == source_code
            )
        self._validate()

    @classmethod
    def from_path(cls, path: Path) -> SecurityIdentityResolver:
        """Load and validate one immutable JSON mapping artifact."""

        if not path.is_file():
            raise DataValidationError(
                f"SECURITY_IDENTITY_MAPPING_INVALID: mapping file does not exist: {path}"
            )
        raw = path.read_bytes()
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DataValidationError(
                f"SECURITY_IDENTITY_MAPPING_INVALID: cannot parse {path}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise DataValidationError("SECURITY_IDENTITY_MAPPING_INVALID: root must be an object")
        if payload.get("schema_version") != 1:
            raise DataValidationError(
                "SECURITY_IDENTITY_MAPPING_INVALID: unsupported schema_version"
            )
        if payload.get("artifact_name") != "security_identity_mapping":
            raise DataValidationError("SECURITY_IDENTITY_MAPPING_INVALID: unexpected artifact_name")
        version = str(payload.get("mapping_version", "")).strip()
        rows = payload.get("aliases")
        if not version or not isinstance(rows, list):
            raise DataValidationError(
                "SECURITY_IDENTITY_MAPPING_INVALID: mapping_version and aliases are required"
            )
        aliases = tuple(_parse_alias(row) for row in rows)
        return cls(
            mapping_version=version,
            mapping_hash=hashlib.sha256(raw).hexdigest(),
            aliases=aliases,
            mapping_path=path,
        )

    @classmethod
    def empty(cls) -> SecurityIdentityResolver:
        """Return an explicit empty resolver for isolated callers and fixtures."""

        return cls(
            mapping_version="none",
            mapping_hash=hashlib.sha256(b"security-identity:none").hexdigest(),
            aliases=(),
        )

    @property
    def alias_count(self) -> int:
        """Return the number of explicit alias records."""

        return len(self._aliases)

    @property
    def source_codes(self) -> frozenset[str]:
        """Return explicitly mapped source aliases."""

        return frozenset(self._by_source)

    def provenance(self) -> dict[str, str | int]:
        """Return stable mapping provenance for processed manifests."""

        return {
            "security_identity_mapping_version": self.mapping_version,
            "security_identity_mapping_hash": self.mapping_hash,
            "security_identity_alias_count": self.alias_count,
        }

    def canonicalize(self, ts_code: str, *, as_of_date: str | None = None) -> str:
        """Resolve a code without guessing unknown aliases."""

        code = str(ts_code).strip().upper()
        matches = [alias for alias in self._by_source.get(code, ()) if alias.applies(as_of_date)]
        if not matches:
            return code
        canonical_codes = {alias.canonical_code for alias in matches}
        if len(canonical_codes) != 1:
            raise DataValidationError(
                "SECURITY_IDENTITY_UNRESOLVED: multiple mappings apply "
                f"source_code={code} as_of_date={as_of_date}"
            )
        return next(iter(canonical_codes))

    def canonicalize_frame(self, frame: DataFrame, dataset_name: str) -> DataFrame:
        """Canonicalize a copied dataset frame and validate alias collisions."""

        if frame.empty or "ts_code" not in frame.columns:
            return frame.copy()
        if dataset_name not in SECURITY_IDENTITY_DATASETS:
            raise DataValidationError(
                f"SECURITY_IDENTITY_MAPPING_INVALID: unsupported dataset={dataset_name}"
            )
        working = frame.copy()
        working["source_ts_code"] = working["ts_code"].astype(str).str.strip().str.upper()
        date_column = get_dataset_spec(dataset_name).date_column
        if date_column is not None and date_column in working.columns:
            dates = working[date_column].map(_optional_date)
            working["ts_code"] = [
                self.canonicalize(code, as_of_date=date)
                for code, date in zip(working["source_ts_code"], dates, strict=True)
            ]
        else:
            working["ts_code"] = working["source_ts_code"].map(self.canonicalize)
        return self._deduplicate_or_fail(working, dataset_name)

    def _deduplicate_or_fail(self, frame: DataFrame, dataset_name: str) -> DataFrame:
        spec = get_dataset_spec(dataset_name)
        keys = _COLLISION_KEY_OVERRIDES.get(dataset_name, spec.primary_key)
        available_keys = tuple(column for column in keys if column in frame.columns)
        if "ts_code" not in available_keys:
            return frame
        duplicate_mask = frame.duplicated(subset=list(available_keys), keep=False)
        if not duplicate_mask.any():
            return frame
        drop_indexes: list[int] = []
        semantic_columns = [
            column for column in frame.columns if column not in {*available_keys, "source_ts_code"}
        ]
        for key_values, group in frame.loc[duplicate_mask].groupby(
            list(available_keys), dropna=False, sort=True
        ):
            source_codes = sorted(group["source_ts_code"].astype(str).unique())
            if len(source_codes) == 1:
                # Multiple provider events for one original code are dataset semantics,
                # not an alias collision. Downstream dataset normalization owns them.
                continue
            if dataset_name == "namechange":
                canonical_source = group[
                    group["source_ts_code"].astype(str) == group["ts_code"].astype(str)
                ]
                if not canonical_source.empty:
                    drop_indexes.extend(group.index.difference(canonical_source.index).tolist())
                    continue
            if not _source_groups_semantically_equal(group, semantic_columns):
                raise DataValidationError(
                    "SECURITY_IDENTITY_COLLISION: conflicting canonical rows "
                    f"dataset={dataset_name} key={key_values} "
                    f"source_codes={source_codes}"
                )
            selected_source = source_codes[0]
            drop_indexes.extend(
                group.index[group["source_ts_code"].astype(str) != selected_source].tolist()
            )
        return frame.drop(index=drop_indexes).reset_index(drop=True)

    def _validate(self) -> None:
        if len({(a.source_code, a.effective_from, a.effective_to) for a in self._aliases}) != len(
            self._aliases
        ):
            raise DataValidationError(
                "SECURITY_IDENTITY_MAPPING_INVALID: duplicate source/effective interval"
            )
        canonical_sources = {alias.source_code for alias in self._aliases}
        for alias in self._aliases:
            if alias.source_code == alias.canonical_code:
                raise DataValidationError(
                    "SECURITY_IDENTITY_MAPPING_INVALID: identity aliases must be omitted"
                )
            if alias.canonical_code in canonical_sources:
                raise DataValidationError(
                    "SECURITY_IDENTITY_MAPPING_INVALID: alias chains are not supported"
                )
            if alias.effective_from and alias.effective_to:
                if alias.effective_from > alias.effective_to:
                    raise DataValidationError(
                        "SECURITY_IDENTITY_MAPPING_INVALID: reversed effective interval"
                    )
        for source_code, aliases in self._by_source.items():
            for index, left in enumerate(aliases):
                for right in aliases[index + 1 :]:
                    if _intervals_overlap(left, right):
                        raise DataValidationError(
                            "SECURITY_IDENTITY_MAPPING_INVALID: overlapping intervals "
                            f"source_code={source_code}"
                        )


def _parse_alias(value: object) -> SecurityAlias:
    if not isinstance(value, dict):
        raise DataValidationError("SECURITY_IDENTITY_MAPPING_INVALID: alias must be an object")
    required = ("source_code", "canonical_code", "exchange", "mapping_source")
    fields = {name: str(value.get(name, "")).strip().upper() for name in required[:3]}
    mapping_source = str(value.get("mapping_source", "")).strip()
    if any(not fields[name] for name in required[:3]) or not mapping_source:
        raise DataValidationError("SECURITY_IDENTITY_MAPPING_INVALID: incomplete alias")
    source_code = fields["source_code"]
    canonical_code = fields["canonical_code"]
    exchange = fields["exchange"]
    if not source_code.endswith(f".{exchange}") or not canonical_code.endswith(f".{exchange}"):
        raise DataValidationError(
            "SECURITY_IDENTITY_MAPPING_INVALID: code suffix and exchange disagree"
        )
    return SecurityAlias(
        source_code=source_code,
        canonical_code=canonical_code,
        exchange=exchange,
        effective_from=_validated_date(value.get("effective_from")),
        effective_to=_validated_date(value.get("effective_to")),
        mapping_source=mapping_source,
    )


def _validated_date(value: object) -> str | None:
    if value is None or str(value).strip() in {"", "None", "nan", "NaT"}:
        return None
    date = str(value).strip()
    if len(date) != 8 or not date.isdigit():
        raise DataValidationError(
            f"SECURITY_IDENTITY_MAPPING_INVALID: effective date must be YYYYMMDD: {date}"
        )
    try:
        pd.Timestamp(date)
    except ValueError as error:
        raise DataValidationError(
            f"SECURITY_IDENTITY_MAPPING_INVALID: invalid effective date: {date}"
        ) from error
    return date


def _optional_date(value: object) -> str | None:
    if value is None or str(value) in {"", "<NA>", "NaT", "nan"}:
        return None
    return str(value)


def _source_groups_semantically_equal(frame: DataFrame, columns: list[str]) -> bool:
    signatures: list[tuple[tuple[str, ...], ...]] = []
    for _, source_rows in frame.groupby("source_ts_code", sort=True):
        if not columns:
            signatures.append(())
            continue
        normalized = (
            source_rows[columns].astype(object).where(source_rows[columns].notna(), "<NULL>")
        )
        signatures.append(tuple(sorted(tuple(map(str, row)) for row in normalized.to_numpy())))
    return all(signature == signatures[0] for signature in signatures[1:])


def _intervals_overlap(left: SecurityAlias, right: SecurityAlias) -> bool:
    left_start = left.effective_from or "00000000"
    left_end = left.effective_to or "99999999"
    right_start = right.effective_from or "00000000"
    right_end = right.effective_to or "99999999"
    return max(left_start, right_start) <= min(left_end, right_end)


def canonicalize_security_datasets(
    inputs: dict[str, DataFrame], resolver: SecurityIdentityResolver
) -> dict[str, DataFrame]:
    """Canonicalize all supported security datasets in a copied input mapping."""

    return {
        name: resolver.canonicalize_frame(frame, name)
        if name in SECURITY_IDENTITY_DATASETS
        else frame.copy()
        for name, frame in inputs.items()
    }


def scan_cross_source_identity(
    frames: dict[str, DataFrame], resolver: SecurityIdentityResolver
) -> SecurityIdentityScanResult:
    """Validate mapped alias keys against canonical daily/limit/universe identities."""

    canonical: dict[str, DataFrame] = {}
    observed_alias_rows = 0
    observed_aliases: set[str] = set()
    same_source_multi_event_keys = 0
    for name, frame in frames.items():
        if frame.empty or "ts_code" not in frame.columns:
            canonical[name] = frame.copy()
            continue
        source_codes = frame["ts_code"].astype(str).str.strip().str.upper()
        alias_mask = source_codes.isin(resolver.source_codes)
        observed_alias_rows += int(alias_mask.sum())
        observed_aliases.update(source_codes[alias_mask].tolist())
        if {"ts_code", "trade_date"}.issubset(frame.columns):
            same_source_multi_event_keys += int(
                frame.loc[frame.duplicated(["ts_code", "trade_date"], keep=False)]
                .loc[:, ["ts_code", "trade_date"]]
                .drop_duplicates()
                .shape[0]
            )
        canonical[name] = (
            resolver.canonicalize_frame(frame, name)
            if name in SECURITY_IDENTITY_DATASETS
            else frame.copy()
        )

    reference_keys: set[tuple[str, str]] = set()
    for name in ("daily", "stk_limit", "universe_daily"):
        frame = canonical.get(name, pd.DataFrame())
        if {"ts_code", "trade_date"}.issubset(frame.columns):
            reference_keys.update(
                zip(
                    frame["ts_code"].astype(str),
                    frame["trade_date"].astype(str),
                    strict=True,
                )
            )
    unresolved: list[str] = []
    for name, frame in canonical.items():
        if name not in {"suspend_d", "stk_limit"} or frame.empty:
            continue
        source_alias_mask = frame["source_ts_code"].astype(str).isin(resolver.source_codes)
        for _, row in frame.loc[source_alias_mask].iterrows():
            key = (str(row["ts_code"]), str(row["trade_date"]))
            if key not in reference_keys:
                unresolved.append(f"{name}:{key[1]}:{key[0]}")
    return SecurityIdentityScanResult(
        mapping_version=resolver.mapping_version,
        mapping_hash=resolver.mapping_hash,
        configured_aliases=resolver.alias_count,
        observed_alias_rows=observed_alias_rows,
        observed_aliases=len(observed_aliases),
        unresolved_mismatches=tuple(sorted(set(unresolved))),
        same_source_multi_event_keys=same_source_multi_event_keys,
    )
