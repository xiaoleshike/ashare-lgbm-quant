"""Atomic logical-append storage for paper-trading ledgers."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError

type DataFrame = pd.DataFrame


def read_ledger(path: Path) -> DataFrame:
    """Read one ledger, returning an empty frame when it has not been created."""

    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def append_ledger(
    path: Path,
    rows: DataFrame,
    *,
    unique_columns: tuple[str, ...],
    sort_columns: tuple[str, ...],
) -> int:
    """Atomically append immutable rows while rejecting conflicting identities."""

    if rows.empty:
        return 0
    missing = [column for column in unique_columns if column not in rows.columns]
    if missing:
        raise DataValidationError(f"paper ledger lacks identity columns: {missing}")
    if rows.duplicated(list(unique_columns)).any():
        raise DataValidationError("new paper ledger rows contain duplicate identities")
    existing = read_ledger(path)
    new_rows = rows.copy()
    if not existing.empty:
        if set(existing.columns) != set(new_rows.columns):
            raise DataValidationError(f"paper ledger schema changed: {path}")
        new_rows = new_rows.loc[:, existing.columns]
        existing_keys = {
            tuple(value)
            for value in existing.loc[:, list(unique_columns)].itertuples(index=False, name=None)
        }
        keep: list[bool] = []
        for row_index, row in enumerate(new_rows.itertuples(index=False)):
            payload = cast(Any, row)._asdict()
            identity = tuple(payload[column] for column in unique_columns)
            if identity not in existing_keys:
                keep.append(True)
                continue
            old = existing
            for column, value in zip(unique_columns, identity, strict=True):
                old = old.loc[old[column] == value]
            comparable = new_rows.iloc[[row_index]].loc[:, existing.columns].reset_index(drop=True)
            if not old.reset_index(drop=True).equals(comparable):
                raise DataValidationError(
                    f"append-only paper ledger identity has conflicting payload: {identity}"
                )
            keep.append(False)
        new_rows = new_rows.loc[keep]
    if new_rows.empty:
        return 0
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined = combined.sort_values(list(sort_columns), kind="mergesort").reset_index(drop=True)
    _atomic_write_parquet(path, combined)
    return len(new_rows)


def file_sha256(path: Path) -> str:
    """Return the SHA256 identity of an existing source artifact."""

    if not path.is_file():
        raise DataValidationError(f"paper-trading source artifact does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_sha256(payload: object) -> str:
    """Return a deterministic digest for JSON-compatible identity values."""

    import json

    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_parquet(path: Path, frame: DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
