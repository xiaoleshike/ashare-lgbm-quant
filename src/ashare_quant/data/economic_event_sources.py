"""Audit-bound exposure inventory and isolated, explicit dividend acquisition."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ashare_quant.backtest.continuous import validate_continuous_strict_oos_artifact
from ashare_quant.backtest.continuous_source import file_hash, payload_hash, read_json
from ashare_quant.config.settings import load_settings
from ashare_quant.data.economic_events import (
    AMOUNTS,
    CONTRACT,
    DIVIDEND_FIELDS,
    date_value,
    require,
)
from ashare_quant.data.exceptions import DataIngestionError
from ashare_quant.data.security_identity import SecurityIdentityResolver
from ashare_quant.data.security_lifecycle_source_probe import ProviderClient
from ashare_quant.utils.manifest import atomic_write_json, utc_now_iso

AUDIT_FILES = (
    "adjustment_crossing_audit.csv",
    "adjustment_change_details.csv",
    "terminal_writeoff_audit.csv",
    "terminal_official_index_matches.csv",
    "positions.csv",
)
EXPOSURE_COLUMNS = [
    "exposure_id",
    "economic_boundary_id",
    "exposure_type",
    "ts_code",
    "position_id",
    "top_n",
    "entry_date",
    "target_exit_date",
    "resolution_date",
    "candidate_event_date",
    "previous_observation",
    "adj_factor_before",
    "adj_factor_after",
    "entry_gross",
    "continuous_run_id",
    "continuous_manifest_hash",
    "audit_manifest_hash",
]
RAW_COLUMNS = ["ts_code", "source_ts_code", "request_id", "provider_raw_row_hash", "raw_row_json"]


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-safe primitive rows; null stays null, no finite-value coercion."""
    result: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        result.append(
            {
                str(k): None if pd.isna(v) else v.item() if isinstance(v, np.generic) else v
                for k, v in row.items()
            }
        )
    payload_hash(result)
    return result


def content_hash(frame: pd.DataFrame) -> str:
    return payload_hash({"columns": list(frame.columns), "rows": records(frame)})


def typed_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Stable nullable Parquet field types, including zero-row evidence tables."""
    floating = {*AMOUNTS.values(), "adj_factor_before", "adj_factor_after", "entry_gross"}
    integer = {"top_n", "exposure_count"}
    boolean = {"within_execution_support", "observed_adjustment"}
    return frame.astype(
        {
            name: "Float64"
            if name in floating
            else "Int64"
            if name in integer
            else "boolean"
            if name in boolean
            else "string"
            for name in frame
        }
    )


def safe_file(root: Path, relative: str) -> Path:
    path = root / relative
    require(
        not Path(relative).is_absolute() and path.resolve().is_relative_to(root.resolve()),
        "PATH_ESCAPE",
    )
    require(path.is_file() and not path.is_symlink(), "FILE_MISSING")
    return path


def implementation_hash() -> str:
    base = Path(__file__).parent
    return payload_hash(
        {
            name: file_hash(base / name)
            for name in (
                "economic_events.py",
                "economic_event_sources.py",
                "economic_event_evidence.py",
            )
        }
    )


def compatible_implementation(digest: str) -> bool:
    """Read v1 inputs, but always rederive their business tables with hardened dates.

    The pinned pre-fix implementation is readable, not authorization to reuse its
    terminal completeness result. New publications bind the current content hash.
    """
    return digest in {
        implementation_hash(),
        "b6b3bdc88be0be193f225b58141e3186dfd239b03dad1318a29fdad5ff8289c4",
    }


def publish(
    root: Path,
    kind: str,
    logical: dict[str, Any],
    locators: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    objects: dict[str, Any],
    validate: Callable[[Path], object],
    documents: dict[str, bytes] | None = None,
) -> Path:
    """Use the repository's manifest-last/staging pattern, never overwrite a version."""
    artifact_id = kind + "_" + payload_hash(logical)[:24]
    output = root / artifact_id
    if output.exists():
        validate(output)
        return output
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".economic-staging-", dir=root) as tmp:
        stage = Path(tmp) / artifact_id
        stage.mkdir()
        for name, frame in tables.items():
            typed_table(frame).to_parquet(stage / name, index=False)
        for name, value in objects.items():
            atomic_write_json(stage / name, value)
        for name, data in (documents or {}).items():
            (stage / name).parent.mkdir(parents=True, exist_ok=True)
            (stage / name).write_bytes(data)
        hashes = {
            str(p.relative_to(stage)): file_hash(p) for p in sorted(stage.rglob("*")) if p.is_file()
        }
        atomic_write_json(
            stage / "manifest.json",
            {
                "schema_version": 1,
                "artifact_type": kind,
                "artifact_id": artifact_id,
                "status": "COMPLETE",
                "logical_identity": logical,
                "locators": locators,
                "artifact_hashes": hashes,
            },
        )
        validate(stage)
        require(not output.exists(), "PUBLICATION_RACE")
        os.rename(stage, output)
    return output


def validate_envelope(root: Path, kind: str) -> dict[str, Any]:
    m = read_json(root / "manifest.json")
    require(
        m.get("schema_version") == 1
        and m.get("status") == "COMPLETE"
        and m.get("artifact_type") == kind,
        "SCHEMA_INVALID",
    )
    expected = kind + "_" + payload_hash(m["logical_identity"])[:24]
    require(root.name == m.get("artifact_id") == expected, "ID_MISMATCH")
    require(
        set(m["artifact_hashes"])
        == {
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file() and p != root / "manifest.json"
        },
        "FILE_SET_MISMATCH",
    )
    for rel, digest in m["artifact_hashes"].items():
        require(file_hash(safe_file(root, rel)) == digest, "CHILD_HASH_MISMATCH")
    return m


@dataclass(frozen=True)
class EconomicExposureSource:
    run: Path
    audit: Path
    identity: dict[str, Any]
    exposures: pd.DataFrame
    positions: pd.DataFrame
    holdings: pd.DataFrame
    calendar: tuple[str, ...]
    mapping: SecurityIdentityResolver


def load_exposure_source(
    run: Path,
    audit: Path,
    *,
    continuous_hash: str,
    audit_hash: str,
) -> EconomicExposureSource:
    """Validate the existing source; retain audit membership, never another run's exposure."""
    require(file_hash(run / "manifest.json") == continuous_hash, "CONTINUOUS_HASH_MISMATCH")
    source = validate_continuous_strict_oos_artifact(run)
    require(file_hash(audit / "audit_manifest.json") == audit_hash, "AUDIT_HASH_MISMATCH")
    am = read_json(audit / "audit_manifest.json")
    require(
        am.get("status") == "COMPLETE"
        and am.get("source_run_id") == source["run_id"]
        and am.get("source_manifest_hash") == continuous_hash,
        "AUDIT_SOURCE_MISMATCH",
    )
    require(set(AUDIT_FILES) <= set(am["output_hashes"]), "AUDIT_FILES_MISSING")
    for rel, digest in am["output_hashes"].items():
        require(file_hash(safe_file(audit, rel)) == digest, "AUDIT_CHILD_MISMATCH")
    for path, digest in am["input_child_hashes"].items():
        require(file_hash(Path(path)) == digest, "AUDIT_INPUT_MISMATCH")
    for rel, digest in am["analysis_script_hashes"].items():
        require(file_hash(safe_file(audit, rel)) == digest, "AUDIT_SCRIPT_MISMATCH")
    frames = {n: pd.read_csv(audit / n, dtype=str, keep_default_na=False) for n in AUDIT_FILES}
    positions = frames["positions.csv"].copy()
    require(not positions.duplicated(["top_n", "position_id"]).any(), "DUPLICATE_POSITION")
    holdings, calendars = [], []
    portfolio_ids = sorted(
        {
            Path(rel).parts[0].removeprefix("top_")
            for rel in source["artifact_hashes"]
            if rel.endswith("/trades.parquet")
        }
    )
    require(set(positions.top_n) <= set(portfolio_ids), "PORTFOLIO_SET_MISMATCH")
    for n in portfolio_ids:
        trades = pd.read_parquet(run / f"top_{n}" / "trades.parquet")
        buys = trades[trades.side.eq("buy") & trades.status.eq("filled")].set_index("position_id")
        ends = trades[
            trades.side.eq("sell") & trades.status.isin(["filled", "terminal_writeoff"])
        ].set_index("position_id")
        p = positions[positions.top_n.eq(n)].set_index("position_id")
        require(set(p.index) == set(buys.index) == set(ends.index), "POSITION_MEMBERSHIP_MISMATCH")
        for field, actual in (
            ("entry_date", buys.trade_date),
            ("resolution_date", ends.trade_date),
            ("ts_code", buys.ts_code),
        ):
            require(
                p[field].equals(actual.reindex(p.index).rename(field)), "POSITION_LINEAGE_MISMATCH"
            )
        require(
            np.allclose(
                p.entry_gross.astype(float), buys.gross_value.reindex(p.index), rtol=1e-12, atol=0
            ),
            "POSITION_CAPITAL_MISMATCH",
        )
        h = pd.read_parquet(
            run / f"top_{n}" / "holdings.parquet",
            columns=["trade_date", "position_id", "ts_code", "target_exit_date"],
        )
        targets = h.groupby("position_id").target_exit_date.first().reindex(p.index)
        require(
            p.target_exit_date.equals(targets.rename("target_exit_date")),
            "POSITION_TARGET_MISMATCH",
        )
        h["top_n"] = int(n)
        holdings.append(h)
        daily = pd.read_parquet(run / f"top_{n}" / "daily_returns.parquet", columns=["trade_date"])
        calendars.append(tuple(daily.trade_date))
        t = frames["terminal_writeoff_audit.csv"]
        require(
            set(t[t.top_n.eq(n)].position_id)
            == set(ends[ends.status.eq("terminal_writeoff")].index),
            "TERMINAL_MEMBERSHIP_MISMATCH",
        )
    require(bool(calendars) and all(c == calendars[0] for c in calendars), "CALENDAR_MISMATCH")
    for col in ("entry_date", "target_exit_date", "resolution_date"):
        for value in positions[col]:
            require(date_value(value) is not None, "POSITION_DATE_MISSING")
    crossing = frames["adjustment_crossing_audit.csv"]
    details = frames["adjustment_change_details.csv"]
    require(not crossing.duplicated(["top_n", "position_id"]).any(), "DUPLICATE_AUDIT_POSITION")
    require(set(crossing.position_id) == set(positions.position_id), "AUDIT_POSITION_SET_MISMATCH")
    for crossed in crossing.itertuples():
        ds = details[details.position_id.eq(crossed.position_id)]
        require(
            sorted(ds.change_date) == sorted(json.loads(str(crossed.change_dates)))
            and len(ds) == int(str(crossed.change_count)),
            "ADJUSTMENT_SET_MISMATCH",
        )
    rows = []
    lookup = positions.set_index(["top_n", "position_id"])
    for exposure_type, frame, date_col in (
        ("ADJ_FACTOR_CHANGE", details, "change_date"),
        ("TERMINAL_EVENT", frames["terminal_writeoff_audit.csv"], "writeoff_date"),
    ):
        for item in frame.to_dict("records"):
            p = lookup.loc[(item["top_n"], item["position_id"])]
            dt = date_value(item[date_col])
            require(
                dt is not None
                and p.entry_date <= dt <= p.resolution_date
                and item["ts_code"] == p.ts_code,
                "EXPOSURE_DATE_OR_CODE_MISMATCH",
            )
            if exposure_type == "TERMINAL_EVENT":
                require(dt == p.resolution_date, "TERMINAL_DATE_MISMATCH")
            else:
                previous = date_value(item["previous_observation"])
                require(
                    previous is not None and p.entry_date <= previous < str(dt),
                    "FACTOR_OBSERVATION_DATE_INVALID",
                )
            boundary = payload_hash([exposure_type, p.ts_code, dt])
            before = (
                float(item["previous_factor"]) if exposure_type == "ADJ_FACTOR_CHANGE" else None
            )
            after = float(item["new_factor"]) if exposure_type == "ADJ_FACTOR_CHANGE" else None
            require(
                before is None
                or (
                    np.isfinite(before)
                    and before > 0
                    and after is not None
                    and np.isfinite(after)
                    and after > 0
                    and before != after
                ),
                "FACTOR_INVALID",
            )
            rows.append(
                {
                    "exposure_id": payload_hash([boundary, item["top_n"], item["position_id"]]),
                    "economic_boundary_id": boundary,
                    "exposure_type": exposure_type,
                    "ts_code": p.ts_code,
                    "position_id": item["position_id"],
                    "top_n": int(item["top_n"]),
                    "entry_date": p.entry_date,
                    "target_exit_date": p.target_exit_date,
                    "resolution_date": p.resolution_date,
                    "candidate_event_date": dt,
                    "previous_observation": item.get("previous_observation"),
                    "adj_factor_before": before,
                    "adj_factor_after": after,
                    "entry_gross": float(p.entry_gross),
                    "continuous_run_id": source["run_id"],
                    "continuous_manifest_hash": continuous_hash,
                    "audit_manifest_hash": audit_hash,
                }
            )
    exposures = (
        pd.DataFrame(rows, columns=EXPOSURE_COLUMNS)
        .sort_values("exposure_id")
        .reset_index(drop=True)
    )
    require(not exposures.exposure_id.duplicated().any(), "DUPLICATE_EXPOSURE")
    changes = exposures[exposures.exposure_type.eq("ADJ_FACTOR_CHANGE")]
    require(
        not (
            changes.groupby("economic_boundary_id")[
                ["adj_factor_before", "adj_factor_after"]
            ].nunique(dropna=False)
            > 1
        )
        .any()
        .any(),
        "CONFLICTING_FACTOR_OBSERVATIONS",
    )
    positions["top_n"] = positions.top_n.astype(int)
    positions["entry_gross"] = positions.entry_gross.astype(float)
    inv = read_json(run / "source_inventory.json")
    settings = load_settings(Path(inv["locators"]["config_path"]))
    mapping = SecurityIdentityResolver.from_path(settings.security_identity.mapping_path)
    identity = {
        "continuous_run_id": source["run_id"],
        "continuous_manifest_hash": continuous_hash,
        "audit_identity": am["analysis_identity"],
        "audit_manifest_hash": audit_hash,
        "audit_definitions_hash": payload_hash(am["definitions"]),
        "exposure_inventory_hash": content_hash(exposures),
        "identity_transition_hash": file_hash(
            Path(inv["locators"]["identity_transition_artifact"]) / "manifest.json"
        ),
        "mapping_hash": mapping.mapping_hash,
        "signal_start": calendars[0][0],
        "execution_end": calendars[0][-1],
        "governed_execution_cutoff": source["logical_identity"]["execution_inputs"][
            "governed_execution_cutoff"
        ],
    }
    return EconomicExposureSource(
        run, audit, identity, exposures, positions, pd.concat(holdings), calendars[0], mapping
    )


def query_targets(source: EconomicExposureSource) -> list[str]:
    return sorted(source.mapping.source_codes_for(set(source.exposures.ts_code)))


def raw_rows(responses: list[dict[str, Any]], mapping: SecurityIdentityResolver) -> pd.DataFrame:
    rows = []
    for request in responses:
        for body in request["rows"]:
            require(body.get("ts_code") == request["ts_code"], "PROVIDER_CODE_MISMATCH")
            canonical = mapping.canonicalize(str(body["ts_code"]))
            rows.append(
                {
                    "ts_code": canonical,
                    "source_ts_code": body["ts_code"],
                    "request_id": request["request_id"],
                    "provider_raw_row_hash": payload_hash(body),
                    "raw_row_json": json.dumps(
                        body, sort_keys=True, separators=(",", ":"), allow_nan=False
                    ),
                }
            )
    return (
        pd.DataFrame(rows, columns=RAW_COLUMNS)
        .sort_values(["request_id", "provider_raw_row_hash"])
        .reset_index(drop=True)
    )


def probe_dividends(
    source: EconomicExposureSource, client: ProviderClient, output_root: Path
) -> Path:
    """Explicit network entry only; per-security history, capped responses fail closed."""
    responses = []
    for code in query_targets(source):
        params = {"ts_code": code, "fields": ",".join(DIVIDEND_FIELDS)}
        request_id = payload_hash({"endpoint": "dividend", "params": params})
        item: dict[str, Any] = {
            "request_id": request_id,
            "ts_code": code,
            "params": params,
            "rows": [],
            "columns": [],
        }
        try:
            frame = client.query("dividend", **params)
            item["columns"] = list(frame.columns)
            item["rows"] = records(frame)
            if len(frame) >= 2000:
                item["status"] = "POSSIBLY_TRUNCATED"
            elif not set(DIVIDEND_FIELDS) <= set(frame.columns):
                item["status"] = "SCHEMA_INCOMPLETE"
            else:
                item["status"] = "RESPONSE_CAPTURED" if len(frame) else "EMPTY_RESPONSE_NOT_PROOF"
        except DataIngestionError as error:
            # Never persist provider exception text, which may contain a credential.
            item["status"] = "OPERATOR_ACTION_REQUIRED"
            item["error_type"] = type(error).__name__
        responses.append(item)
    logical = {
        "contract": CONTRACT,
        "provider": "TUSHARE_PRO",
        "endpoint": "dividend",
        "source": source.identity,
        "requests_hash": payload_hash(responses),
        "target_codes": query_targets(source),
        "retrieval_contract": "per_security_history_cap_2000_v1",
        "implementation_hash": implementation_hash(),
        "evidence_use": "POST_HOC_ACCOUNTING_EVIDENCE",
    }
    raw = raw_rows(responses, source.mapping)
    source_id = "economic_event_source_" + payload_hash(logical)[:24]
    template = {
        "reviews": [
            {
                "provider_source_id": source_id,
                "provider_raw_row_hash": h,
                "decision": "PENDING",
                "reviewed_by": "",
                "reviewed_fact": "",
            }
            for h in sorted(set(raw.provider_raw_row_hash))
        ]
    }
    return publish(
        output_root,
        "economic_event_source",
        logical,
        {"continuous_run": str(source.run.resolve()), "audit_root": str(source.audit.resolve())},
        {"raw_dividend_evidence.parquet": raw, "exposure_inventory.parquet": source.exposures},
        {
            "responses.json": {"requests": responses},
            "dividend_review_template.json": template,
            "retrieval.json": {"retrieved_at": utc_now_iso(), "provider": "TUSHARE_PRO"},
        },
        lambda p: validate_provider_source(p, source),
    )


def validate_provider_source(path: Path, source: EconomicExposureSource) -> dict[str, Any]:
    m = validate_envelope(path, "economic_event_source")
    logical = m["logical_identity"]
    require(
        logical["source"] == source.identity
        and logical["contract"] == CONTRACT
        and compatible_implementation(logical["implementation_hash"]),
        "SOURCE_CONTRACT_MISMATCH",
    )
    require(
        logical.get("provider") == "TUSHARE_PRO"
        and logical.get("endpoint") == "dividend"
        and logical.get("retrieval_contract") == "per_security_history_cap_2000_v1"
        and logical.get("evidence_use") == "POST_HOC_ACCOUNTING_EVIDENCE",
        "PROVIDER_CONTRACT_MISMATCH",
    )
    require(
        set(m["artifact_hashes"])
        == {
            "responses.json",
            "raw_dividend_evidence.parquet",
            "exposure_inventory.parquet",
            "dividend_review_template.json",
            "retrieval.json",
        },
        "PROVIDER_FILE_SET_MISMATCH",
    )
    req = read_json(path / "responses.json")["requests"]
    require(
        [r["ts_code"] for r in req] == query_targets(source) == logical["target_codes"],
        "PROVIDER_TARGET_SET_MISMATCH",
    )
    require(payload_hash(req) == logical["requests_hash"], "PROVIDER_CONTENT_MISMATCH")
    for r in req:
        params = {"ts_code": r["ts_code"], "fields": ",".join(DIVIDEND_FIELDS)}
        require(
            r["params"] == params
            and r["request_id"] == payload_hash({"endpoint": "dividend", "params": params}),
            "REQUEST_MISMATCH",
        )
        if r["status"] == "RESPONSE_CAPTURED":
            require(
                0 < len(r["rows"]) < 2000 and set(DIVIDEND_FIELDS) <= set(r["columns"]),
                "PROVIDER_COVERAGE_INVALID",
            )
        else:
            require(
                r["status"]
                in {
                    "POSSIBLY_TRUNCATED",
                    "SCHEMA_INCOMPLETE",
                    "EMPTY_RESPONSE_NOT_PROOF",
                    "OPERATOR_ACTION_REQUIRED",
                },
                "PROVIDER_STATUS_INVALID",
            )
            if r["status"] == "EMPTY_RESPONSE_NOT_PROOF":
                require(not r["rows"], "PROVIDER_STATUS_INVALID")
            elif r["status"] == "POSSIBLY_TRUNCATED":
                require(len(r["rows"]) >= 2000, "PROVIDER_STATUS_INVALID")
            elif r["status"] == "SCHEMA_INCOMPLETE":
                require(not set(DIVIDEND_FIELDS) <= set(r["columns"]), "PROVIDER_STATUS_INVALID")
            else:
                require(not r["rows"] and bool(r.get("error_type")), "PROVIDER_STATUS_INVALID")
    require(
        content_hash(pd.read_parquet(path / "raw_dividend_evidence.parquet"))
        == content_hash(raw_rows(req, source.mapping)),
        "RAW_RECONSTRUCTION_MISMATCH",
    )
    require(
        content_hash(pd.read_parquet(path / "exposure_inventory.parquet"))
        == content_hash(source.exposures),
        "EXPOSURE_RECONSTRUCTION_MISMATCH",
    )
    return m
