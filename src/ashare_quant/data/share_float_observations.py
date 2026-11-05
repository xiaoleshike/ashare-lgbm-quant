"""Lossless, content-addressed share_float observations, not resolved unlock events."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError

CONTRACT = "share_float_observation_v1"
BUSINESS_KEY = ("ts_code", "ann_date", "float_date", "holder_name", "share_type")
METADATA = (
    "observation_contract",
    "observation_hash",
    "raw_row_json",
    "source_requests_json",
    "legacy_provenance",
    "identity_status",
    "identity_variant_count",
)


def _json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def _hash(raw: str) -> str:
    return hashlib.sha256(f"{CONTRACT}\n{raw}".encode()).hexdigest()


def _scalar(value: object) -> str | int | float | bool | None:
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise DataValidationError("share_float contains a non-finite value")
        return value
    raise DataValidationError("share_float contains an unsupported provider value")


def _check_identity(record: dict[str, Any]) -> None:
    if any(not isinstance(record.get(key), str) or not record[key].strip() for key in BUSINESS_KEY):
        raise DataValidationError("share_float has missing or null detail identity")


def _requests(value: str, ann_date: str) -> list[dict[str, Any]]:
    requests = json.loads(value)
    if not isinstance(requests, list):
        raise DataValidationError("share_float invalid request provenance")
    for request in requests:
        if not isinstance(request, dict) or set(request) != {"endpoint", "params"}:
            raise DataValidationError("share_float invalid request provenance")
        params = request["params"]
        if (
            request["endpoint"] != "share_float"
            or not isinstance(params, dict)
            or not {"ann_date", "limit", "offset"}.issubset(params)
            or set(params) - {"ann_date", "limit", "offset", "fields"}
            or params["ann_date"] != ann_date
            or type(params["limit"]) is not int
            or not 1 <= params["limit"] <= 6000
            or type(params["offset"]) is not int
            or params["offset"] < 0
            or params["offset"] % params["limit"]
            or ("fields" in params and not isinstance(params["fields"], str))
        ):
            raise DataValidationError("share_float invalid request provenance")
    if value != _json([json.loads(item) for item in sorted({_json(r) for r in requests})]):
        raise DataValidationError("share_float non-canonical request provenance")
    return requests


def _annotate(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["identity_variant_count"] = result.groupby(list(BUSINESS_KEY))[
        "observation_hash"
    ].transform("nunique")
    result["identity_status"] = result["identity_variant_count"].map(
        lambda count: "AMBIGUOUS" if count > 1 else "SINGLE_OBSERVATION"
    )
    return result.sort_values([*BUSINESS_KEY, "observation_hash"]).reset_index(drop=True)


def capture_observations(
    frame: pd.DataFrame, request: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Freeze parsed provider values. Missing request provenance is explicitly legacy/unknown."""

    if set(frame.columns).intersection(METADATA):
        raise DataValidationError("share_float provider response contains reserved metadata")
    if not frame.columns.is_unique or not all(isinstance(c, str) for c in frame.columns):
        raise DataValidationError("share_float invalid provider columns")
    rows = []
    for record in frame.to_dict("records"):
        raw_record = {str(key): _scalar(value) for key, value in record.items()}
        _check_identity(raw_record)
        raw = _json(raw_record)
        provenance = _json([] if request is None else [request])
        _requests(provenance, str(raw_record["ann_date"]))
        rows.append(
            {
                **raw_record,
                "observation_contract": CONTRACT,
                "observation_hash": _hash(raw),
                "raw_row_json": raw,
                "source_requests_json": provenance,
                "legacy_provenance": request is None,
            }
        )
    if not rows:
        return pd.DataFrame(columns=[*frame.columns, *METADATA])
    return _annotate(pd.DataFrame(rows).drop_duplicates(subset="observation_hash"))


def validate_observations(frame: pd.DataFrame) -> None:
    """Recompute row identities and group ambiguity; never silently repair stored metadata."""

    try:
        if not set(METADATA).issubset(frame.columns):
            raise DataValidationError("share_float incomplete observation metadata")
        if frame.observation_hash.duplicated().any():
            raise DataValidationError("share_float duplicate observation hash")
        for record in frame.to_dict("records"):
            raw = record["raw_row_json"]
            payload = json.loads(raw)
            if not isinstance(payload, dict) or set(payload).intersection(METADATA):
                raise DataValidationError("share_float invalid raw row")
            _check_identity(payload)
            if (
                record["observation_contract"] != CONTRACT
                or raw != _json(payload)
                or record["observation_hash"] != _hash(raw)
            ):
                raise DataValidationError("share_float observation hash/contract mismatch")
            for column, value in payload.items():
                if column not in record or _scalar(record[column]) != value:
                    raise DataValidationError("share_float raw/provider value mismatch")
            for column in set(record) - set(payload) - set(METADATA):
                if _scalar(record[column]) is not None:
                    raise DataValidationError("share_float unbound provider column")
            requests = _requests(record["source_requests_json"], payload["ann_date"])
            if type(record["legacy_provenance"]) is not bool or (
                not requests and not record["legacy_provenance"]
            ):
                raise DataValidationError("share_float missing request provenance")
        expected = _annotate(frame).set_index("observation_hash")
        observed = frame.set_index("observation_hash").loc[expected.index]
        for column in ("identity_status", "identity_variant_count"):
            if not observed[column].equals(expected[column]):
                raise DataValidationError("share_float identity ambiguity annotation mismatch")
    except (TypeError, ValueError, KeyError) as error:
        raise DataValidationError(f"share_float invalid observation encoding: {error}") from error


def prepare_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Adapt legacy rows in memory or strictly validate a new-format partition."""

    if not set(frame.columns).intersection(METADATA):
        return capture_observations(frame)
    validate_observations(frame)
    return frame.copy()


def merge_observations(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Union all distinct observations and request references, then recompute ambiguity."""

    combined = pd.concat([prepare_observations(frame) for frame in frames], ignore_index=True)
    rows: dict[str, dict[str, Any]] = {}
    for incoming in combined.to_dict("records"):
        digest = str(incoming["observation_hash"])
        if digest not in rows:
            rows[digest] = {str(key): value for key, value in incoming.items()}
            continue
        record = rows[digest]
        if record["raw_row_json"] != incoming["raw_row_json"]:
            raise DataValidationError("share_float observation hash collision")
        requests = {
            _json(request)
            for value in (record["source_requests_json"], incoming["source_requests_json"])
            for request in json.loads(value)
        }
        record["source_requests_json"] = _json([json.loads(value) for value in sorted(requests)])
        record["legacy_provenance"] = bool(
            record["legacy_provenance"] or incoming["legacy_provenance"]
        )
    result = _annotate(pd.DataFrame(rows.values()))
    validate_observations(result)
    return result
