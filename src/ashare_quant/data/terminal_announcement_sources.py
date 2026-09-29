"""Optional bounded announcement metadata discovery. Never fetch document bodies."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.backtest.continuous_source import payload_hash, read_json
from ashare_quant.data.economic_event_sources import (
    EconomicExposureSource,
    publish,
    records,
    validate_envelope,
)
from ashare_quant.data.economic_events import date_value, require
from ashare_quant.data.exceptions import DataIngestionError, TusharePermissionError
from ashare_quant.data.security_lifecycle_source_probe import ProviderClient

DISCOVERY_CONTRACT = "terminal_announcement_discovery_v1"
WINDOW_DAYS = 365
CAP = 2000
FIELDS = ("ann_date", "ts_code", "name", "title", "url")
TERMS = tuple(
    (
        "\u9000\u5e02 \u7ec8\u6b62\u4e0a\u5e02 \u5438\u6536\u5408\u5e76 "
        "\u6362\u80a1 \u73b0\u91d1\u9009\u62e9\u6743 \u7834\u4ea7 "
        "\u91cd\u6574 \u6e05\u7b97 \u80a1\u6743 \u786e\u6743 \u8f6c\u677f \u6458\u724c"
    ).split()
)
CANDIDATE_COLUMNS = [
    "terminal_event_id",
    "ts_code",
    "terminal_date",
    "announcement_date",
    "title",
    "url",
    "metadata_hash",
    "request_id",
    "evidence_use",
]


def requests_for(source: EconomicExposureSource) -> list[dict[str, Any]]:
    result = []
    terms = source.exposures[source.exposures.exposure_type.eq("TERMINAL_EVENT")]
    for boundary, group in terms.groupby("economic_boundary_id", sort=True):
        first = group.iloc[0]
        dt = datetime.strptime(first.candidate_event_date, "%Y%m%d")
        for code in sorted(source.mapping.source_codes_for({first.ts_code})):
            request = {
                "terminal_event_id": str(boundary),
                "canonical_ts_code": first.ts_code,
                "terminal_date": first.candidate_event_date,
                "endpoint": "anns_d",
                "params": {
                    "ts_code": code,
                    "start_date": (dt - timedelta(days=WINDOW_DAYS)).strftime("%Y%m%d"),
                    "end_date": (dt + timedelta(days=WINDOW_DAYS)).strftime("%Y%m%d"),
                    "fields": ",".join(FIELDS),
                },
            }
            request["request_id"] = payload_hash(request)
            result.append(request)
    return result


def response_status(rows: list[dict[str, Any]], request: dict[str, Any]) -> str:
    if len(rows) >= CAP:
        return "POSSIBLY_TRUNCATED"
    if not rows:
        return "EMPTY_RESPONSE_NOT_PROOF"
    for row in rows:
        if not set(FIELDS) <= set(row):
            return "SCHEMA_INCOMPLETE"
        try:
            dt = date_value(row["ann_date"])
        except DataIngestionError:
            return "RESPONSE_SCOPE_INVALID"
        if (
            not dt
            or row["ts_code"] != request["params"]["ts_code"]
            or not request["params"]["start_date"] <= dt <= request["params"]["end_date"]
        ):
            return "RESPONSE_SCOPE_INVALID"
        if not isinstance(row["title"], str) or not isinstance(row["url"], str):
            return "SCHEMA_INCOMPLETE"
    return "METADATA_CAPTURED"


def candidate_rows(responses: list[dict[str, Any]]) -> pd.DataFrame:
    result = []
    for response in responses:
        if response["status"] != "METADATA_CAPTURED":
            continue
        for row in response["rows"]:
            if not any(term in row["title"] for term in TERMS):
                continue
            result.append(
                {
                    "terminal_event_id": response["terminal_event_id"],
                    "ts_code": response["canonical_ts_code"],
                    "terminal_date": response["terminal_date"],
                    "announcement_date": row["ann_date"],
                    "title": row["title"],
                    "url": row["url"],
                    "metadata_hash": payload_hash(row),
                    "request_id": response["request_id"],
                    "evidence_use": "DISCOVERY_ONLY_NOT_ECONOMIC_EVIDENCE",
                }
            )
    return (
        pd.DataFrame(result, columns=CANDIDATE_COLUMNS)
        .drop_duplicates()
        .sort_values(["terminal_event_id", "metadata_hash"])
        .reset_index(drop=True)
    )


def probe_announcements(source: EconomicExposureSource, client: ProviderClient, root: Path) -> Path:
    responses = []
    denied = False
    for request in requests_for(source):
        rows: list[dict[str, Any]] = []
        if denied:
            status = "PERMISSION_REQUIRED"
        else:
            try:
                rows = records(client.query("anns_d", **request["params"]))
                status = response_status(rows, request)
            except TusharePermissionError:
                denied = True
                status = "PERMISSION_REQUIRED"
            except DataIngestionError:
                status = "RETRIEVAL_FAILED"
        responses.append({**request, "rows": rows, "status": status})
    logical = {
        "contract": DISCOVERY_CONTRACT,
        "source": source.identity,
        "responses_hash": payload_hash(responses),
        "window_calendar_days_each_side": WINDOW_DAYS,
        "title_terms": list(TERMS),
        "evidence_use": "POST_HOC_DISCOVERY_ONLY",
    }
    return publish(
        root,
        "terminal_announcement_source",
        logical,
        {},
        {},
        {"responses.json": {"requests": responses}, "summary.json": discovery_summary(responses)},
        lambda p: validate_announcement_source(p, source),
    )


def validate_announcement_source(path: Path, source: EconomicExposureSource) -> pd.DataFrame:
    m = validate_envelope(path, "terminal_announcement_source")
    require(
        set(m["artifact_hashes"]) == {"responses.json", "summary.json"}, "ANNOUNCEMENT_FILE_SET"
    )
    responses = read_json(path / "responses.json")["requests"]
    require(
        m["logical_identity"]
        == {
            "contract": DISCOVERY_CONTRACT,
            "source": source.identity,
            "responses_hash": payload_hash(responses),
            "window_calendar_days_each_side": WINDOW_DAYS,
            "title_terms": list(TERMS),
            "evidence_use": "POST_HOC_DISCOVERY_ONLY",
        },
        "ANNOUNCEMENT_IDENTITY_MISMATCH",
    )
    expected = requests_for(source)
    require(len(expected) == len(responses), "ANNOUNCEMENT_REQUEST_SET")
    for request, response in zip(expected, responses, strict=True):
        require(
            {k: v for k, v in response.items() if k not in {"rows", "status"}} == request,
            "ANNOUNCEMENT_REQUEST_MISMATCH",
        )
        if response["status"] in {"PERMISSION_REQUIRED", "RETRIEVAL_FAILED"}:
            require(response["rows"] == [], "ANNOUNCEMENT_FAILURE_WITH_ROWS")
        else:
            require(
                response["status"] == response_status(response["rows"], request),
                "ANNOUNCEMENT_STATUS_MISMATCH",
            )
    require(
        read_json(path / "summary.json") == discovery_summary(responses),
        "ANNOUNCEMENT_SUMMARY_MISMATCH",
    )
    return candidate_rows(responses)


def discovery_summary(responses: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = {r["status"] for r in responses}
    return {
        "terminal_announcement_probe_status": "PERMISSION_REQUIRED"
        if "PERMISSION_REQUIRED" in statuses
        else "CAPTURED"
        if statuses <= {"METADATA_CAPTURED"}
        else "PARTIAL",
        "candidate_documents": len(candidate_rows(responses)),
        "economic_treatment_authorized": False,
        "operator_actions": [
            {
                "terminal_event_id": r["terminal_event_id"],
                "params": r["params"],
                "status": r["status"],
                "action": (
                    "Obtain permission or manually locate and freeze official documents; "
                    "metadata is not economic evidence."
                ),
            }
            for r in responses
            if r["status"] != "METADATA_CAPTURED"
        ],
    }
