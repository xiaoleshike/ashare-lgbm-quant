"""Single-security CNINFO discovery, replayable entirely from frozen HTTP bytes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any, Protocol
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ashare_quant.backtest.continuous_source import payload_hash
from ashare_quant.data.announcement_http import (
    AnnouncementError,
    SafeHTTPClient,
    check,
    official_url,
    request_spec,
    response_json,
)

TOP_SEARCH = "https://www.cninfo.com.cn/new/information/topSearch/query"
QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
STATIC_URL = "https://static.cninfo.com.cn/"
PROVIDER = "CNINFO_PUBLIC_WEB"
PLATES = {"SH": "sh", "SZ": "sz", "BJ": "bj"}
EXCHANGES = {
    "sh": "SH",
    "sse": "SH",
    "SH": "SH",
    "sz": "SZ",
    "szse": "SZ",
    "SZ": "SZ",
    "bj": "BJ",
    "bse": "BJ",
    "BJ": "BJ",
}
CATALOG_COLUMNS = [
    "provider",
    "provider_request_id",
    "source_sec_code",
    "canonical_ts_code",
    "exchange",
    "plate",
    "org_id",
    "org_id_response_hash",
    "announcement_id",
    "announcement_date",
    "announcement_timestamp",
    "announcement_title",
    "announcement_type",
    "adjunct_url",
    "official_document_url",
    "raw_response_hash",
    "raw_row_hash",
    "keyword_match",
    "source_format",
    "document_sha256",
    "document_size",
    "fetch_status",
    "extraction_status",
]


class AnnouncementQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ts_code: str = Field(pattern=r"^[0-9]{6}\.(SH|SZ|BJ)$")
    start_date: str = Field(pattern=r"^[0-9]{8}$")
    end_date: str = Field(pattern=r"^[0-9]{8}$")
    keyword: str = Field(default="", max_length=120)
    category: str = Field(default="", pattern=r"^[a-zA-Z0-9_;]*$", max_length=120)
    page_size: int = Field(default=30, ge=1, le=30)
    max_pages: int = Field(default=100, ge=1, le=100)

    @model_validator(mode="after")
    def dates(self) -> AnnouncementQuery:
        for dt in (self.start_date, self.end_date):
            datetime.strptime(dt, "%Y%m%d")
        check(self.start_date <= self.end_date, "QUERY_DATE_ORDER")
        return self


@dataclass
class SearchProgress:
    records: list[dict[str, Any]]
    status: str
    issues: list[str]
    next_request: dict[str, Any] | None


@dataclass
class SearchResult:
    query: AnnouncementQuery
    captures: list[dict[str, Any]]
    blobs: dict[str, bytes]
    progress: SearchProgress


class AnnouncementProvider(Protocol):
    def search(self, query: AnnouncementQuery) -> SearchResult: ...


class _Title(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def title_text(text: str) -> str:
    parser = _Title()
    parser.feed(text)
    return "".join(parser.parts).strip()


def org_identity(query: AnnouncementQuery, response: dict[str, Any] | list[Any]) -> str:
    rows = response.get("stockList") if isinstance(response, dict) else response
    check(isinstance(rows, list), "ORG_RESPONSE_SCHEMA")
    assert isinstance(rows, list)
    matches: set[tuple[str, str]] = set()
    for row in rows:
        check(isinstance(row, dict), "ORG_RESPONSE_SCHEMA")
        codes = {str(row[k]) for k in ("code", "secCode", "stockCode") if row.get(k)}
        if query.ts_code[:6] not in codes:
            continue
        check(codes == {query.ts_code[:6]}, "ORG_CODE_CONFLICT")
        exchanges = {
            EXCHANGES[str(row[k])] for k in ("plate", "exchange") if str(row.get(k)) in EXCHANGES
        }
        if not exchanges and row.get("column") == "sse":
            exchanges = {"SH"}
        check(len(exchanges) == 1, "ORG_EXCHANGE_UNVERIFIED")
        if row.get("column") is not None:
            expected_column = "sse" if exchanges == {"SH"} else "szse"
            check(row["column"] == expected_column, "ORG_COLUMN_CONFLICT")
        check(isinstance(row.get("orgId"), str) and bool(row["orgId"].strip()), "ORG_ID_MISSING")
        matches.add((row["orgId"], next(iter(exchanges))))
    check(len(matches) == 1, "ORG_NOT_FOUND_OR_AMBIGUOUS")
    org, exchange = next(iter(matches))
    check(exchange == query.ts_code[-2:], "ORG_EXCHANGE_MISMATCH")
    return org


def page_request(query: AnnouncementQuery, org: str, page: int) -> dict[str, Any]:
    def fmt(dt: str) -> str:
        return datetime.strptime(dt, "%Y%m%d").strftime("%Y-%m-%d")

    return request_spec(
        QUERY_URL,
        {
            "stock": f"{query.ts_code[:6]},{org}",
            "tabName": "fulltext",
            "pageSize": str(query.page_size),
            "pageNum": str(page),
            "column": "sse" if query.ts_code.endswith(".SH") else "szse",
            "plate": PLATES[query.ts_code[-2:]],
            "searchkey": query.keyword,
            "category": query.category,
            "seDate": f"{fmt(query.start_date)}~{fmt(query.end_date)}",
            "isHLtitle": "false",
        },
    )


def announcement_record(
    row: dict[str, Any],
    query: AnnouncementQuery,
    org: str,
    capture: dict[str, Any],
    org_hash: str,
) -> dict[str, Any]:
    check(
        row.get("secCode") == query.ts_code[:6] and row.get("orgId") == org,
        "ANNOUNCEMENT_IDENTITY_MISMATCH",
    )
    for field in ("announcementId", "announcementTitle", "adjunctUrl", "adjunctType"):
        check(
            isinstance(row.get(field), str) and bool(row[field].strip()), "ANNOUNCEMENT_ROW_SCHEMA"
        )
    check(isinstance(row.get("announcementType"), (str, type(None))), "ANNOUNCEMENT_TYPE_SCHEMA")
    timestamp = row.get("announcementTime")
    check(type(timestamp) is int and 0 <= timestamp < 32_503_680_000_000, "ANNOUNCEMENT_TIMESTAMP")
    assert isinstance(timestamp, int)
    dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone(timedelta(hours=8)))
    date = dt.strftime("%Y%m%d")
    check(query.start_date <= date <= query.end_date, "ANNOUNCEMENT_OUT_OF_WINDOW")
    url = official_url(urljoin(STATIC_URL, row["adjunctUrl"]))
    check(url.startswith(STATIC_URL), "ANNOUNCEMENT_DOCUMENT_HOST")
    title = title_text(row["announcementTitle"])
    return {
        "provider": PROVIDER,
        "provider_request_id": capture["request_id"],
        "source_sec_code": row["secCode"],
        "canonical_ts_code": query.ts_code,
        "exchange": query.ts_code[-2:],
        "plate": PLATES[query.ts_code[-2:]],
        "org_id": org,
        "org_id_response_hash": org_hash,
        "announcement_id": row["announcementId"],
        "announcement_date": date,
        "announcement_timestamp": dt.isoformat(),
        "announcement_title": title,
        "announcement_type": row.get("announcementType"),
        "adjunct_url": row["adjunctUrl"],
        "official_document_url": url,
        "raw_response_hash": capture["attempts"][-1]["body_sha256"],
        "raw_row_hash": payload_hash(row),
        "keyword_match": "NOT_REQUESTED"
        if not query.keyword
        else "TITLE_LITERAL"
        if query.keyword in title
        else "UNVERIFIED_PROVIDER_MATCH",
        "source_format": row["adjunctType"].upper(),
        "document_sha256": None,
        "document_size": None,
        "fetch_status": "NOT_ATTEMPTED",
        "extraction_status": "NOT_ATTEMPTED",
    }


def analyze_search(
    query: AnnouncementQuery, captures: list[dict[str, Any]], blobs: dict[str, bytes]
) -> SearchProgress:
    """One shared state machine for acquisition and independent business reconstruction."""
    records: list[dict[str, Any]] = []
    initial = request_spec(TOP_SEARCH, {"keyWord": query.ts_code[:6], "maxNum": "10"})
    if not captures:
        return SearchProgress(records, "NOT_ATTEMPTED", [], initial)
    check(captures[0]["request"] == initial, "ORG_REQUEST_MISMATCH")
    failure = captures[0]["status"]
    if failure != "CAPTURED":
        check(len(captures) == 1, "EXTRA_SEARCH_CAPTURE")
        return SearchProgress(
            records,
            "PERMISSION_REQUIRED" if failure == "PERMISSION_REQUIRED" else "FAILED",
            [failure],
            None,
        )
    try:
        org = org_identity(query, response_json(captures[0], blobs))
    except AnnouncementError as error:
        check(len(captures) == 1, "EXTRA_SEARCH_CAPTURE")
        return SearchProgress(records, "FAILED", [str(error)], None)
    org_hash = captures[0]["attempts"][-1]["body_sha256"]
    total: int | None = None
    seen: set[str] = set()
    for page, capture in enumerate(captures[1:], 1):
        check(
            page <= query.max_pages and capture["request"] == page_request(query, org, page),
            "PAGE_REQUEST_MISMATCH",
        )
        issues: list[str] = []
        status = "PARTIAL"
        if capture["status"] != "CAPTURED":
            issues = [capture["status"]]
            if capture["status"] == "PERMISSION_REQUIRED":
                status = "PERMISSION_REQUIRED"
        else:
            try:
                body = response_json(capture, blobs)
                check(isinstance(body, dict), "PAGE_SCHEMA")
                assert isinstance(body, dict)
                count, pages, more = (
                    body.get("totalAnnouncement"),
                    body.get("totalpages"),
                    body.get("hasMore"),
                )
                check(
                    type(count) is int
                    and count >= 0
                    and type(pages) is int
                    and pages >= 0
                    and type(more) is bool,
                    "PAGE_COUNTER_SCHEMA",
                )
                assert isinstance(count, int) and isinstance(pages, int)
                check(
                    pages == (count + query.page_size - 1) // query.page_size,
                    "PAGE_TOTAL_INCONSISTENT",
                )
                check(total is None or total == count, "PAGINATION_TOTAL_CHANGED")
                total = count
                items = body.get("announcements")
                if count == 0 and items is None:
                    items = []
                check(isinstance(items, list) and len(items) <= query.page_size, "PAGE_ROWS_SCHEMA")
                assert isinstance(items, list)
                expected = min(query.page_size, max(0, count - (page - 1) * query.page_size))
                check(
                    len(items) == expected and more == (page < pages), "PAGE_COVERAGE_INCONSISTENT"
                )
                page_rows = []
                for raw in items:
                    check(isinstance(raw, dict), "ANNOUNCEMENT_ROW_SCHEMA")
                    record = announcement_record(raw, query, org, capture, org_hash)
                    check(record["announcement_id"] not in seen, "DUPLICATE_ANNOUNCEMENT_ID")
                    seen.add(record["announcement_id"])
                    page_rows.append(record)
                records.extend(page_rows)
                if not more:
                    check(len(records) == count, "ANNOUNCEMENT_TOTAL_MISMATCH")
                    check(len(captures) == page + 1, "EXTRA_SEARCH_CAPTURE")
                    return SearchProgress(records, "COMPLETE", [], None)
            except AnnouncementError as error:
                issues = [str(error)]
        if issues:
            check(len(captures) == page + 1, "EXTRA_SEARCH_CAPTURE")
            return SearchProgress(records, status, issues, None)
    page = len(captures)
    if page > query.max_pages:
        return SearchProgress(records, "PARTIAL", ["MAX_PAGES_REACHED"], None)
    return SearchProgress(
        records, "PARTIAL", ["PAGINATION_INTERRUPTED"], page_request(query, org, page)
    )


class CNINFOAnnouncementProvider:
    def __init__(self, client: SafeHTTPClient) -> None:
        self.client = client

    def search(self, query: AnnouncementQuery) -> SearchResult:
        captures: list[dict[str, Any]] = []
        blobs: dict[str, bytes] = {}
        while True:
            progress = analyze_search(query, captures, blobs)
            if progress.next_request is None:
                return SearchResult(query, captures, blobs, progress)
            capture, body = self.client.fetch(progress.next_request)
            captures.append(capture)
            blobs.update(body)
