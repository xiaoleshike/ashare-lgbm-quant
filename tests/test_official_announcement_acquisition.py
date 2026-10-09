"""Synthetic/offline only. A socket tripwire prevents accidental live acquisition."""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.backtest.continuous_source import file_hash, payload_hash, read_json
from ashare_quant.cli import main
from ashare_quant.data import announcement_http as http
from ashare_quant.data import announcement_sources as sources
from ashare_quant.data import announcement_text_extraction as text
from ashare_quant.data.announcement_documents import inspect_pdf, run_native
from ashare_quant.data.announcement_http import (
    AnnouncementError,
    CaptureCache,
    Limits,
    Reply,
    SafeHTTPClient,
    digest,
    official_url,
    request_spec,
)
from ashare_quant.data.announcement_sources import (
    acquire,
    probe,
    publish_discovery,
    validate_announcement_source,
)
from ashare_quant.data.cninfo_announcement_provider import (
    AnnouncementQuery,
    CNINFOAnnouncementProvider,
    SearchResult,
    analyze_search,
)
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.utils.manifest import atomic_write_json


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("LIVE_NETWORK_FORBIDDEN_IN_FIXTURES")

    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)


class FakeTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def exchange(self, method, url, form, limit, policy):
        self.calls.append((method, url, form))
        value = self.replies.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def reply(value, code=200, **headers):
    return Reply(code, headers, value if isinstance(value, bytes) else json.dumps(value).encode())


def org(code="688598", plate="sh"):
    return [{"code": code, "orgId": "fixture-org", "plate": plate}]


def announcement(aid="fixture-1", **changes):
    row = dict(
        secCode="688598",
        orgId="fixture-org",
        announcementId=aid,
        announcementTitle="Fixture distribution",
        announcementType="0101",
        announcementTime=int(datetime(2024, 7, 2, 16, 0, tzinfo=UTC).timestamp() * 1000),
        adjunctUrl=f"finalpage/2024-07-03/{aid}.PDF",
        adjunctType="PDF",
    )
    return row | changes


def page(rows, total=None, pages=1, more=False):
    return {
        "announcements": rows,
        "totalAnnouncement": len(rows) if total is None else total,
        "totalpages": pages,
        "hasMore": more,
    }


def client(replies, **kwargs):
    return SafeHTTPClient(
        authorized=True, transport=FakeTransport(replies), sleep=lambda _: None, **kwargs
    )


def query(**kwargs):
    return AnnouncementQuery(
        ts_code="688598.SH", start_date="20240701", end_date="20240705", **kwargs
    )


def web_org(code="688598", org_id="fixture-org"):
    return [{"code": code, "orgId": org_id, "category": "A股", "type": "shj"}]


@pytest.mark.parametrize(
    "code,exchange,board",
    [
        ("002713", "SZ", "SZZB"),
        ("300750", "SZ", "SZCY"),
        ("688598", "SH", "SHKCB"),
        ("600160", "SH", "SHZB"),
    ],
)
def test_web_org_confirmed_by_announcement_market(tmp_path, code, exchange, board):
    q = AnnouncementQuery(ts_code=f"{code}.{exchange}", start_date="20240701", end_date="20240705")
    result = CNINFOAnnouncementProvider(
        client(
            [
                reply(web_org(code)),
                reply(page([announcement(secCode=code, pageColumn=board)], pages=0)),
            ]
        )
    ).search(q)
    assert result.progress.status == "COMPLETE", result.progress.issues
    path = publish_discovery(result, Limits(), tmp_path)
    frame = pd.read_parquet(path / "announcement_catalog.parquet")
    assert frame.exchange.iloc[0] == exchange
    assert frame.exchange_evidence.iloc[0] == "ANNOUNCEMENT_PAGE_COLUMN"
    validate_announcement_source(path)


def test_web_floor_totalpages_keeps_all_31_rows(tmp_path):
    rows = [announcement(str(i), pageColumn="SHKCB") for i in range(31)]
    c = client(
        [
            reply(web_org()),
            reply(page(rows[:30], 31, 1, True)),
            reply(page(rows[30:], 31, 1, False)),
        ]
    )
    source = probe(query(), c, tmp_path)
    assert read_json(source / "acquisition_summary.json")["discovery_status"] == "COMPLETE"
    assert len(pd.read_parquet(source / "announcement_catalog.parquet")) == 31
    assert len(c.transport.calls) == 3


@pytest.mark.parametrize(
    "top,items",
    [
        (web_org(), [announcement(pageColumn="SZCY")]),
        (web_org(), [announcement()]),
        (web_org(), [announcement(pageColumn="UNKNOWN")]),
        (web_org(), [announcement(pageColumn=[])]),
        (
            [web_org()[0] | {"plate": "unknown"}],
            [announcement(pageColumn="SHKCB")],
        ),
        (web_org(), [announcement(pageColumn="SHKCB", orgId="wrong-org")]),
        (web_org(), []),
        (org(), [announcement(pageColumn="SZCY")]),
    ],
)
def test_web_exchange_cannot_be_guessed_or_conflicted(tmp_path, top, items):
    result = CNINFOAnnouncementProvider(
        client(
            [
                reply(top),
                reply(page(items, pages=0 if not items else 1)),
            ]
        )
    ).search(query())
    assert result.progress.status != "COMPLETE"
    path = publish_discovery(result, Limits(), tmp_path)
    validate_announcement_source(path)


@pytest.mark.parametrize("total", [0, 1, 29, 30, 31, 60, 61])
@pytest.mark.parametrize("style", ["floor", "ceil"])
def test_pagination_counter_styles_reconcile_count_and_more(tmp_path, total, style):
    rows = [announcement(str(i)) for i in range(total)]
    count_pages = (total + 29) // 30
    declared = total // 30 if style == "floor" else count_pages
    replies = [reply(org())] + [
        reply(page(rows[(n - 1) * 30 : n * 30], total, declared, n < count_pages))
        for n in range(1, max(1, count_pages) + 1)
    ]
    c = client(replies)
    path = probe(query(), c, tmp_path)
    assert read_json(path / "acquisition_summary.json")["discovery_status"] == "COMPLETE"
    assert len(pd.read_parquet(path / "announcement_catalog.parquet")) == total
    assert len(c.transport.calls) == 1 + max(1, count_pages)


@pytest.mark.parametrize(
    "last_count,last_pages,last_more", [(31, 2, False), (31, 1, True), (32, 1, False)]
)
def test_web_conflicting_pagination_stays_partial(tmp_path, last_count, last_pages, last_more):
    first = [announcement(str(i)) for i in range(30)]
    path = probe(
        query(),
        client(
            [
                reply(org()),
                reply(page(first, 31, 1, True)),
                reply(page([announcement("last")], last_count, last_pages, last_more)),
            ]
        ),
        tmp_path,
    )
    assert read_json(path / "acquisition_summary.json")["discovery_status"] == "PARTIAL"
    validate_announcement_source(path)


def test_rehashed_exchange_evidence_still_reconstructed(tmp_path):
    path = probe(
        query(),
        client([reply(web_org()), reply(page([announcement(pageColumn="SHKCB")], pages=0))]),
        tmp_path,
    )
    frame = pd.read_parquet(path / "announcement_catalog.parquet")
    frame.loc[0, "exchange_evidence"] = "CODE_PREFIX"
    frame.to_parquet(path / "announcement_catalog.parquet", index=False)
    rehash(path, "announcement_catalog.parquet")
    with pytest.raises(AnnouncementError, match="CATALOG_RECONSTRUCTION"):
        validate_announcement_source(path)


def test_cli_published_failed_discovery_has_nonzero_exit(tmp_path, capsys):
    cache = CaptureCache(tmp_path / "cache", Limits())
    probe(query(), client([reply([])], cache=cache), tmp_path / "sources")
    result = main(
        [
            "data",
            "announcement-cninfo-probe",
            "--ts-code",
            "688598.SH",
            "--start-date",
            "20240701",
            "--end-date",
            "20240705",
            "--max-pages",
            "100",
            "--cache-root",
            str(tmp_path / "cache"),
            "--output-root",
            str(tmp_path / "sources"),
        ]
    )
    assert result == 2
    output = json.loads(capsys.readouterr().out)
    assert output["discovery_status"] == "FAILED"
    validate_announcement_source(Path(output["artifact"]))


def pdf(pages=("Synthetic announcement", "Second page")):
    """Generate a tiny native PDF, including Unicode CMap; no downloaded fixtures."""
    chars = sorted(set("".join(pages)))
    mappings = "\n".join(f"<{ord(c):04x}> <{ord(c):04x}>" for c in chars)
    cmap = (
        "/CIDInit /ProcSet findresource begin 12 dict begin begincmap "
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def "
        "/CMapName /Fixture def /CMapType 2 def 1 begincodespacerange <0000> <FFFF> "
        f"endcodespacerange {len(chars)} beginbfchar\n{mappings}\nendbfchar endcmap "
        "CMapName currentdict /CMap defineresource pop end end"
    ).encode()

    def stream(data):
        return f"<< /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream"

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",
        b"<< /Type /Font /Subtype /Type0 /BaseFont /STSong-Light /Encoding /Identity-H "
        b"/DescendantFonts [4 0 R] /ToUnicode 5 0 R >>",
        b"<< /Type /Font /Subtype /CIDFontType0 /BaseFont /STSong-Light /CIDSystemInfo "
        b"<< /Registry (Adobe) /Ordering (GB1) /Supplement 4 >> /DW 1000 >>",
        stream(cmap),
    ]
    kids = []
    for body in pages:
        number = len(objects) + 1
        kids.append(f"{number} 0 R")
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {number + 1} 0 R >>"
            ).encode()
        )
        content = (
            f"BT /F1 12 Tf 50 750 Td <{body.encode('utf-16-be').hex()}> Tj ET".encode()
            if body
            else b""
        )
        objects.append(stream(content))
    objects[1] = f"<< /Type /Pages /Count {len(pages)} /Kids [{' '.join(kids)}] >>".encode()
    data, offsets = b"%PDF-1.4\n", [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    offset = len(data)
    data += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    data += b"".join(f"{n:010} 00000 n \n".encode() for n in offsets[1:])
    return (
        data
        + f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{offset}\n%%EOF\n".encode()
    )


def discovery(tmp_path, rows=None):
    return probe(
        query(), client([reply(org()), reply(page(rows or [announcement()]))]), tmp_path / "sources"
    )


def acquired(tmp_path, rows=None, documents=None):
    rows = rows or [announcement()]
    source = discovery(tmp_path, rows)
    return acquire(
        source,
        [r["announcementId"] for r in rows],
        client([reply(x) for x in (documents or [pdf()])]),
        tmp_path / "sources",
    )


def rehash(root, name):
    m = read_json(root / "manifest.json")
    m["artifact_hashes"][name] = file_hash(root / name)
    atomic_write_json(root / "manifest.json", m)


@pytest.fixture(
    params=["announcement_source_v1_5692b05.json", "announcement_source_v1_829f2d5.json"]
)
def legacy_source(tmp_path, monkeypatch, request):
    """Run exact frozen producer bytes, not a current artifact relabeled as legacy."""
    frozen = read_json(Path(__file__).parent / "fixtures" / request.param)
    assert frozen["source_commit"] in sources.COMPATIBLE_SOURCE_COMPONENTS
    hashes = {n: digest(s.encode()) for n, s in frozen["source_utf8"].items()}
    assert hashes == frozen["component_sha256"]
    assert payload_hash(hashes) == frozen["implementation_hash"]
    directory = tmp_path / "frozen_producer"
    directory.mkdir()
    for name, source in frozen["source_utf8"].items():
        (directory / name).write_bytes(source.encode())
    with monkeypatch.context() as patch:
        modules = {}
        for name in (
            "announcement_http",
            "announcement_documents",
            "cninfo_announcement_provider",
            "announcement_sources",
        ):
            qualified = "ashare_quant.data." + name
            spec = importlib.util.spec_from_file_location(qualified, directory / (name + ".py"))
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            patch.setitem(sys.modules, qualified, module)
            spec.loader.exec_module(module)
            modules[name] = module
        old = modules["announcement_sources"]
        assert old.implementation_hash() == frozen["implementation_hash"]
        old_http = modules["announcement_http"]
        old_query = modules["cninfo_announcement_provider"].AnnouncementQuery

        def fake_client(replies):
            return old_http.SafeHTTPClient(
                authorized=True, transport=FakeTransport(replies), sleep=lambda _: None
            )

        parent = old.probe(
            old_query(**query().model_dump()),
            fake_client([reply(org()), reply(page([announcement()]))]),
            tmp_path / "legacy",
        )
        artifact = old.acquire(
            parent, ["fixture-1"], fake_client([reply(pdf())]), tmp_path / "legacy"
        )
        old_text = text.publish_text(artifact, tmp_path / "legacy_text")
        frozen["failed_source"] = old.probe(
            old_query(**query().model_dump()),
            fake_client([reply(web_org())]),
            tmp_path / "legacy_failures",
        )
    return parent, artifact, old_text, frozen


def test_v1_failure_stays_frozen_under_v2_validator(legacy_source):
    _, _, _, frozen = legacy_source
    path = frozen["failed_source"]
    before = file_hash(path / "manifest.json")
    sources.validate_announcement_source(path)
    summary = read_json(path / "acquisition_summary.json")
    assert summary["discovery_status"] == "FAILED"
    assert summary["discovery_issues"] == ["ORG_EXCHANGE_UNVERIFIED"]
    assert file_hash(path / "manifest.json") == before


def test_v1_producer_cannot_masquerade_as_v2(legacy_source):
    parent, _, _, _ = legacy_source
    logical = read_json(parent / "manifest.json")["logical_identity"]
    logical["contract"] = sources.CONTRACT
    path = reseal_fixture(parent, logical)
    with pytest.raises(AnnouncementError, match="SOURCE_IMPLEMENTATION_UNSUPPORTED"):
        sources.validate_announcement_source(path)


def test_legacy_real_producer_survives_implementation_change(legacy_source, monkeypatch):
    parent, artifact, old_text, frozen = legacy_source
    before = {
        p: file_hash(p)
        for root in (parent, artifact, old_text)
        for p in root.rglob("*")
        if p.is_file()
    }
    # A nonsemantic source-byte change exercises the defect even before the fix.
    components = dict(frozen["component_sha256"])
    components["announcement_sources.py"] = digest(
        (frozen["source_utf8"]["announcement_sources.py"] + "\n").encode()
    )
    monkeypatch.setattr(sources, "implementation_hash", lambda: payload_hash(components))
    manifest = sources.validate_announcement_source(artifact)
    assert manifest["logical_identity"]["implementation_hash"] == frozen["implementation_hash"]
    text.validate_text_artifact(old_text)
    assert all(file_hash(p) == h for p, h in before.items())


def reseal_fixture(root, logical):
    """Adversarial fixture: repair envelope identity, never a production operation."""
    manifest = read_json(root / "manifest.json")
    manifest["logical_identity"] = logical
    manifest["artifact_id"] = sources.KIND + "_" + payload_hash(logical)[:24]
    atomic_write_json(root / "manifest.json", manifest)
    destination = root.with_name(manifest["artifact_id"])
    if destination != root:
        root.rename(destination)
    return destination


def test_current_and_pinned_producers_use_distinct_unchanged_business(legacy_source, tmp_path):
    parent, artifact, _, frozen = legacy_source
    assert sources.SOURCE_IMPLEMENTATION_COMPATIBILITY_VERSION == 2
    assert (
        sources.COMPATIBLE_SOURCE_COMPONENTS[frozen["source_commit"]] == frozen["component_sha256"]
    )
    assert sources.implementation_hash() != frozen["implementation_hash"]
    assert sources.validate_announcement_source(artifact)["status"] == "COMPLETE"
    current = acquire(parent, ["fixture-1"], client([reply(pdf())]), tmp_path / "current")
    manifest = sources.validate_announcement_source(current)
    assert manifest["logical_identity"]["implementation_hash"] == sources.implementation_hash()
    assert current.name != artifact.name
    old_frame = pd.read_parquet(artifact / "announcement_catalog.parquet")
    current_frame = pd.read_parquet(current / "announcement_catalog.parquet")
    assert current_frame[old_frame.columns].equals(old_frame)
    assert manifest["logical_identity"]["contract"] == sources.CONTRACT
    assert (
        read_json(artifact / "manifest.json")["logical_identity"]["contract"]
        == sources.LEGACY_CONTRACT
    )
    assert acquire(parent, ["fixture-1"], client([reply(pdf())]), tmp_path / "current") == current


@pytest.mark.parametrize("fingerprint", ["f" * 64, "bad", "A" * 64, "0" * 63, None, 42, []])
def test_unsupported_source_producer_rejected(legacy_source, fingerprint):
    parent, _, _, _ = legacy_source
    logical = read_json(parent / "manifest.json")["logical_identity"]
    logical["implementation_hash"] = fingerprint
    path = reseal_fixture(parent, logical)
    with pytest.raises(AnnouncementError, match="SOURCE_IMPLEMENTATION_UNSUPPORTED"):
        sources.validate_announcement_source(path)


@pytest.mark.parametrize(
    "child",
    [
        "announcement_catalog.parquet",
        "document_inventory.json",
        "acquisition_summary.json",
        "requests.json",
    ],
)
def test_legacy_rehashed_children_still_reconstructed(legacy_source, child):
    _, artifact, _, _ = legacy_source
    if child.endswith(".parquet"):
        frame = pd.read_parquet(artifact / child)
        frame.loc[0, "announcement_title"] = "false metadata"
        frame.to_parquet(artifact / child, index=False)
    else:
        obj = read_json(artifact / child)
        obj["invented"] = True
        atomic_write_json(artifact / child, obj)
    rehash(artifact, child)
    with pytest.raises(AnnouncementError, match="RECONSTRUCTION"):
        sources.validate_announcement_source(artifact)


@pytest.mark.parametrize("target", ["parent", "response", "pdf"])
def test_legacy_original_bytes_and_parent_remain_checked(legacy_source, target):
    parent, artifact, _, _ = legacy_source
    if target == "parent":
        child = parent / "acquisition_summary.json"
    else:
        child = next((artifact / ("responses" if target == "response" else "documents")).iterdir())
    child.write_bytes(b"tampered")
    with pytest.raises(DataValidationError, match="CHILD_HASH_MISMATCH"):
        sources.validate_announcement_source(artifact)
    if target == "pdf":
        rehash(artifact, str(child.relative_to(artifact)))
        with pytest.raises(AnnouncementError, match="DOCUMENT_RECONSTRUCTION"):
            sources.validate_announcement_source(artifact)
    elif target == "response":
        rehash(artifact, str(child.relative_to(artifact)))
        with pytest.raises(AnnouncementError, match="RESPONSE_HASH"):
            sources.validate_announcement_source(artifact)


@pytest.mark.parametrize("mutation", ["pagination", "security", "org", "title", "capture"])
def test_legacy_resealed_response_requires_business_reconstruction(legacy_source, mutation):
    parent, _, _, _ = legacy_source
    body = read_json(parent / "responses.json")
    if mutation == "capture":
        body["captures"][1]["request_id"] = "0" * 64
    else:
        index = 0 if mutation == "org" else 1
        attempt = body["captures"][index]["attempts"][-1]
        old_name = "responses/" + attempt["body_sha256"] + ".bin"
        raw = json.loads((parent / old_name).read_bytes())
        if mutation == "org":
            raw[0]["orgId"] = "different-org"
        elif mutation == "pagination":
            raw["totalAnnouncement"] = 2
        elif mutation == "security":
            raw["announcements"][0]["secCode"] = "999999"
        else:
            raw["announcements"][0]["announcementTitle"] = "different title"
        data = json.dumps(raw).encode()
        attempt.update(body_sha256=digest(data), size=len(data))
        new_name = "responses/" + digest(data) + ".bin"
        (parent / new_name).write_bytes(data)
        (parent / old_name).unlink()
        manifest = read_json(parent / "manifest.json")
        del manifest["artifact_hashes"][old_name]
        manifest["artifact_hashes"][new_name] = digest(data)
        atomic_write_json(parent / "manifest.json", manifest)
    atomic_write_json(parent / "responses.json", body)
    rehash(parent, "responses.json")
    logical = read_json(parent / "manifest.json")["logical_identity"]
    logical["acquisition_content_hash"] = payload_hash(body)
    parent = reseal_fixture(parent, logical)
    with pytest.raises(AnnouncementError, match="RECONSTRUCTION|REQUEST_MISMATCH|REQUEST_ID"):
        sources.validate_announcement_source(parent)


def test_legacy_download_request_not_accepted_by_compatibility(legacy_source):
    _, artifact, _, _ = legacy_source
    body = read_json(artifact / "responses.json")
    capture = body["captures"][-1]
    capture["request"] = request_spec("https://static.cninfo.com.cn/unrelated.pdf")
    capture["request_id"] = payload_hash(capture["request"])
    capture["attempts"][-1]["url"] = capture["request"]["url"]
    atomic_write_json(artifact / "responses.json", body)
    rehash(artifact, "responses.json")
    logical = read_json(artifact / "manifest.json")["logical_identity"]
    logical["acquisition_content_hash"] = payload_hash(body)
    artifact = reseal_fixture(artifact, logical)
    with pytest.raises(AnnouncementError, match="DOCUMENT_REQUEST_IDENTITY"):
        sources.validate_announcement_source(artifact)


def test_legacy_different_business_results_fail_closed(legacy_source, monkeypatch):
    _, artifact, _, _ = legacy_source
    original = sources.inspect_pdf

    def changed(data):
        result = original(data)
        result["total_pages"] += 1
        return result

    monkeypatch.setattr(sources, "inspect_pdf", changed)
    with pytest.raises(AnnouncementError, match="BUSINESS_RECONSTRUCTION"):
        sources.validate_announcement_source(artifact)


def test_source_compatibility_does_not_authorize_new_native_tools(legacy_source, monkeypatch):
    _, artifact, old_text, _ = legacy_source
    original = sources.tool_identity
    monkeypatch.setattr(sources, "tool_identity", lambda n: original(n) | {"version": "different"})
    with pytest.raises(AnnouncementError, match="PDF_INSPECTOR_UNSUPPORTED"):
        sources.validate_announcement_source(artifact)
    monkeypatch.setattr(sources, "tool_identity", original)
    parser = text.parser_identity()
    parser["tools"][1]["version"] = "different"
    monkeypatch.setattr(text, "parser_identity", lambda: parser)
    with pytest.raises(AnnouncementError, match="PARSER_ENVIRONMENT_UNSUPPORTED"):
        text.validate_text_artifact(old_text)


def test_legacy_resealed_inspector_identity_rejected(legacy_source):
    _, artifact, _, _ = legacy_source
    logical = read_json(artifact / "manifest.json")["logical_identity"]
    logical["pdf_inspector"]["version"] = "untrusted-version"
    artifact = reseal_fixture(artifact, logical)
    with pytest.raises(AnnouncementError, match="PDF_INSPECTOR_UNSUPPORTED"):
        sources.validate_announcement_source(artifact)


@pytest.mark.parametrize(
    "ts,code,plate",
    [
        ("688598.SH", "688598", "sh"),
        ("001203.SZ", "001203", "sz"),
        ("002865.SZ", "002865", "sz"),
        ("920001.BJ", "920001", "bj"),
    ],
)
def test_explicit_exchange_and_beijing_date(tmp_path, ts, code, plate):
    q = AnnouncementQuery(ts_code=ts, start_date="20240703", end_date="20240703")
    result = CNINFOAnnouncementProvider(
        client([reply(org(code, plate)), reply(page([announcement(secCode=code)]))])
    ).search(q)
    assert result.progress.status == "COMPLETE"
    assert result.progress.records[0]["announcement_date"] == "20240703"
    assert result.progress.records[0]["exchange"] == ts[-2:]
    path = publish_discovery(result, Limits(), tmp_path)
    validate_announcement_source(path)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        org("999999"),
        org(plate="sz"),
        [{"code": "688598", "orgId": "x"}],
        org() + [{"code": "688598", "orgId": "other", "plate": "sh"}],
    ],
)
def test_unproven_org_identity_stops(rows):
    c = client([reply(rows)])
    result = CNINFOAnnouncementProvider(c).search(query())
    assert result.progress.status == "FAILED" and len(c.transport.calls) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ts_code": "688598"},
        {"ts_code": "bad.SH"},
        {"page_size": 31},
        {"start_date": "20240230"},
        {"end_date": "20230101"},
    ],
)
def test_invalid_queries(kwargs):
    with pytest.raises(ValueError):
        AnnouncementQuery(
            **({"ts_code": "688598.SH", "start_date": "20240701", "end_date": "20240705"} | kwargs)
        )


def test_complete_pagination_and_preserve_types(tmp_path):
    rows = [announcement(str(i), announcementType=str(i % 2)) for i in range(31)]
    c = client(
        [reply(org()), reply(page(rows[:30], 31, 2, True)), reply(page(rows[30:], 31, 2, False))]
    )
    source = probe(query(), c, tmp_path)
    frame = pd.read_parquet(source / "announcement_catalog.parquet")
    assert len(frame) == 31 and frame.announcement_type.nunique() == 2
    assert c.transport.calls[-1][2]["pageNum"] == "2"
    assert read_json(source / "acquisition_summary.json")["discovery_status"] == "COMPLETE"


@pytest.mark.parametrize(
    "body",
    [
        page([], 1, 1, True),
        page([announcement()], 2, 1, False),
        page([announcement()], 1, 2, False),
        {"announcements": []},
        page([announcement(), announcement()], 2),
        page([announcement(secCode="000001")]),
        page([announcement(announcementTime=None)]),
        page([announcement()] * 31, 31, 2, True),
        [],
        b"<html>captcha</html>",
    ],
)
def test_schema_pagination_identity_fail_closed(tmp_path, body):
    source = probe(query(), client([reply(org()), reply(body)]), tmp_path)
    assert read_json(source / "acquisition_summary.json")["discovery_status"] != "COMPLETE"
    validate_announcement_source(source)


def test_empty_result_and_fuzzy_keyword(tmp_path):
    empty = probe(query(), client([reply(org()), reply(page([], 0, 0))]), tmp_path)
    summary = read_json(empty / "acquisition_summary.json")
    assert (
        summary["discovery_status"] == "COMPLETE"
        and "NOT_GENERAL_ABSENCE" in summary["empty_result_semantics"]
    )
    source = probe(
        query(keyword="unrelated"), client([reply(org()), reply(page([announcement()]))]), tmp_path
    )
    assert (
        pd.read_parquet(source / "announcement_catalog.parquet").keyword_match.iloc[0]
        == "UNVERIFIED_PROVIDER_MATCH"
    )


def test_interrupted_pagination_checkpoint_and_offline_resume(tmp_path):
    cache = CaptureCache(tmp_path / "cache", Limits())
    first = client(
        [
            reply(org()),
            reply(page([announcement(str(i)) for i in range(30)], 31, 2, True)),
            TimeoutError(),
            TimeoutError(),
            TimeoutError(),
        ],
        cache=cache,
    )
    partial = probe(query(), first, tmp_path / "sources")
    assert read_json(partial / "acquisition_summary.json")["discovery_status"] == "PARTIAL"
    retry = client([reply(page([announcement("last")], 31, 2))], cache=cache, retry_failed=True)
    complete = probe(query(), retry, tmp_path / "sources")
    assert len(retry.transport.calls) == 1 and complete != partial
    offline = SafeHTTPClient(cache=cache, transport=FakeTransport([]))
    assert probe(query(), offline, tmp_path / "sources") == complete
    assert len(list((tmp_path / "cache/captures").glob("*.json"))) >= 5


def test_max_pages_and_inconsistent_totals(tmp_path):
    first = page([announcement(str(i)) for i in range(30)], 31, 2, True)
    capped = probe(query(max_pages=1), client([reply(org()), reply(first)]), tmp_path)
    assert read_json(capped / "acquisition_summary.json")["discovery_issues"] == [
        "MAX_PAGES_REACHED"
    ]
    changed = probe(
        query(),
        client([reply(org()), reply(first), reply(page([announcement("last")], 32, 2))]),
        tmp_path,
    )
    assert (
        "PAGINATION_TOTAL_CHANGED"
        in read_json(changed / "acquisition_summary.json")["discovery_issues"]
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://www.cninfo.com.cn/x",
        "https://evil.cninfo.com.cn/x",
        "https://localhost/x",
        "https://127.0.0.1/x",
        "https://www.cninfo.com.cn:80/x",
        "https://www.cninfo.com.cn@evil.test/x",
    ],
)
def test_url_boundary(url):
    with pytest.raises(AnnouncementError, match="UNSAFE_URL"):
        official_url(url)


@pytest.mark.parametrize(
    "destination", ["http://static.cninfo.com.cn/x", "https://evil.test/x", "https://127.0.0.1/x"]
)
def test_unsafe_redirect_never_requested(destination):
    c = client([reply(b"", 302, location=destination)])
    capture, _ = c.fetch(request_spec("https://static.cninfo.com.cn/start"))
    assert capture["status"] == "REDIRECT_REJECTED" and len(c.transport.calls) == 1


def test_checked_redirect_and_bound():
    c = client([reply(b"", 302, location="/next"), reply(b"pdf")])
    capture, _ = c.fetch(request_spec("https://static.cninfo.com.cn/start"))
    assert (
        capture["status"] == "CAPTURED"
        and capture["attempts"][1]["url"] == "https://static.cninfo.com.cn/next"
    )
    c = client([reply(b"", 302, location="/again")] * 5)
    capture, _ = c.fetch(request_spec("https://static.cninfo.com.cn/start"))
    assert capture["status"] == "REDIRECT_REJECTED" and len(c.transport.calls) == 4


@pytest.mark.parametrize(
    "status,expected",
    [
        (403, "PERMISSION_REQUIRED"),
        (429, "RATE_LIMITED"),
        (503, "SERVER_REJECTED"),
        (404, "HTTP_ERROR"),
    ],
)
def test_rejections_not_retried(status, expected):
    c = client([reply(b"rejected", status)])
    capture, _ = c.fetch(request_spec("https://static.cninfo.com.cn/x"))
    assert capture["status"] == expected and len(c.transport.calls) == 1


def test_authorization_limits_retry_rate_and_dns(monkeypatch):
    fake = FakeTransport([])
    with pytest.raises(AnnouncementError, match="NOT_AUTHORIZED"):
        SafeHTTPClient(transport=fake).fetch(request_spec("https://www.cninfo.com.cn/x"))
    assert not fake.calls
    with pytest.raises(AnnouncementError, match="NOT_AUTHORIZED"):
        http.HTTPSWire().exchange("GET", "https://www.cninfo.com.cn/x", {}, 100, Limits())
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    )
    with pytest.raises(AnnouncementError, match="NON_PUBLIC"):
        http.PublicHTTPSConnection("www.cninfo.com.cn").connect()
    waits = []
    c = SafeHTTPClient(
        authorized=True, transport=FakeTransport([TimeoutError()] * 3), sleep=waits.append
    )
    capture, _ = c.fetch(request_spec("https://www.cninfo.com.cn/x", {}))
    assert capture["status"] == "TIMEOUT" and len(waits) == 3 and waits == [1, 2, 3]
    c = client([reply(b"x" * 129)], policy=Limits(response_bytes=128, pdf_bytes=128))
    assert (
        c.fetch(request_spec("https://www.cninfo.com.cn/x", {}))[0]["status"]
        == "RESPONSE_TOO_LARGE"
    )
    c = client([reply(b"x" * 129)], policy=Limits(pdf_bytes=128))
    assert (
        c.fetch(request_spec("https://static.cninfo.com.cn/x"))[0]["status"] == "RESPONSE_TOO_LARGE"
    )
    c = client([TimeoutError()], policy=Limits(max_requests=1))
    assert c.fetch(request_spec("https://static.cninfo.com.cn/x"))[0]["status"] == "REQUEST_LIMIT"


def test_pdf_identity_dedup_only_by_bytes(tmp_path):
    rows = [announcement("a"), announcement("b")]
    raw = pdf()
    path = acquired(tmp_path, rows, [raw, raw])
    frame = pd.read_parquet(path / "announcement_catalog.parquet")
    assert len(frame) == 2 and frame.document_sha256.nunique() == 1
    assert len(list((path / "documents").glob("*.pdf"))) == 1
    assert (path / "documents" / f"{digest(raw)}.pdf").read_bytes() == raw
    other = acquire(
        discovery(tmp_path, rows),
        ["a", "b"],
        client([reply(raw), reply(pdf(("different",)))]),
        tmp_path / "sources",
    )
    assert pd.read_parquet(other / "announcement_catalog.parquet").document_sha256.nunique() == 2


@pytest.mark.parametrize("bad", [b"<html>not pdf</html>", b"%PDF-1.4\nbroken\n%%EOF"])
def test_invalid_pdf_partial_and_raw_retained(tmp_path, bad):
    path = acquired(tmp_path, [announcement("a"), announcement("b")], [pdf(), bad])
    assert read_json(path / "acquisition_summary.json")["acquisition_status"] == "PARTIAL"
    assert (path / "responses" / f"{digest(bad)}.bin").read_bytes() == bad
    validate_announcement_source(path)


def test_pdf_404_permission_and_not_attempted(tmp_path):
    source = discovery(tmp_path, [announcement("a"), announcement("b")])
    path = acquire(source, ["a", "b"], client([reply(b"no", 403)]), tmp_path / "sources")
    assert (
        read_json(path / "acquisition_summary.json")["acquisition_status"] == "PERMISSION_REQUIRED"
    )
    assert list(pd.read_parquet(path / "announcement_catalog.parquet").fetch_status) == [
        "PERMISSION_REQUIRED",
        "NOT_ATTEMPTED",
    ]
    path = acquire(source, ["a"], client([reply(b"no", 404)]), tmp_path / "sources")
    assert read_json(path / "acquisition_summary.json")["acquisition_status"] == "FAILED"


def test_cache_corruption_cannot_trigger_network(tmp_path):
    cache = CaptureCache(tmp_path / "cache", Limits())
    spec = request_spec("https://static.cninfo.com.cn/x")
    client([reply(pdf())], cache=cache).fetch(spec)
    next((tmp_path / "cache/blobs").iterdir()).write_bytes(b"corrupted")
    c = client([], cache=cache, retry_failed=True)
    with pytest.raises(AnnouncementError, match="RESPONSE_HASH"):
        c.fetch(spec)
    assert not c.transport.calls


def test_page_text_unicode_identity_and_fulltext(tmp_path):
    raw = pdf(("中文公告", "column A   column B"))
    assert inspect_pdf(raw)["status"] == "VALID_PDF"
    source = acquired(tmp_path, documents=[raw])
    source_hash = file_hash(source / "manifest.json")
    path = text.publish_text(source, tmp_path / "text")
    text.validate_text_artifact(path)
    extraction = read_json(path / f"extracted_text/{digest(raw)}/extraction.json")
    assert [p["page_number"] for p in extraction["pages"]] == [1, 2]
    assert "中文公告" in extraction["pages"][0]["text"]
    assert extraction["extraction_status"] == "COMPLETE"
    assert (
        "columns" in extraction["limitations"]
        and extraction["parser_identity"]["config"]["ocr"] is False
    )
    for p in extraction["pages"]:
        assert p["text_sha256"] == digest(p["text"].encode()) and p["char_count"] == len(p["text"])
    assert (path / f"extracted_text/{digest(raw)}/full_text.txt").read_bytes() == text.full_text(
        extraction
    )
    assert text.publish_text(source, tmp_path / "text") == path
    assert file_hash(source / "manifest.json") == source_hash


def test_scanned_and_single_page_failure(monkeypatch):
    raw = pdf(("",))
    result = text.extract_pdf(raw, text.parser_identity())
    assert result["extraction_status"] == "SCANNED_OR_NO_TEXT" and result["extracted_pages"] == 0
    original = text.page_text

    def broken(path, page_number):
        if page_number == 2:
            raise AnnouncementError("PARSER_FAILED")
        return original(path, page_number)

    monkeypatch.setattr(text, "page_text", broken)
    result = text.extract_pdf(pdf(), text.parser_identity())
    assert (
        result["extraction_status"] == "PARTIAL"
        and result["pages"][1]["extraction_status"] == "PAGE_PARSE_FAILED"
    )


def test_parser_upgrade_gets_new_identity(tmp_path, monkeypatch):
    source = acquired(tmp_path)
    old = text.publish_text(source, tmp_path / "texts")
    old_hash = file_hash(old / "manifest.json")
    identity = deepcopy(text.parser_identity())
    identity["tools"][1]["version"] = "fixture-new-parser"
    monkeypatch.setattr(text, "parser_identity", lambda: identity)
    new = text.publish_text(source, tmp_path / "texts")
    assert old != new and file_hash(old / "manifest.json") == old_hash
    with pytest.raises(AnnouncementError, match="PARSER_ENVIRONMENT_UNSUPPORTED"):
        text.validate_text_artifact(old)


@pytest.mark.parametrize(
    "child",
    [
        "announcement_catalog.parquet",
        "acquisition_summary.json",
        "document_inventory.json",
        "requests.json",
    ],
)
def test_source_rehashed_child_business_tampering(tmp_path, child):
    path = acquired(tmp_path)
    if child.endswith(".parquet"):
        frame = pd.read_parquet(path / child)
        frame.loc[0, "announcement_title"] = "tampered"
        frame.to_parquet(path / child, index=False)
    else:
        body = read_json(path / child)
        body["tampered"] = True
        atomic_write_json(path / child, body)
    rehash(path, child)
    with pytest.raises(AnnouncementError, match="RECONSTRUCTION"):
        validate_announcement_source(path)


def test_text_rehashed_child_still_reconstructed(tmp_path):
    source = acquired(tmp_path)
    path = text.publish_text(source, tmp_path / "texts")
    name = next(
        n
        for n in read_json(path / "manifest.json")["artifact_hashes"]
        if n.endswith("extraction.json")
    )
    body = read_json(path / name)
    body["pages"][0]["text"] = "invented text"
    body["pages"][0]["text_sha256"] = digest(b"invented text")
    atomic_write_json(path / name, body)
    rehash(path, name)
    with pytest.raises(AnnouncementError, match="BUSINESS_RECONSTRUCTION"):
        text.validate_text_artifact(path)


def test_offline_cli_no_default_network(tmp_path, capsys):
    args = [
        "data",
        "announcement-cninfo-probe",
        "--ts-code",
        "688598.SH",
        "--start-date",
        "20240701",
        "--end-date",
        "20240705",
        "--output-root",
        str(tmp_path / "sources"),
        "--cache-root",
        str(tmp_path / "cache"),
    ]
    assert main(args) == 2
    assert "NETWORK_NOT_AUTHORIZED" in capsys.readouterr().err
    cache = CaptureCache(tmp_path / "cache", Limits())
    probe(
        query(),
        client([reply(org()), reply(page([announcement()]))], cache=cache),
        tmp_path / "sources",
    )
    assert main(args) == 0
    source = Path(json.loads(capsys.readouterr().out)["artifact"])
    assert main(["data", "announcement-source-validate", "--artifact", str(source)]) == 0


def test_interrupted_capture_can_publish_partial_offline(tmp_path):
    c = client([reply(org())])
    cap, blobs = c.fetch(
        request_spec(
            "https://www.cninfo.com.cn/new/information/topSearch/query",
            {"keyWord": "688598", "maxNum": "10"},
        )
    )
    result = SearchResult(query(), [cap], blobs, analyze_search(query(), [cap], blobs))
    path = publish_discovery(result, Limits(), tmp_path)
    assert read_json(path / "acquisition_summary.json")["discovery_status"] == "PARTIAL"


def test_parser_stderr_cannot_become_official_text():
    with pytest.raises(AnnouncementError, match="PARSER_STDERR"):
        run_native(
            [sys.executable, "-c", "import sys; print('body'); print('warning', file=sys.stderr)"],
            limit=4096,
        )


def test_capture_validator_rechecks_retry_budget():
    c = client([TimeoutError()] * 3)
    capture, blobs = c.fetch(request_spec("https://static.cninfo.com.cn/x"))
    capture["attempts"].append(deepcopy(capture["attempts"][-1]))
    with pytest.raises(AnnouncementError, match="RETRY_LIMIT"):
        http.validate_capture(capture, blobs, Limits())


def test_duplicate_json_keys_not_silently_selected():
    c = client([reply(b'{"code":"wrong","code":"688598"}')])
    capture, blobs = c.fetch(request_spec("https://www.cninfo.com.cn/x", {}))
    with pytest.raises(AnnouncementError, match="JSON"):
        http.response_json(capture, blobs)


@pytest.mark.parametrize(
    "headers,body,expected",
    [
        ({"Content-Length": "200"}, b"xx", "RESPONSE_TOO_LARGE"),
        ({"Content-Length": "4"}, b"xx", "RESPONSE_TRUNCATED"),
        ({"Content-Encoding": "gzip"}, b"xx", "ENCODING_UNSUPPORTED"),
        ({}, b"x" * 129, "RESPONSE_TOO_LARGE"),
    ],
)
def test_real_transport_bounded_reader_without_sockets(monkeypatch, headers, body, expected):
    class FakeSocket:
        def settimeout(self, value):
            assert value > 0

    class Response:
        status = 200

        def __init__(self):
            self.remaining = body

        def getheaders(self):
            return list(headers.items())

        def read1(self, n):
            value, self.remaining = self.remaining[:n], self.remaining[n:]
            return value

    class Connection:
        def __init__(self, *a, **kw):
            self.sock = FakeSocket()

        def connect(self):
            pass

        def request(self, *a, **kw):
            assert kw["headers"]["Accept-Encoding"] == "identity"

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(http, "PublicHTTPSConnection", Connection)
    with pytest.raises(AnnouncementError, match=expected):
        http.HTTPSWire(authorized=True).exchange(
            "GET", "https://static.cninfo.com.cn/x", {}, 128, Limits()
        )


def test_parser_resource_bounds():
    with pytest.raises(AnnouncementError, match="OUTPUT_LIMIT"):
        run_native([sys.executable, "-c", "print('x'*4096)"], limit=128)
    with pytest.raises(AnnouncementError, match="PARSER_TIMEOUT"):
        run_native([sys.executable, "-c", "import time; time.sleep(1)"], limit=128, timeout=0.05)


def test_raw_response_and_pdf_tamper_rejected(tmp_path):
    path = acquired(tmp_path)
    name = next(
        n
        for n in read_json(path / "manifest.json")["artifact_hashes"]
        if n.startswith("documents/")
    )
    (path / name).write_bytes(pdf(("altered",)))
    with pytest.raises(DataValidationError, match="CHILD_HASH"):
        validate_announcement_source(path)
    rehash(path, name)
    with pytest.raises(AnnouncementError, match="DOCUMENT_RECONSTRUCTION"):
        validate_announcement_source(path)


def test_full_text_tamper_rejected_after_rehash(tmp_path):
    path = text.publish_text(acquired(tmp_path), tmp_path / "text")
    name = next(
        n
        for n in read_json(path / "manifest.json")["artifact_hashes"]
        if n.endswith("full_text.txt")
    )
    (path / name).write_text("invented")
    rehash(path, name)
    with pytest.raises(AnnouncementError, match="BUSINESS_RECONSTRUCTION"):
        text.validate_text_artifact(path)


def test_all_cli_stages_offline_cached(tmp_path, capsys):
    cache = CaptureCache(tmp_path / "cache", Limits())
    source = discovery(tmp_path)
    row = announcement()
    client([reply(pdf())], cache=cache).fetch(
        request_spec("https://static.cninfo.com.cn/" + row["adjunctUrl"])
    )
    args = [
        "data",
        "announcement-cninfo-acquire",
        "--source",
        str(source),
        "--source-manifest-sha256",
        file_hash(source / "manifest.json"),
        "--announcement-id",
        "fixture-1",
        "--cache-root",
        str(tmp_path / "cache"),
        "--output-root",
        str(tmp_path / "sources"),
    ]
    assert main(args) == 0
    acquired_path = Path(json.loads(capsys.readouterr().out)["artifact"])
    assert (
        main(
            [
                "data",
                "announcement-text-extract",
                "--source",
                str(acquired_path),
                "--source-manifest-sha256",
                file_hash(acquired_path / "manifest.json"),
                "--output-root",
                str(tmp_path / "text"),
            ]
        )
        == 0
    )
    text_path = Path(json.loads(capsys.readouterr().out)["artifact"])
    assert main(["data", "announcement-text-validate", "--artifact", str(text_path)]) == 0
    assert read_json(next((tmp_path / "cache/worklists").glob("*.json")))[
        "source_manifest_hash"
    ] == file_hash(source / "manifest.json")


def test_duplicate_across_pages_not_complete(tmp_path):
    rows = [announcement(str(i)) for i in range(30)]
    source = probe(
        query(),
        client([reply(org()), reply(page(rows, 31, 2, True)), reply(page([rows[0]], 31, 2))]),
        tmp_path,
    )
    summary = read_json(source / "acquisition_summary.json")
    assert summary["discovery_status"] == "PARTIAL" and summary["discovery_issues"] == [
        "DUPLICATE_ANNOUNCEMENT_ID"
    ]


def test_post_redirect_rewrite_refused():
    c = client([reply(b"", 302, location="/new")])
    capture, _ = c.fetch(request_spec("https://www.cninfo.com.cn/start", {"stock": "fixture"}))
    assert capture["status"] == "REDIRECT_REJECTED" and len(c.transport.calls) == 1


def test_cache_replays_timeout_then_success_within_one_request(tmp_path):
    cache = CaptureCache(tmp_path / "cache", Limits())
    spec = request_spec("https://static.cninfo.com.cn/x")
    live_fixture = client([TimeoutError(), reply(pdf())], cache=cache)
    capture, _ = live_fixture.fetch(spec)
    assert capture["status"] == "CAPTURED"
    offline = SafeHTTPClient(cache=cache, transport=FakeTransport([]))
    replayed, _ = offline.fetch(spec)
    assert replayed == capture and not offline.transport.calls
