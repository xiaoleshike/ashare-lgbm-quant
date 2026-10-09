# Official announcement acquisition

This is an independent discovery, acquisition and native-text evidence layer.
It does not approve economic events, alter security identity mappings, or call
portfolio/accounting services. Source and text outputs always bind
`economic_evidence_approved=false` and never authorize execution.

## Provider and interface

`CNINFOAnnouncementProvider.search(AnnouncementQuery)` implements the small
`AnnouncementProvider` protocol. Its result includes normalized records,
original response bytes, exact request identities and a completeness result.
Using a result list alone would discard the evidence needed to distinguish a
complete search from an interrupted one.

The endpoints are the public website interfaces, without a commercial SLA:

- HTTPS POST `www.cninfo.com.cn/new/information/topSearch/query`
- HTTPS POST `www.cninfo.com.cn/new/hisAnnouncement/query`
- HTTPS GET attachments under `static.cninfo.com.cn`

Interface references were reviewed in
[use_cninfo API documentation](https://github.com/rollysys/use_cninfo/blob/main/docs/api_reference.md)
and the operator's local `cninfo_announcement` implementation. Their observed
30-row limit and field names are a versioned adapter assumption, not a promise
that the remote website will remain stable. A bounded live test on 2026-10-09
established the v2 adapter rules below. No source code from those projects was copied.

Queries require one exact `NNNNNN.SH/SZ/BJ` code and inclusive real calendar
dates in `YYYYMMDD`, with optional keyword/category. Query dates are not inferred
from a clock or from existing directory timestamps. Category is the CNINFO query
code (for example `category_ndbg_szsh`), not an economic-event classification.
Keywords are sent unchanged; a returned title lacking the literal keyword is
retained as `UNVERIFIED_PROVIDER_MATCH`, not silently treated as a literal match.

orgId must match the exact topSearch security code, without prefix or substring
guesses. Explicit returned `plate`/`exchange` (or unambiguous SSE `column`) can
establish exchange. Actual topSearch responses can omit those fields; v2 allows
an exact `category=A股` organization match to remain pending until each returned
announcement proves exchange via `pageColumn`: SZZB/SZCY/SZSM for SZ and
SHZB/SHKCB for SH. Announcement security and orgId must also match. Unknown boards,
contradictory exchanges and ambiguous organizations fail closed. The catalog binds
`exchange_evidence` and `exchange_response_hash` to the actual proving response.
There is no code-prefix inference. An empty window with pending exchange remains
unverified; it cannot establish an exchange or a complete absence observation.
Beijing queries require explicit org exchange evidence; no unobserved board label
is guessed by this adapter.

Every announcement must match the proven security/orgId and requested date
window. `announcementTime` is converted from UTC epoch milliseconds to UTC+08.
Raw title markup is preserved in responses; normalized titles use HTML parsing.
The original source code, canonical requested code, exchange/plate, orgId,
orgId response hash, request ID and raw row hash remain in the catalog.

## Pagination and completeness

Page size is 1..30 (CLI uses 30). V2 accepts the two observed counter conventions:
`totalpages=floor(totalAnnouncement/pageSize)` and the original ceil convention.
The declared value must stay consistent across pages. Actual page progression
is derived from totalAnnouncement and pageSize, never from totalpages alone:
expected rows and hasMore must match each page, and the final unique ID count must
equal the declared total. In the live 31-record sample totalpages=1, hasMore=true
on page 1 and false on page 2; both pages are required. Counts of exactly 30/60
do not manufacture an additional empty page. Page number, declared total,
hasMore, actual row count and unique announcement IDs must all reconcile.
Changing totals, repeated IDs (including identical repeated rows), empty
intermediate pages, missing counters, foreign securities and changed schemas
stop discovery with an explicit issue. Original responses remain available.
The maximum is 100 pages per explicit query; reaching the bound is PARTIAL.

`discovery_status` is COMPLETE, PARTIAL, FAILED or PERMISSION_REQUIRED.
Zero results are complete only with consistent zero counters and hasMore=false;
they mean that this query returned zero, not proof of universal announcement
absence. The website does not provide a transactionally frozen database snapshot;
completeness means the captured pages reconcile under this contract.

`acquisition_status` independently describes the requested document ID worklist:
COMPLETE, PARTIAL, FAILED, PERMISSION_REQUIRED or NOT_ATTEMPTED. Unrequested
documents remain NOT_ATTEMPTED. COMPLETE never implies all documents in an
unrequested larger universe were downloaded. Up to 30 explicitly selected IDs
can be downloaded per invocation. An access/rate/server rejection stops the
remaining worklist, which retains NOT_ATTEMPTED entries.

The shared publication envelope's `status=COMPLETE` only means that the immutable
artifact was completely written and validated. Always inspect discovery and
acquisition status before interpreting source coverage. Probe/acquire CLI output
includes those statuses; exit 0 requires complete discovery (and, for acquire,
complete selected-document acquisition). Incomplete results are still preserved
but return exit 2. A validator may correctly return PASS for a faithfully recorded
FAILED discovery; this does not authorize acquisition from it.

## HTTPS boundary

Only the exact hosts `www.cninfo.com.cn` and `static.cninfo.com.cn` are accepted,
on HTTPS/443 without userinfo or fragments. Redirects are followed manually,
checking every target before requesting it. HTTP downgrade and other hosts are
rejected. DNS answers must all be public addresses; the socket connects to the
checked address directly, while TLS verifies the official hostname. This avoids
a second unvalidated DNS resolution. Environment proxies are not used.

Default bounds: connect 10 seconds, read 30 seconds, response 4 MB, PDF 50 MB,
three redirects, two retries, 200 actual HTTP attempts per client invocation,
and at least one second between requests. Timeout and 500/502/504 can retry with
bounded exponential delay. 401/403, 429 and 503 stop; there is no CAPTCHA solving,
cookie harvesting or anti-bot bypass. Non-JSON access-challenge responses are
frozen and classified as unusable discovery responses. Compressed HTTP payloads
are rejected in v1 (`Accept-Encoding: identity`), avoiding decompression bombs.

`--allow-network` is required to fill a missing cache entry. Import, CLI parsing,
validation and text extraction never authorize network. Existing validated cache
entries can be used offline. `--retry-failed` additionally requires
`--allow-network`; neither switch overrides corrupt-cache validation.

## Original PDF evidence

`official_announcement_source_v2` freezes request descriptions, each captured
response body, normalized catalog, worklist and document inventory. Original PDF
bytes are copied to `documents/<sha256>.pdf`. HTTP status, size, `%PDF-` header,
EOF marker, native parseability, encryption and page-count bounds are checked.
Invalid bytes remain in captured responses; PDF-looking but unparseable originals
are still retained with PDF_PARSE_FAILED. No extension-only success inference.

Multiple announcement IDs pointing to the same bytes retain separate catalog
and inventory entries. The PDF content store coalesces only equal SHA256 bytes.
Titles, file sizes, dates and ID order never select or discard announcements.

An acquire artifact is a new child of a hash-bound discovery artifact. Its
validator recursively checks that parent and requires exact preservation of its
query, limits and search captures. It then replays the same discovery state
machine, verifies HTTP traces, repeats PDF inspection and rebuilds the catalog,
document inventory and summary. Changing a derived child and merely recomputing
its file hash does not bypass reconstruction.

## Page-level text

Separate `official_announcement_text_extraction_v1` artifacts bind the source
manifest, PDF SHA256, parser identity and derived content identity. They do not
change the source acquisition ID or PDF bytes. The native implementation uses
installed Poppler `pdfinfo` and `pdftotext -layout -enc UTF-8`, consistent with the
existing supplement citation extractor. No new Python dependencies are needed;
Poppler is an explicit runtime prerequisite. Development verification used
Poppler 24.02.0. Missing tools fail explicitly.

Each PDF has `extracted_text/<document_sha256>/extraction.json` and `full_text.txt`.
Pages are 1-based and retain text, SHA256, character count, failure code and
extraction status. Per-page parser failures do not erase other pages. Empty or
unusable native text is SCANNED_OR_NO_TEXT, never COMPLETE; mixed failures are
PARTIAL. Parser stderr is not inserted into official text. All pages carry
`layout_status=NATIVE_LAYOUT_UNVERIFIED`: column/table relations, reading order,
headers and footers are not semantically reconstructed. Original whitespace is
retained. No OCR is run.

The convenience full text has `===== PAGE N =====` boundaries and is reconstructed
from the page records. It is not the official original. A future citation must
identify document hash, page, exact quote and extraction identity, and still pass
the separate economic evidence/human review workflow.

Parser identity includes native tool version, binary hashes, fixed options and
implementation hashes. Changed identity publishes a new text artifact without
overwriting old output. Strict revalidation needs the recorded compatible parser
environment; a different environment fails PARSER_ENVIRONMENT_UNSUPPORTED.
There is no arbitrary historical implementation allowlist. Future compatibility
requires an explicit reviewed contract, not relaxing reconstruction.

## Cache and recovery

The mutable, explicit cache root contains content-addressed response blobs,
capture snapshots, request-history journals and hash-bound worklists. Each
complete HTTP response/redirect attempt is checkpointed before continuing.
Previously captured raw responses survive interruption. Request identity binds
method, official URL and exact form. Worklist binds explicit window or source
manifest and document IDs. Recovery uses that identity, not mtime.

CLI invocations take a nonblocking advisory lock on this cache only. A second
collector cannot share it concurrently. Service callers must likewise serialize
access to a shared cache. Successful requests are reused; explicit retry retains
the earlier failure snapshots. Corruption fails closed and never triggers an
automatic online repair. An operator can authorize reacquisition into a fresh
cache after investigating the mismatch.

Resume granularity is a complete HTTP request, not byte-range PDF transfer.
An interrupted in-flight response may need redownloading after separate network
authorization; completed pages/documents are reused. No HTTP Range/ETag resume,
full-market scheduler or rolling watermark is implemented. Future incremental
work can supply an explicit new window and publish a new content identity.

Immutable publication reuses the repository staging, manifest-last validation
and atomic rename helper. Hashes bind all children. Current source producer
implementation and PDF inspector identity must be supported; unknown producer
changes cannot silently validate old artifacts under different semantics.
No retrieval time, host, PID, mtime or absolute path enters logical identity.

## Source producer compatibility

`SOURCE_IMPLEMENTATION_COMPATIBILITY_VERSION = 2` dispatches reconstruction by
the explicit source contract. New publication uses official_announcement_source_v2
and binds current component bytes. V2 does not accept v1 producer hashes. V1
reading retains its strict org exchange and ceil-only pagination rules, original
catalog columns and original summaries; old FAILED results are not upgraded.
V1 accepts only the following pinned producers, with no manifest modification:

```text
implementation: 2044ac3cefd69cf586718e96096c3a6e2a2d9da0f4296b63e41cff71d3e36543
announcement_sources.py: ac1fcfbfd47d131b75d547918bdd16fe3b918588c1828f95b5d2e511677b2478
announcement_http.py: c3e6173b66509dadcbb90d696d4c9b18ba09d5dd5c474230f2008682f1101f9a
cninfo_announcement_provider.py: 7ca9f936d6db591d7a4c1098f117bec00ef16d07a224b82874276da88b78d62f
announcement_documents.py: 5071c5aa5bdcb4ded36993e40edce7bd3e6c1dbd2c501e9a07477cb4c2b01ef9

commit: 829f2d52d77217753be3ee47b87743da8bb43614
implementation: 42eff8cd0c14d0d014b80a979eec784372f84f0d5a1733b95953adeaee85f4b6
announcement_sources.py: 9edb5617a0577051e640e0612f6ad64e7d47384b5679ac94ae93e7af3e91d2a5
other three component hashes: unchanged from the preceding producer
```

The aggregate uses the existing canonical `payload_hash(component_sha256)` rule.
The test-only JSON fixtures freeze these exact UTF-8 source bytes and provenance.
Tests execute the old producer in a temporary module context to publish synthetic
discovery, acquisition and text artifacts, then validate them with current code.
Production never loads or executes those fixtures. V1 and v2 use explicitly
different org/pagination rules. HTTP and PDF processing remain unchanged.
Future compatibility entries require explicit source review and regression proof;
unknown or malformed hashes fail `SOURCE_IMPLEMENTATION_UNSUPPORTED`.

Accepting a producer digest permits reconstruction, not automatic acceptance of
its historical results. The original producer hash is retained, and every other
identity field must equal the current reconstruction. Recursive parent validation,
request/capture hashes, orgId/security, pagination, metadata, document selection,
GET identity, PDF bytes/inspection, catalog, inventory, summary and child hashes
remain mandatory. Even a repaired child hash cannot bypass reconstruction. A
different current business result fails closed; it requires a separate semantic
compatibility review, not editing the frozen artifact or broadening this list.

PDF inspector identity is a separate gate: a differing recorded/current Poppler
identity fails `PDF_INSPECTOR_UNSUPPORTED`, even if the producer is compatible.
Changed inspection output still fails business reconstruction. Text extraction
has no new compatibility allowance: its exact parser identity remains required
and mismatches fail `PARSER_ENVIRONMENT_UNSUPPORTED`. A text artifact can reference
a compatible legacy source without changing its own identity. No validator uses
the network, rewrites old children, or silently substitutes newer parser results.

## Operator workflow

After code review and a separate explicit authorization for one small live query:

```bash
cd /home/zhangkangle/workspaces/ashare-lgbm-quant
.venv/bin/ashare-quant --config config/default.yaml data announcement-cninfo-probe \
  --ts-code 688598.SH --start-date 20240627 --end-date 20240627 \
  --max-pages 2 \
  --cache-root data/research/announcement_cache/manual_sample_1 \
  --output-root reports/official_announcements/manual_sample_1 \
  --allow-network
```

Inspect the returned artifact and acquisition summary. Select explicit
announcement IDs from its catalog for a separately authorized PDF download:

```text
ashare-quant data announcement-cninfo-acquire
  --source DISCOVERY_ARTIFACT --source-manifest-sha256 EXACT_MANIFEST_HASH
  --announcement-id EXACT_CATALOG_ID
  --cache-root EXPLICIT_CACHE --output-root ISOLATED_OUTPUT --allow-network
```

Those capitalized values are intentionally unresolved operational inputs; use the
actual probe output, never a guessed/latest directory. No live artifact exists
from this implementation task. Offline operations require no network flag:

```text
ashare-quant data announcement-source-validate --artifact SOURCE_ARTIFACT
ashare-quant data announcement-text-extract
  --source SOURCE_ARTIFACT --source-manifest-sha256 EXACT_MANIFEST_HASH
  --output-root ISOLATED_TEXT_OUTPUT
ashare-quant data announcement-text-validate --artifact TEXT_ARTIFACT
```

The acquisition layer has no economic approval command. It does not establish
tax policy, entitlements, terminal recovery, execution support or production
readiness. Full-text indexing, OCR, batch collection, other providers and
economic review adapters remain separate work.
