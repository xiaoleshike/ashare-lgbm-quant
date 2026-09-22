# Data Ingestion and Storage

Phase 1 implements the raw Tushare ingestion boundary. Unit tests use mocked API
responses only; real API checks must be marked with `@pytest.mark.integration`.

## Raw Identity and Schema Stability

Raw Tushare Parquet preserves provider `ts_code` values. Security aliases are
canonicalized only when data enters universe, feature, label, or executable research
joins. This preserves historical trading-code lineage while providing one stable
cross-source security identity. The checked-in mapping is sourced from the complete
official Beijing Stock Exchange new-old code table and carries a version plus a
content hash in every processed artifact identity.

`suspend_d.suspend_timing` and `suspend_d.suspend_type` are coerced to nullable
string dtype before future Parquet writes. An all-null monthly partition therefore
does not receive Parquet `null` physical type while another partition uses string.
Existing raw partitions are not rewritten automatically; historical repair remains
a separate reviewed migration.

Run the read-only alias consistency scan with:

```bash
ashare-quant --config config/default.yaml data security-identity-scan \
  --processed-root data/research/RESEARCH_SNAPSHOT \
  --start-date YYYYMMDD --end-date YYYYMMDD
```

Security identity and security lifecycle are separate controls. After alias validation, run the
offline lifecycle completeness audit against the exact processed research snapshot:

```bash
ashare-quant --config config/default.yaml \
  data --storage-root data_on_sata/parquet security-lifecycle-scan \
  --start-date 20100101 \
  --end-date 20260710 \
  --processed-root "$RESEARCH_PROCESSED" \
  --reports-root "$RESEARCH_REPORTS" \
  --lifecycle-policy config/security_identity/security_lifecycle_policy.json \
  --lifecycle-evidence config/security_identity/security_lifecycle_events.json
```

The command is deterministic, offline, and read-only for raw and processed data. It publishes an
immutable report under `reports_root/security_lifecycle/<scan_id>/`. A missing quote is only an
investigation candidate: it does not prove suspension or delisting. Ordinary suspension requires
`suspend_d`; historical listing suspension requires frozen verified evidence; terminal state
requires `stock_basic.delist_date`. Whole-market raw-data loss and unresolved security-specific
gaps remain blocking even though they have been classified.

`suspend_d` is interpreted as daily suspension/resumption evidence. An untimed `S` proves only
that session's full-day suspension; it does not open a persistent state until a later `R`.
Consecutive open sessions with explicit untimed `S` snapshots are compressed into one reporting
interval. A timed `S` is intraday evidence and cannot explain an entirely missing daily quote.
`R` is resume-day boundary evidence: a missing local S/R pair is diagnostic unless an actual quote
gap remains unexplained. Same-day S/R records are evaluated with timing, quote, universe, and
stronger lifecycle evidence and are not policy collisions merely because both row types exist.

The project-wide terminal contract treats `stock_basic.delist_date` as the first non-listed date:
UniverseBuilder uses `trade_date < delist_date`, the execution loader forces `is_listed=false` for
`trade_date >= delist_date`, and terminal handling additionally requires explicit delist metadata.
The last observed quote never supplies or changes this date.

The configured v3 event file is a known verified evidence set, not a claim of complete historical
coverage. Rules live separately in `security_lifecycle_policy.json`; adding verified stock-level
events requires a new append-only evidence version rather than rewriting v3.

Exchange ticker aliases and corporate code transitions are also distinct. The BSE alias artifact
normalizes alternate provider keys for one observation. A reviewed code transition has a
predecessor, successor, effective date, continuity type, and official evidence package. It is
published under the research reports root as `security_identity_transitions_v1`; SH/SZ corporate
events must not be appended to the BSE alias file. When supplied with
`--identity-transition-artifact`, lifecycle scanner schema v4 stops expecting predecessor-code
quotes on and after the verified effective date and binds the transition artifact hash into the
scan identity. Quote history is only a consistency check and never establishes the effective date.

When a blocked scan has been triaged, source completeness is investigated separately from lifecycle
truth. The explicit networked probe is the only lifecycle command that contacts Tushare; it freezes
current `suspend_d` and `daily` responses under the research reports root and never writes the raw
store:

```bash
ashare-quant --config config/default.yaml \
  data --storage-root data_on_sata/parquet security-lifecycle-source-probe \
  --triage-manifest "$TRIAGE_MANIFEST" \
  --reports-root "$RESEARCH_REPORTS"
```

`LOCAL_SUSPEND_D_INCOMPLETE` and `LOCAL_DAILY_INCOMPLETE` prove a provider/local snapshot
difference, but do not repair it. `PROVIDER_HAS_NO_SUSPEND_EVIDENCE` does not prove that a missing
quote was a suspension. Official exchange or issuer documents are frozen and hash-validated in a
separate evidence package. `security-lifecycle-resolve` reconciles the exact parent interval IDs
from triage with provider and verified official evidence; it emits repair and evidence plans but
does not update raw data, lifecycle catalogs, Universe, or model artifacts. Provider request times,
absolute paths, hostnames, tokens, and error text are excluded from logical evidence identity.

D.3.1 keeps provider completeness and lifecycle truth in separate fields. In particular,
`source_resolution=PROVIDER_HAS_NO_SUSPEND_EVIDENCE` must retain
`lifecycle_resolution=STILL_UNRESOLVED` until an official lifecycle fact covers the interval.
Partially covered parent intervals are expanded over `trade_cal`, resolved session by session, and
then compressed into deterministic child segments. Parent identity is retained and child sessions
must reconcile exactly, so one repaired date cannot incorrectly resolve an entire interval.

Official Factbooks and exchange lists can be frozen once and normalized into a reusable research
index. These commands write only below the selected reports root:

```bash
ashare-quant --config config/default.yaml data security-lifecycle-bulk-source-freeze \
  --source-json SOURCE.json --records-json NORMALIZED_ROWS.json \
  --document OFFICIAL_DOCUMENT.pdf --reports-root "$RESEARCH_REPORTS"

ashare-quant --config config/default.yaml data security-lifecycle-official-index \
  --reports-root "$RESEARCH_REPORTS" \
  --bulk-source-package BULK_PACKAGE \
  --official-evidence-package VERIFIED_EVIDENCE_PACKAGE

ashare-quant --config config/default.yaml \
  data --storage-root data_on_sata/parquet security-lifecycle-resolution-harden \
  --triage-manifest "$TRIAGE_MANIFEST" \
  --provider-probe-manifest "$SOURCE_PROBE_MANIFEST" \
  --d3-resolution-manifest "$D3_RESOLUTION_MANIFEST" \
  --official-index-manifest "$OFFICIAL_INDEX_MANIFEST" \
  --reports-root "$RESEARCH_REPORTS"
```

Only official-domain documents with an exact security and effective-date match can resolve a
segment. Discovery links and heuristic matches remain unverified. Bulk source packages, the
official index, and hardened resolution are immutable manifest-last artifacts; none of them mutates
canonical raw data or the processed universe.

Formal lifecycle evidence may start before the governed research period. The official index marks
such records as `carry_in=true`; a verified pre-period suspension start can establish the state at
`20100101`, but only a verified resumption or terminal event closes it. A quote reappearing is a
consistency observation, not a substitute for that closing evidence.

D.3.2 closes newly indexed evidence over the exact immutable D.3.1 unresolved queue:

```bash
ashare-quant --config config/default.yaml \
  data --storage-root data_on_sata/parquet security-lifecycle-evidence-close \
  --baseline-hardened-manifest BASELINE_HARDENED_MANIFEST \
  --current-hardened-manifest CURRENT_HARDENED_MANIFEST \
  --official-index-manifest OFFICIAL_INDEX_MANIFEST \
  --triage-manifest TRIAGE_MANIFEST \
  --reports-root "$RESEARCH_REPORTS"
```

Every original work-queue session must map exactly once to a current evidence child. The published
H5 reachability view covers only remaining blocking children and is informational; it never relaxes
the global lifecycle gate and does not read labels, returns, or the possibly stale model-universe
flag.

Before D.4 repairs or rebuilds, raw inputs must be materialized as a content-defined
`ResearchSourceSnapshot`. The contract uses a staging physical copy plus atomic rename and
manifest-last publication. Identity binds relative file names, content SHA256 values, dataset
fingerprints and date coverage, plus security-identity and lifecycle-evidence hashes. A manifest
that points back to mutable production files is not a reproducible snapshot. The current dependency
set is declared in `config/research_source_snapshot_contract.json`; it is derived from the actual
Universe, Feature and Label builders rather than including every configured ingestion endpoint.

The scan reports configured and observed aliases and unresolved mapped keys, and it
fails closed on canonical collisions. It does not infer unknown aliases.

## Token Handling

`TUSHARE_TOKEN` is read only from the process environment by `load_settings()`.
Do not place real tokens in YAML, `.env.example`, tests, or committed files.

## Canonical Parquet Layout

Raw datasets are stored under `paths.parquet_store`, defaulting to `data/parquet`.
Reference snapshots use:

```text
data/parquet/stock_basic/snapshot=latest/data.parquet
```

Date-based datasets use month partitions:

```text
data/parquet/daily/year=2024/month=01/data.parquet
data/parquet/trade_cal/year=2024/month=01/data.parquet
```

`trade_cal` is the authoritative calendar. Daily equity endpoints are downloaded
by open trading day from `trade_cal`; `index_daily` is downloaded by configured
index codes and date range.

## Idempotency and Resume

Every write reads the existing partition, appends the newly downloaded rows,
drops duplicates by the configured primary key, sorts by key, and atomically
replaces the partition file. Re-running the same command for the same date range
therefore produces stable data instead of duplicate rows. Trade-date datasets are
written after each successful trading-day request, and date-range datasets are
chunked by calendar year before writing. If a process stops mid-run, completed
partitions remain valid and the next run resumes by merging or filling the
remaining dates.

## Validation

`ashare-quant data validate` checks required columns, primary-key uniqueness,
duplicate rows, and missing open trading days for datasets that should have data
on every open day. `suspend_d` is allowed to be empty on open days because no
suspension events may occur.

## Permission Handling

Tushare account permissions can differ by endpoint. Permission failures are
reported as clear diagnostics and the affected dataset is skipped safely; other
configured datasets can continue.

## Extended Tushare Datasets

The initial default dataset set remains conservative: stock list, trading calendar,
stock daily prices, adjustment factors, daily basic indicators, suspensions, price
limits, and selected index daily bars. This prevents an accidental `data init` from
starting a very large multi-domain download.

Additional configured datasets can be selected explicitly with repeated `--dataset`
arguments, or all configured datasets can be requested with `--all-datasets`.
Examples:

```bash
ashare-quant data init --dataset fund_basic --dataset fund_daily --start-date 20240101
ashare-quant data init --dataset income --dataset balancesheet --start-date 20200101
ashare-quant data init --all-datasets --start-date 20200101
```

Currently configured extended datasets include:

- ETF and fund data: `fund_basic`, `fund_daily`.
- Options: `opt_basic`.
- Low-frequency quotes: `weekly`, `monthly`.
- Financial statements and forecasts: `income`, `balancesheet`, `cashflow`,
  `fina_indicator`, `forecast`, `express`.
- ST and connect references: `namechange`, `hs_const`.
- Reference data: `pledge_stat`, `pledge_detail`, `share_float`, `repurchase`,
  `stk_holdertrade`, `top_list`, `top_inst`, `margin`, `margin_detail`.
- Special datasets: `concept`, `concept_detail`, `moneyflow`, `moneyflow_hsgt`,
  `broker_recommend`, `cyq_chips`, `cyq_perf`, `stk_factor`.
- Macro data: `cn_gdp`, `cn_cpi`, `cn_ppi`, `cn_m`.

Tushare permissions and fields vary by account and endpoint. Permission errors are
reported per dataset and skipped safely. If Tushare changes endpoint fields, update
`src/ashare_quant/data/datasets.py` before running large downloads.

The global request ceiling is configured by `data.rate_limit_per_minute`. Endpoints
with stricter service-specific limits use `data.endpoint_rate_limits_per_minute`;
`cyq_chips` is paced at 200 requests per minute independently of the global ceiling.


### `cyq_chips` row-limit protection

Tushare requires `ts_code` for `cyq_chips` and returns at most 6000 rows per
request. A response at that exact limit is treated as potentially truncated. The
ingestion service recursively splits the requested range at authoritative open
trading dates until every accepted response contains fewer than 6000 rows. If a
single trading day still reaches the limit, ingestion fails explicitly rather than
storing data whose completeness cannot be guaranteed.

Repair historical coverage idempotently with:

```bash
ashare-quant --config config/default.yaml data init \
  --dataset cyq_chips --start-date 20180101
```

Existing rows are merged by `(ts_code, trade_date, price)`; the repair does not
create duplicate canonical rows.

## Request Granularity

- `stock_basic` is requested for all configured listing statuses: `L`, `D`, and `P`,
  so historical universes are not built only from currently listed stocks.
- Trade-date datasets such as `daily`, `adj_factor`, `daily_basic`, `margin_detail`,
  and `moneyflow` are requested one open trading day at a time using `trade_cal`.
- Date-range reference datasets are split into calendar-year chunks to reduce
  timeout and row-limit risk.
- Financial datasets use the account's VIP endpoints when available. They are
  queried by report quarter and paginated with `limit`/`offset`; this replaces
  thousands of per-stock requests with a small number of cross-sectional calls.
- If a VIP financial endpoint is unavailable, ingestion falls back to the ordinary
  per-stock endpoint and passes `start_date`/`end_date` to the server. The fallback
  is correct but materially slower.
- Snapshot datasets without a date column are skipped during `data update` once a
  local snapshot exists; refresh them explicitly with `data init --dataset ...`.


## Data Quality Error Logs

Data ingestion commands append structured JSONL records to:

```text
logs/data_quality/YYYYMMDD/errors.jsonl
```

The log captures ingestion failures, post-ingestion validation warnings/errors,
and issues found by non-blocking cross-source checks. Raw Parquet data is not
rewritten by these checks. Any downstream cleaning must be explicit and
reproducible.

After each successful `data init` or `data update`, the CLI always runs local
validation. The optional background BaoStock comparison is controlled by
`data.run_baostock_post_ingestion_check` and is disabled by default while the
BaoStock service is unreliable. When enabled, the CLI launches
`scripts/data_checks/run_baostock_previous_day_check.py` without blocking
ingestion and logs only failures or mismatches. The comparison can still be run
manually while automatic execution is disabled.

## Raw OHLC Reliability Rule

Tushare raw `daily` rows can contain source-side OHLC inconsistencies, especially
in historical pre-BSE data mapped to current BJ codes. Keep raw data unchanged,
but apply this cleaning rule before features, labels, or backtests:

- If `trade_date < 20200101` and `high < max(open, close)`, `low > min(open, close)`,
  or `high < low`, exclude the row from research datasets.
- If `trade_date >= 20200101` and the same condition occurs, keep the raw row but
  mark it unavailable for modelling and trading logic.

This rule prevents invalid high/low values from contaminating volatility,
tradability, execution, and limit-price assumptions while preserving the canonical
raw Tushare record for audit.


## Limit Price Special Values

Keep raw `stk_limit` values unchanged, but downstream tradability logic must not
interpret special sentinel values as executable prices:

- `up_limit=99999.99` and `down_limit=0` means no price-limit bound for that
  stock-date, commonly BSE listing days or other no-limit trading sessions.
- `up_limit=0` and `down_limit=0` on rows matched by `suspend_d` means the stock
  was suspended and has no executable limit price for that date.

Feature generation and backtests should convert both cases into explicit
tradability flags instead of numeric limit prices.


## Rolling Revision Backfill

Some Tushare endpoints publish late or revise rows after the first successful
response for a trade date. Incremental updates therefore re-fetch the most recent
five open trading days for revision-prone trade-date datasets and rely on
primary-key idempotent Parquet writes to merge corrections without duplicates.

Current rolling-backfill datasets are `top_list`, `top_inst`, `margin`,
`margin_detail`, `moneyflow`, `moneyflow_hsgt`, `cyq_chips`, `cyq_perf`, and
`stk_factor`. This fixes cases where a date exists locally but Tushare later
adds additional rows for that same date.

`top_list` uses `(ts_code, trade_date, reason)` as its canonical primary key
because the same stock can appear on the same trade date for multiple
Longhubang reasons. Fully duplicated source rows are collapsed during the
idempotent Parquet merge.

## Financial Statement Batching

`income`, `balancesheet`, `cashflow`, `fina_indicator`, `forecast`, and `express`
are downloaded through their corresponding `*_vip` endpoints by report period.
Pages contain at most `data.tushare_page_size` rows (default 6000) and are written
immediately, so a stopped process can safely resume through idempotent merges.

Financial incremental updates re-fetch announcements within
`data.finance_revision_lookback_days` (default 550 days). Report periods are
enumerated for an additional 550-day lookback because annual reports and later
corrections can be announced well after the period end. This bounded revision
window avoids scanning every stock's complete history on every daily update.

Income, balance-sheet, and cash-flow keys include `f_ann_date` and `update_flag`;
forecast and express keys also include `update_flag`. Both original and updated
versions therefore remain auditable. Exact duplicate source rows are collapsed.
Downstream point-in-time joins must select the version available as of the research
date and must not backfill later revisions into earlier dates.

Stores created before `update_flag` became part of the financial primary keys may
have retained only one version of older revised reports. The rolling update repairs
the configured 550-day window. To reconstruct all historical versions, run one
idempotent financial re-initialization during a maintenance window:

```bash
ashare-quant data init \
  --dataset income --dataset balancesheet --dataset cashflow \
  --dataset fina_indicator --dataset forecast --dataset express \
  --start-date 20100101
```

Use the read-only benchmark before changing request granularity:

```bash
python scripts/experiments/tushare_batch_probe.py \
  --start-date 20260708 --end-date 20260709 \
  --finance-period 20260331 --finance-endpoint cashflow_vip
```

## Lifecycle Residual Triage And Universe Overlay

`security-lifecycle-triage` consumes one immutable blocked lifecycle scan and publishes
diagnostic candidate groups. A triage label such as `LIKELY_FORMAL_LISTING_SUSPENSION` is not
authoritative evidence and cannot make the lifecycle preflight pass. The offline scanner remains
the sole classifier; official exchange or issuer evidence must be frozen separately before it can
change lifecycle state.

New universe builds expose a precedence-ordered lifecycle overlay:

```text
TERMINAL > LISTING_SUSPENSION > ORDINARY_SUSPENSION > ACTIVE
```

`suspend_d` contributes only explicit untimed full-day `S` snapshots to ordinary suspension.
The versioned lifecycle evidence catalog contributes formal listing-suspension intervals.
`stock_basic.delist_date` starts terminal state on that date under the existing project contract.
Formal listing suspension remains inside the known listed lifecycle (`is_listed=true`) but is not
tradable and is excluded from the model universe. It is not terminal state.

Run residual triage without modifying source data:

```bash
ashare-quant --config config/default.yaml data security-lifecycle-triage \
  --processed-root "$RESEARCH_PROCESSED" \
  --reports-root "$RESEARCH_REPORTS" \
  --lifecycle-scan-manifest "$LIFECYCLE_SCAN_MANIFEST"
```

When a corrected lifecycle overlay changes `universe_daily`, the old universe-dependent research
lineage is not current governed evidence. The dependency chain is:

```text
universe_daily
  -> features_daily + labels_forward
  -> diagnostics
  -> governed feature-set provenance
  -> walk-forward plan
  -> horizon plan
  -> executable walk-forward evidence
```

All prior artifacts remain immutable historical evidence. Rebuild into a new isolated processed
and reports root; never rewrite the old lineage or alter hashes to force compatibility.
### Typed runtime evidence

`security-lifecycle-catalog-compile` converts only `VERIFIED` official-index rows into an
immutable, explicitly `partial` runtime catalog. Ordinary suspension and formal listing
suspension remain distinct. Terminal metadata continues to come from `stock_basic.delist_date`,
and code transitions remain in the separate identity-transition artifact. A partial catalog
does not imply full-market lifecycle PASS.

Real stock-level evidence files and official packages are intentionally local and ignored by
Git. A fresh checkout must receive explicit base-catalog, official-index, and transition-artifact
paths. Missing files fail closed; an empty catalog is not treated as complete evidence.

### Coherent research snapshots

Snapshot schema v2 captures enabled dependencies while holding the same configured production
writer lock (`paths.runs/.production.lock`). It compares the complete source inventory before
and after copying, records a generation proof, then independently validates copied bytes. A
research-only lock cannot establish source consistency.
