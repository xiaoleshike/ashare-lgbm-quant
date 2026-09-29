# Economic entitlement evidence (D.8C / P0-D1)

This is an evidence contract, **not an execution policy**. Accounting schema 3,
`corporate_action_execution_policy_v1`, raw-price fills, terminal writeoffs and
float-share sizing are unchanged. No model or replay is run by these commands.
Economic evidence must never be consumed by features, predictions or trade decisions.

## Inputs and identities

The operator supplies exact continuous and accounting-audit manifest hashes. The
existing `validate_continuous_strict_oos_artifact()` recursively validates the replay.
The audit loader validates output/input/script hashes, position membership against
the actual buy/close ledger, target dates against holdings, and adjustment-detail
membership against the audit's crossing table. It does not select another run or
recompute membership from other predictions. Tiny positions are retained.

`exposure_inventory.parquet` has one row per account-position adjustment boundary or
terminal event. `economic_boundary_id` hashes only event category, canonical code and
candidate date, independently of Top-N; `exposure_id` additionally binds portfolio and
position. The root identity binds the complete inventory hash. This is a candidate
boundary, not an inferred dividend, split or recovery.

The read-only preflight for the specified D.8B audit reports 792 exposure rows:
770 adjustment observations and 22 terminal account events, 517 unique boundaries
and 325 securities. These are acceptance observations, not constants in core code.

## Provider source contract

The explicit `economic-dividend-probe` command is the only networked entry point.
It reuses `TushareClient` retries, pacing and permission handling, querying historical
`dividend` records only for the exposed codes and validated BSE source aliases.
Code-change continuity does not authorize arbitrary historical code aliasing.
Requests, returned column names, ordered primitive response rows, row hashes and
request statuses are frozen in a separate `economic_event_source_*` artifact.
Raw nulls, proposal rows and extra returned fields are preserved. The original
response is the SDK DataFrame, not a claim to capture HTTP wire bytes.

Reference: [official Tushare dividend contract](https://tushare.pro/document/2?doc_id=103).
The documented per-call ceiling is 2,000 rows; reaching it is
`POSSIBLY_TRUNCATED`, not complete evidence. Empty responses, incomplete schemas
and permissions/errors also block completeness. They never prove no entitlement.
Exception text is not persisted because it could contain credentials.

Only the documented `div_proc=实施` state becomes a candidate. It is **not approved
automatically**. `dividend_review_template.json` contains PENDING row hashes.
Copy it outside the immutable source artifact to prepare a review: each decision
must be APPROVED/REJECTED/PENDING, tied to the exact source ID and raw-row hash, with
reviewer and reviewed fact. An omitted review leaves all candidates unreviewed.
`dividend_candidates.parquet` retains implemented candidates and diagnostics;
`normalized_corporate_actions.parquet` contains only reviewed, supported-component
events with valid effective terms inside the governed execution support.
Review never authorizes account mutation. Conflicting implemented economics cannot
be resolved by choosing the latest row or closest factor ratio.

`cash_div` and `cash_div_tax` are preserved as after-tax and before-tax per-share
fields respectively. Both remain evidence, with
`cash_tax_semantics=UNRESOLVED_EXECUTION_POLICY`. Missing economic components remain
null and ambiguous, never zero. Bonus, reserve conversion and total distribution
are separate fields; inconsistent totals fail closed. Record/ex/pay/share-listing
dates are distinct real dates. Missing or contradictory dates are diagnostics.

## Reconciliation and ownership

Forward matching uses canonical code and exact ex-date only. A missing ex-date or
a possible boundary between sparse factor observations is an explicit date conflict,
not a nearest-date match. Unmatched changes remain `NO_EVENT_FOUND`; factor ratios
never backsolve economics. Unsupported economics and conflicting revisions remain
explicit. All raw observations are retained even if identical event records deduplicate.

Reverse coverage examines every normalized nonzero event for the queried securities
against **all** actual positions, including positions with no observed factor change.
Record-date eligibility is based on the immutable **end-of-day holdings** table.
Entry on ex-date is not record-date ownership. Selling after record date does not
erase a later payable entitlement. Missing record date, dates outside observed
history, non-session record dates and inconsistent holdings are separate states.
This does not claim market-wide event discovery for unqueried securities.

Effective ex-dates after the governed execution cutoff remain raw/candidate evidence
but are excluded from activation and portfolio coverage. Later publication/payment
dates are retained without bringing later prices into this analysis. Every artifact
is bound to `POST_HOC_ACCOUNTING_EVIDENCE`; no signal-time authorization is implied.
The account's actual resolution end and the governed support cutoff are separate:
an ex-date inside support can still reveal an entitlement established on a record
date before the account closed. Missing ownership history is not synthesized.

## Terminal evidence

`terminal-economic-evidence-publish` copies local reviewed document bytes into an
append-only content-addressed package. No document is fetched automatically.
`records.json` uses `terminal_economic_event_v1` and strict Pydantic records.
Required provenance includes source type, URL/locator, document ID, SHA256,
page/paragraph locator, reviewed fact, reviewer and VERIFIED review status.
`evidence_scope` must be `ECONOMIC_ENTITLEMENT`, never listing-only evidence.

Allowed economic classifications are cash settlement, successor share entitlement,
transferred security right, liquidation receivable, proven zero recovery and unknown.
Cash settlement needs an amount and settlement date; successor shares need identified
security, positive ratio and settlement date. Non-cash rights need explicit terms
and effective date. A known receivable identifies a right, **not its fair value or
guaranteed recovery**. Proven zero requires explicit reviewed zero-recovery evidence,
zero cash, no other entitlement and effective date. A terminal date alone remains unknown.
Conflicting packages cannot silently supersede one another.

Example review shape (synthetic values, never publish as real evidence):

```json
{
  "schema": "terminal_economic_event_v1",
  "records": [{
    "ts_code": "AAA.SZ", "terminal_date": "20240112",
    "economic_resolution_type": "CASH_SETTLEMENT", "cash_per_share": 2.0,
    "effective_date": "20240112", "settlement_date": "20240115",
    "source_type": "EXCHANGE", "source_url": "official reviewed locator",
    "document_id": "document identifier", "document_file": "notice.pdf",
    "document_sha256": "64 hexadecimal characters from actual frozen bytes",
    "document_location": "page and paragraph", "reviewed_fact": "exact proven terms",
    "reviewed_by": "operator identity", "review_status": "VERIFIED",
    "evidence_scope": "ECONOMIC_ENTITLEMENT"
  }]
}
```

## Publication and completeness

`economic_event_evidence_v1` binds source/audit identities, inventory, provider bytes,
review, all terminal packages, identity transitions, mapping, audit definitions and
implementation source content. Paths/retrieval time are operational metadata, not
logical identity. Publication uses staging, read-back/business validation,
manifest-last and atomic rename, with validation/reuse of existing identical output.

The validator invokes the existing source validator again, revalidates audit/probe
and document dependencies, and recomputes normalization, matching, ownership,
terminal coverage and summary. Rehashing a modified derived child cannot bypass it.

Artifact `status=COMPLETE` means publication complete, **not evidence complete**.
`summary.json:economic_event_evidence_status` is independently derived:

- BLOCKED: unknown/partial/conflicting terminal economics, conflicting implemented
  events, or incomplete provider capture.
- PARTIAL: unresolved adjustment matches or entitlement eligibility/review gaps.
- COMPLETE: all required evidence and coverage resolved. Tax/execution policy still
  remains unresolved; `execution_authorized=false` even for a complete evidence set.

Dust reporting uses the D.8B descriptive threshold `entry_gross > 1` only; raw counts
and capital are also reported. `dust_execution_policy_status` remains
`DEFERRED_SEPARATE_EXECUTION_SEMANTICS`. Announcement metadata probing and optional
factor/price consistency diagnostics are not implemented; neither is required to
establish evidence completeness and neither would authorize accounting instructions.

## Operator commands for the specified lineage

Run from the repository root. These commands do **not** run a corrected portfolio.
The first command is a real network acquisition and must be operator-authorized.
It queries roughly the exposed-security population, not the entire market.
Receipts are explicit outputs, not a search for a newest artifact. `noclobber`
protects an existing receipt; never delete evidence to repeat a command.

### 1. Provider evidence acquisition

```bash
set -euo pipefail
set -o noclobber
cd /home/zhangkangle/workspaces/ashare-lgbm-quant
if [[ -z "${TUSHARE_TOKEN:-}" ]]; then
  read -r -s -p 'TUSHARE_TOKEN: ' TUSHARE_TOKEN
  printf '\n'
fi
export TUSHARE_TOKEN
mkdir -p reports/research_d8c_accounting_evidence
.venv/bin/ashare-quant --config config/default.yaml data economic-dividend-probe \
  --continuous-run reports/research_d8a_continuous_h5/research/continuous_strict_oos/continuous_strict_oos_f0eeb5ddd6e0faf28bf4feeb \
  --continuous-manifest-sha256 45be5dc7d4af5e4e21c6d424084ba3f488bb96bb0f2f5665ebb3b13cc24d2d22 \
  --audit-root reports/continuous_oos_accounting_audit_f0eeb5ddd6e0faf28bf4feeb_20260929 \
  --audit-manifest-sha256 9a125394f845431b6bc71612a831821f4c1a8b97e4720964881697343c94f056 \
  --output-root data/research/accounting_evidence_sources \
  > reports/research_d8c_accounting_evidence/provider_receipt.json
```

### 2. Initial evidence compilation

This initial command deliberately does not approve provider candidates or supply
unproven terminal terms. It publishes the actual unresolved coverage. After review,
a separately authorized compilation can add `--dividend-review` and repeated
`--terminal-package` paths, producing a new identity without overwriting this one.

```bash
set -euo pipefail
set -o noclobber
cd /home/zhangkangle/workspaces/ashare-lgbm-quant
PROVIDER_SOURCE="$(.venv/bin/python -c 'import json; print(json.load(open("reports/research_d8c_accounting_evidence/provider_receipt.json"))["artifact"])')"
.venv/bin/ashare-quant --config config/default.yaml data economic-evidence-compile \
  --continuous-run reports/research_d8a_continuous_h5/research/continuous_strict_oos/continuous_strict_oos_f0eeb5ddd6e0faf28bf4feeb \
  --continuous-manifest-sha256 45be5dc7d4af5e4e21c6d424084ba3f488bb96bb0f2f5665ebb3b13cc24d2d22 \
  --audit-root reports/continuous_oos_accounting_audit_f0eeb5ddd6e0faf28bf4feeb_20260929 \
  --audit-manifest-sha256 9a125394f845431b6bc71612a831821f4c1a8b97e4720964881697343c94f056 \
  --provider-source "$PROVIDER_SOURCE" \
  --output-root reports/research_d8c_accounting_evidence \
  > reports/research_d8c_accounting_evidence/evidence_receipt.json
```

### 3. Independent validation

```bash
set -euo pipefail
cd /home/zhangkangle/workspaces/ashare-lgbm-quant
EVIDENCE="$(.venv/bin/python -c 'import json; print(json.load(open("reports/research_d8c_accounting_evidence/evidence_receipt.json"))["artifact"])')"
.venv/bin/ashare-quant --config config/default.yaml data economic-evidence-validate \
  --artifact "$EVIDENCE"
.venv/bin/python -c 'import json, pathlib, sys; print(json.dumps(json.loads((pathlib.Path(sys.argv[1])/"summary.json").read_text()), indent=2))' "$EVIDENCE"
```
