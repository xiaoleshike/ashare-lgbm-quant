# Relevant economic event review

This is P0-D2 evidence infrastructure, not an accounting policy. The engine,
accounting schema 3, terminal writeoffs and economic execution authorization are
unchanged. No labels, performance metrics or position PnL enter candidate selection
or review. The continuous/audit/provider/D1 identities are explicit inputs.

## Contracts

`economic_event_review_v1` first recursively validates the D1 compilation,
including the existing continuous artifact validator and audit hashes. It also
validates the supplied provider copy, rather than trusting an equivalent manifest
with unvalidated children. Historical D1 content fingerprints are explicitly
recognized for reading; every derived table is still recomputed with current
validators. New D1 publications bind current implementation content.

## Official nullable-field supplementation

The [Tushare dividend field reference](https://tushare.pro/document/2?doc_id=103)
defines total distribution, bonus, reserve conversion and both cash fields. It does
**not** establish a universal null-to-zero convention. Neither `stk_div=0` nor a
factor movement authorizes filling missing components. Missing pay/share-listing
dates are distinct from the five-field economics completeness check.

`economic_official_field_supplement_v1` is an additive, evidence-only path over an
exact validated D2 parent. It does not reinterpret or replace its revision groups,
queue membership, original nullable terms, conflicts or terminal coverage.

- `economic-review-supplement --parent ... --parent-manifest-sha256 ...
  --supplements ... --output-root ...` publishes a new immutable qualification
  artifact. `economic-review-supplement-validate --artifact ...` recursively
  validates its parent and independently rederives proposals and review outputs.
- Input JSON contains `contract`, `parent_manifest_hash`, `documents`,
  `supplements` and `decisions`. A supplement binds the exact event/group, canonical
  security, record/ex dates, **all** provider-row hashes, and page-specific identity
  citations. Each amount claim binds document SHA256, page, verbatim clause,
  field/value, disclosed value and explicit per-share/per-ten-shares/zero basis.
- Original PDF/UTF-8 HTML bytes are copied into `documents/` and hashed. PDFs use
  native `pdftotext` (required on the host), not OCR. Authority is restricted to
  exchange/CNINFO HTTPS domains. Every cited document must separately contain the
  security/date identity. Quote presence is a locator guard, **not** a machine
  assertion that the quoted words establish the economics. That is a human duty.
- Only missing amounts may be supplemented. Known provider amounts, missing or
  differing record/ex dates, nearby dates and conflicting revisions cannot be
  overridden. Payment/listing dates stay nullable; no execution date is invented.
- `decisions: []` publishes proposals and a blank `review_template.json`, never
  approves them. A separate explicit decision binds `supplement_hash`, reviewer,
  reason and `APPROVE_AS_EVIDENCE`/rejection/further-document requirement. The hash
  binds all claims and cited document hashes. Timestamp is non-logical metadata.
- An explicit approval additionally requires a single exact candidate, no original
  conflicts, complete consistent supplemented amounts and valid dates. It publishes
  `reviewed_corporate_actions.parquet`, still **qualification only**; it cannot
  authorize engine execution or make global economic completeness PASS.
- `field_proposals.json` preserves raw versus proposed versus reviewed terms and
  field-level source claims. The copied queue keeps all original columns unchanged
  and adds supplement status only. Cash tax semantics remain
  `UNRESOLVED_EXECUTION_POLICY`. Terminal recovery remains a separate evidence track.

To review, create a **new** local input JSON retaining the frozen documents and
supplements, fill only the decisions actually reviewed using the published template,
and compile to the same output root. Never edit a published artifact. Changed
decisions produce a new content identity; unchanged inputs validate and reuse.

The D1 terminal guard now normalizes blank effective/settlement strings to null.
They can no longer satisfy `EVIDENCE_COMPLETE`. The explicitly pinned pre-fix D1
fingerprint is readable only with full current business reconstruction; it is not
a blanket exemption for old unsafe terminal-completeness results. No accounting,
writeoff, cost, dividend-tax or portfolio behavior changes in this contract.

Candidate canonical codes come from the already validated D1 alias resolver.
Provider codes and every raw-row hash are preserved. No name/price/factor-based
alias or new identity transition is introduced.

Primary review population: one row per audit-observed adjustment boundary.
Terminal boundaries have a separate queue. Supplementary cash/share events with
an ex-date during actual holdings or record-date EOD ownership are separately
reported even without a detected factor change. Relevant undated events get a
separate security-level unresolved queue; an unknown date is not proof of irrelevance.
The operator must investigate a queue that remains comparable to the full historical
provider population, not approve or delete rows to reach a target count.

Only exact canonical-code/ex-date equality is a match. Other event dates within the
actual position's holding interval are context, not a +/- day approval tolerance.
An undated same-security observation cannot be assumed irrelevant. Dated historical
rows with no relationship to holdings remain immutable raw evidence but leave the
review workload. No performance or amount comparison selects an event date.

Implemented rows are grouped by report period and nullable material terms:
record/ex/pay/share-listing dates and all five cash/share fields. Announcement and
implementation-announcement dates are retained as metadata lists, not a rule for
choosing the latest version. Equivalent material terms retain all raw hashes.
Invalid values cannot collapse into valid nulls. Different material terms sharing
code and ex-date or report period are conservatively conflicting. Same-period
multiple distributions require official clarification; no supersession is guessed.

`stk_div` versus bonus + reserve is a **diagnostic**, using absolute tolerance
1e-10 and relative tolerance 1e-8. No absent component is inferred. Missing cash/share
fields remain incomplete. Both tax amounts are retained; tax policy stays
`UNRESOLVED_EXECUTION_POLICY`. Payment/share-listing dates may remain null in accepted
evidence; that does not authorize accounting at a guessed date. A share diagnostic
requiring clarification still blocks overall evidence completeness after review.
Factor direction is only a side-by-side diagnostic, never an economics estimator.

Review priority is fixed, descriptive and independent of approval: P0 capital >=
100,000; P1 >= 10,000; P2 any position entry gross > 1; otherwise DUST_ONLY. Capital
is summed entry gross across account exposures to each boundary, **not** unique
invested capital or a loss measure. All dust exposures remain in the denominator.

## Explicit decisions

Triage publishes `economic_event_review_template.json` with null decisions. Copy it
outside the immutable artifact, then fill only actually reviewed items. Delete
unreviewed entries from the editable copy, not from the source artifact. Omitted
items stay unresolved. Allowed decisions:

- APPROVE_AS_EVIDENCE
- REJECT_NOT_SAME_EVENT
- BLOCK_CONFLICT
- REQUIRE_OFFICIAL_DOCUMENT

Every decision binds source context, review-event ID, candidate-group ID and exact
raw hashes, plus reviewer and reason. An approval requires a unique exact,
internally usable candidate; nearby, missing-date and conflicting candidates cannot
be approved by merely changing a string. Rejection is retained but does not explain
an adjustment boundary. No automatic approval is implemented.

Compilation publishes an immutable `economic_review_decisions_*` package and a new
`economic_event_review_*` artifact. Reviewer timestamp is stored only as metadata;
canonical decisions, not timestamps, determine both identities. Changing a decision
changes both IDs. Metadata bytes remain child-hash protected. Identical decisions
with a different timestamp reuse the existing validated package without overwriting.

Accepted events feed the existing record-date EOD ownership service for reporting,
not cash/share mutation. Ordinary reviewed rows, terminal coverage, ownership gaps,
provider capture gaps and unresolved findings are persisted separately. Unknown
terminal recovery always keeps overall evidence BLOCKED. Material-only statistics
are descriptive; they never delete unknown rights. `execution_authorized=false`
applies even if all evidence eventually becomes COMPLETE.

## Optional terminal discovery

The only new network operation is the explicit `terminal-announcement-probe`.
It reuses TushareClient retries/rate limits and queries verified codes/aliases for
each actual terminal boundary within a fixed 365-calendar-day window on either
side. Later metadata is POST_HOC discovery, never signal evidence.

[Official anns_d reference](https://tushare.pro/document/2?doc_id=176): separate
permission, date/code filters and a 2,000-row ceiling. Hitting the ceiling is
POSSIBLY_TRUNCATED; empty responses do not prove absence. Permission failure is
PERMISSION_REQUIRED with operator actions, not failure of ordinary triage.
Provider exception text is not persisted. No PDF is fetched or parsed.

Chinese terminal/merger/rights title filters only narrow discovery. Candidate URLs
and metadata hashes never set economic treatment. Existing frozen lifecycle index
matches are shown as context, not zero-recovery evidence. Actual economic treatment
requires the existing D1 `terminal-economic-evidence-publish` reviewed-document
contract. New compilations inherit parent terminal packages and may add immutable
packages via repeated `--terminal-package`; conflicting terms remain blocked.

## Publication and validation

The existing staging, manifest-last, read-back and atomic no-overwrite publisher is
reused. The logical identity binds exact source/audit/D1/provider hashes, triage and
grouping rules, implementation content, canonical decision hash/ID, terminal package
hashes and optional metadata-discovery hash. Absolute locators are operational only.

`validate_reviewed_economic_evidence_artifact()` revalidates parents and rederives
selection, groups, queue, decisions, accepted rows, ownership, terminal evidence,
CSV/Parquet consistency, unresolved findings and summary. Changing a derived child
and rehashing it cannot pass business reconstruction. Artifact publication COMPLETE
does not imply economic evidence COMPLETE or execution authorization.

## Operator commands for the specified committed D1 lineage

These commands were **not run on real inputs during implementation**. Real source
validation was run; candidate reduction counts below must come from manual triage,
not fixture output. Receipts are explicit inputs, never a search for "latest".
Do not overwrite a prior receipt or immutable artifact when rerunning.

### A. Deterministic triage (no token, no provider query)

```bash
set -euo pipefail
set -o noclobber
cd /home/zhangkangle/workspaces/ashare-lgbm-quant
mkdir -p reports/research_d8d_reviewed_accounting_evidence
.venv/bin/ashare-quant --config config/default.yaml data economic-review-triage \
  --initial-evidence reports/research_d8c_accounting_evidence/economic_event_evidence_e0f861271895d9becf1bcc38 \
  --initial-manifest-sha256 410abe4baee44eaa6c1a8b6b1202706a33caa6febb39f39563b0313e01761b9c \
  --provider-source data/research/accounting_evidence_sources/economic_event_source_72a8728ce4dc3ab07fd89c59 \
  --provider-manifest-sha256 d96cb1834112bd0610169fe3017d6c62513c8d20f2b40298d3aeb7df6ab1c811 \
  --output-root reports/research_d8d_reviewed_accounting_evidence \
  > reports/research_d8d_reviewed_accounting_evidence/triage_receipt.json
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('reports/research_d8d_reviewed_accounting_evidence')
p = Path(json.loads((root / 'triage_receipt.json').read_text())['artifact'])
print((p / 'summary.json').read_text())
print('Review queue:', p / 'economic_event_review_queue.csv')
PY
```

### B. Optional terminal announcement metadata (network, no PDFs)

```bash
set -euo pipefail
set +x
set -o noclobber
cd /home/zhangkangle/workspaces/ashare-lgbm-quant
read -r -s -p 'TUSHARE_TOKEN: ' TUSHARE_TOKEN
export TUSHARE_TOKEN
printf '\n'
mkdir -p reports/research_d8d_reviewed_accounting_evidence
.venv/bin/ashare-quant --config config/default.yaml data terminal-announcement-probe \
  --initial-evidence reports/research_d8c_accounting_evidence/economic_event_evidence_e0f861271895d9becf1bcc38 \
  --initial-manifest-sha256 410abe4baee44eaa6c1a8b6b1202706a33caa6febb39f39563b0313e01761b9c \
  --provider-source data/research/accounting_evidence_sources/economic_event_source_72a8728ce4dc3ab07fd89c59 \
  --provider-manifest-sha256 d96cb1834112bd0610169fe3017d6c62513c8d20f2b40298d3aeb7df6ab1c811 \
  --output-root data/research/accounting_evidence_sources \
  > reports/research_d8d_reviewed_accounting_evidence/announcement_receipt.json
unset TUSHARE_TOKEN
.venv/bin/python - <<'PY'
import json
from pathlib import Path
r = Path('reports/research_d8d_reviewed_accounting_evidence/announcement_receipt.json')
p = Path(json.loads(r.read_text())['artifact'])
print((p / 'summary.json').read_text())
PY
```

### C. Materialize an editable review template (no approvals)

The template was generated by A. This command copies it exclusively to an operator
file; existing decisions cannot be overwritten. Edit that new file with actual
reviewer decisions before D. A null/unfilled template deliberately fails compilation.

```bash
set -euo pipefail
cd /home/zhangkangle/workspaces/ashare-lgbm-quant
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('reports/research_d8d_reviewed_accounting_evidence')
p = Path(json.loads((root / 'triage_receipt.json').read_text())['artifact'])
target = root / 'economic_event_review_decisions_v1.json'
with target.open('xb') as stream:
    stream.write((p / 'economic_event_review_template.json').read_bytes())
print('Editable review file:', target)
PY
```

### D. Compile explicit reviewed decisions

This is not automatic review. The optional receipt from B is included only when it
exists explicitly; no directory discovery is performed. Verified terminal packages
can later be supplied using the existing D1 publisher and `--terminal-package`.
Without such packages terminal recovery remains UNKNOWN.

```bash
set -euo pipefail
set -o noclobber
cd /home/zhangkangle/workspaces/ashare-lgbm-quant
EXTRA=()
if [[ -f reports/research_d8d_reviewed_accounting_evidence/announcement_receipt.json ]]; then
  ANNOUNCEMENTS="$(.venv/bin/python -c 'import json; print(json.load(open("reports/research_d8d_reviewed_accounting_evidence/announcement_receipt.json"))["artifact"])')"
  EXTRA+=(--announcement-source "$ANNOUNCEMENTS")
fi
.venv/bin/ashare-quant --config config/default.yaml data economic-review-compile \
  --initial-evidence reports/research_d8c_accounting_evidence/economic_event_evidence_e0f861271895d9becf1bcc38 \
  --initial-manifest-sha256 410abe4baee44eaa6c1a8b6b1202706a33caa6febb39f39563b0313e01761b9c \
  --provider-source data/research/accounting_evidence_sources/economic_event_source_72a8728ce4dc3ab07fd89c59 \
  --provider-manifest-sha256 d96cb1834112bd0610169fe3017d6c62513c8d20f2b40298d3aeb7df6ab1c811 \
  --decisions reports/research_d8d_reviewed_accounting_evidence/economic_event_review_decisions_v1.json \
  --output-root reports/research_d8d_reviewed_accounting_evidence \
  "${EXTRA[@]}" \
  > reports/research_d8d_reviewed_accounting_evidence/reviewed_receipt.json
```

### E. Validate final evidence

```bash
set -euo pipefail
cd /home/zhangkangle/workspaces/ashare-lgbm-quant
ARTIFACT="$(.venv/bin/python -c 'import json; print(json.load(open("reports/research_d8d_reviewed_accounting_evidence/reviewed_receipt.json"))["artifact"])')"
.venv/bin/ashare-quant --config config/default.yaml data economic-review-validate --artifact "$ARTIFACT"
.venv/bin/python -c 'import pathlib,sys; print((pathlib.Path(sys.argv[1])/"summary.json").read_text())' "$ARTIFACT"
```

Stop here. No accounting-engine repair, corrected replay, model training, label
support, lot-size policy change or portfolio optimization is authorized.
