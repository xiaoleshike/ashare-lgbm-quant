# Multi-Fold Research Validation

Phase 2.8.2I-B separates repeatable historical robustness analysis from genuinely prospective
evidence. It does not select a model automatically and does not mutate the Registry, Champion,
Promotion, Paper Trading, or production state.

Test and research configurations derive the production lock from the configured paper-trading
state root: its sibling `runs/.production.lock` is the single lock for that isolated project state.
The default `paper_trading` root therefore continues to resolve to the real repository
`runs/.production.lock`. A temporary test configuration neither observes nor deletes that real
lock, while a lock held in the temporary state still blocks the tested mutation.

## Research Policy

The versioned policy is [`config/research_policy.yaml`](../config/research_policy.yaml). Its
canonical content hash is frozen into every new walk-forward experiment.

- `20230101..20260710` is classified as `HISTORICAL_HOLDOUT`. It has already been inspected and
  must not be called an untouched or pristine final test.
- `20260810` is the prospective lockbox start. The date is a research-policy boundary, not a model
  training timestamp.
- Production Shadow, Performance Observation, future dedicated Challenger Paper Trading, and
  Promotion Evidence are allowed prospective consumers.
- Feature selection, hyperparameter/model-family/horizon/fold selection, threshold tuning, and
  ordinary walk-forward research are forbidden from reading the lockbox.
- Future governed training access is explicitly `NOT_YET_ENABLED` by this policy.

The diagnostics pipeline and walk-forward planning/execution entry points fail with
`RESEARCH_LOCKBOX_VIOLATION` when a forbidden window reaches or crosses `20260810`. Production
inference consumers are not blocked by this research-access policy.

## Temporal Isolation

The authoritative label semantics are signal at close T, entry at T+1, then an H-session forward
holding window. Label maturity therefore requires `H + 1` trading sessions:

| Horizon | Required purge | Required embargo |
| --- | ---: | ---: |
| H5 | 6 | 6 |
| H10 | 11 | 11 |
| H20 | 21 | 21 |
| H60 | 61 | 61 |

`ranker.walk_forward.purge_days` and `embargo_days` default to `auto`. AUTO resolves to the
strictest configured horizon. Explicit integers remain supported, but values below the required
gap fail closed; they are never silently increased. A shared H5/H10/H20/H60 plan uses 61 sessions.
The plan freezes configured/resolved values, horizons, required gap, and label semantics.

## Feature Provenance

Fully governed feature-set evidence binds the ordered feature list and hash, selection policy and
version, selection window, diagnostics run and manifest hash, recommendation hash,
feature-universe identity, and asserted creator. Any semantic change produces a new deterministic
feature-set ID. Schema-v2 provenance stores reports-relative source locators; absolute paths and
`created_at` are non-identity metadata. Validation resolves the locator below the active reports
root and verifies the immutable diagnostics and recommendation hashes, so relocating a checkout
does not change identity.

Governed feature-set provenance is the feature authority for new research. The Champion remains a
reference model for model type, semantic defaults, and comparison, but its feature hash does not
constrain a new research feature set. A new plan freezes `feature_set_id`, ordered feature-list
hash, and exact provenance artifact hash. Walk-forward plan schema v4 passes that lineage to
multi-horizon plan schema v3 and the multi-fold runner verifies all three identities match.
Multi-fold evidence produced under the corrected evaluation-universe contract uses schema v3.
Schema-v2 multi-fold evidence remains hash-valid and readable as `LEGACY_READ_ONLY`, but it is not
reinterpreted as corrected evaluation evidence. Earlier feature provenance schema v1 remains
readable, but a
schema-v1 `GOVERNED` artifact is path-bound legacy evidence and is rejected for new governed plans;
`LEGACY_PROVENANCE_INCOMPLETE` remains explicitly legacy and is never upgraded in place.

The existing `robust_features.json` cannot be linked to an exact immutable diagnostics run or
selection window. Its companion
[`robust_20_v1.provenance.json`](../config/feature_sets/robust_20_v1.provenance.json) therefore says
`LEGACY_PROVENANCE_INCOMPLETE`. It remains readable for legacy model compatibility but cannot be
used as fully governed new multi-fold evidence. No diagnostics identity or historical selection
window was invented.

## Multi-Fold Execution

The runner consumes one exact multi-horizon experiment manifest and executes every referenced
selection and historical-holdout fold. Training and fit-validation continue to require mature
labels. Evaluation first scores the signal-date model universe without reading `labels_forward`,
then freezes the complete prediction keys and scores. Ranking metrics left-join mature labels to
those frozen predictions, while executable simulation receives the unfiltered frozen predictions.
Unavailable future labels can reduce metric coverage but cannot remove a stock from the original
rank or replace it in Top-N. Each fold uses the common `RankerDataLoader`, `fit_ranker`, backend
resolver, and, when required, the shared Phase I-A
`simulate_portfolio` accounting engine. It does not register the fold model.

Artifacts are published under:

```text
reports/research/walk_forward/<run_id>/
  folds/<fold_id>/
    model.txt
    predictions.parquet
    validation_metrics.json
    ranking_metrics.json
    executable_metrics.json
    feature_importance.json
    manifest.json
  aggregate_metrics.json
  fold_summary.parquet
  manifest.json
```

Per-fold publication uses staging, atomic rename, and manifest-last. The top-level manifest is
written only after every required fold validates. Resume reuses hash-valid completed folds and
reruns only an unpublished interrupted fold. Missing or corrupt folds are never excluded from a
successful average. Recovery inspection is read-only.

`COMPLETE` is accepted only after one shared root-to-leaf validator verifies the root schema,
status and identity; `aggregate_metrics.json`; `fold_summary.parquet`; the exact expected fold
directory set; every fold manifest; and every model, prediction, metric, executable, and feature
importance child hash. Status, completed-run resume, and recovery all reuse this validator. Any
tamper returns a failure or `ACTION_REQUIRED`; completed evidence is never regenerated or repaired.

Ranking evidence includes Rank IC mean/median/std/ICIR, positive ratio, NDCG@10/50, date count,
security count, and separate prediction and label coverage. Per-date diagnostics record expected
universe rows, feature rows, scored/finite-score rows, mature/available/unavailable labels, and
unavailable reasons. Top-fraction label proxies freeze membership before labels are joined and
state requested, available, and missing counts.

Feature-selection isolation is evaluated separately from parameter-fit isolation. The diagnostics
selection end is advanced through the configured forward-label maturity sessions using the
exchange calendar. A fold is `STRICT_OOS` only when both parameter fitting and feature-selection
information precede evaluation. Earlier folds remain inspectable as
`RETROSPECTIVE_FIXED_FEATURE_REPLAY`, but their performance appears only in descriptive grouped
statistics, not in the strict-OOS aggregate. Executable evidence requires accounting schema v3,
including a hash-bound corporate-action ledger, and reports Top10, Top20, and Top50 results.
Aggregation reports distributions across eligible folds:
mean, median, standard deviation, minimum, maximum, positive-fold ratio, and best/worst fold index.
Feature-importance rank dispersion is observational only.

Schema-v3 run identity separates modeling identity from execution identity. Execution identity
binds the requested/effective training backend, LightGBM version, evaluation contract, accounting
schema, execution mode, holding period, sell-delay policy, and effective-dated cost-policy hash.
Changing CPU/CUDA, cost policy, or evaluation contract creates a different immutable run and cannot
resume into old evidence.

Evaluation contract v8 adds corporate-action policy, transition, and ledger identity to the
executable contract. It preserves the contract-v7 lifecycle-audit lineage and the contract-v5
separation of delayed-exit alerts from final resolution for walk-forward
execution evidence. The configured 20-session sell-delay value remains an auditable breach
threshold. A breached position is never written off merely for exceeding it: walk-forward
simulation carries an explicitly suspended position to the first authoritative tradable open,
records the maximum delay and resolution date, and stops once all post-evaluation holdings resolve.
The final execution cutoff is the earlier of the immutable processed-universe maximum date and the
day before the prospective lockbox. Data is loaded in bounded session chunks as an operational
optimization, but a chunk boundary is not an economic write-off or final evidence cutoff. No new
signals are created in the tail. Positions still open at the governed cutoff fail closed.

The execution identity also binds the version and hash of the explicit security-lifecycle policy.
That policy covers exchange-authorized listing-suspension intervals which ordinary `suspend_d`
does not necessarily continue to publish. Missing quotes never create lifecycle evidence.

New executable walk-forward runs also require an intact `PASS` lifecycle scan manifest via
`--lifecycle-scan-manifest`. The gate validates the root and every child hash, exact raw and
processed source inventory, identity mapping, lifecycle policy/evidence hashes, and date coverage
through the governed execution cutoff. The scan lineage is part of execution identity, not model
semantic identity. Ranking-only research does not require this execution-data gate.

The governed preflight order is:

```text
security-identity-scan
    -> security-lifecycle-scan
    -> PASS
    -> evidence-grade executable walk-forward
```

`classification != safe execution evidence`: `MISSING_RAW_DATA`, `UNRESOLVED`, policy collisions,
and lifecycle boundary inconsistencies all keep the scan `BLOCKED`. Existing walk-forward artifacts
without this lineage remain immutable and readable as legacy evidence; they are not upgraded in
place.

When preflight is blocked, run `security-lifecycle-triage` before acquiring new evidence or
rebuilding research data. Triage clusters every residual interval by source shape, duration,
exchange, regulatory period and nearby evidence, but its candidate categories never authorize
execution. Terminal and known listing-suspension reconciliation distinguish stale processed
universe snapshots from missing real-world lifecycle evidence.

A universe rebuilt under a changed lifecycle overlay invalidates the current governed status of
the dependent feature, label, diagnostics, feature-provenance, walk-forward-plan, horizon-plan and
walk-forward-evidence chain. Those artifacts are retained as immutable prior-contract evidence;
new artifacts must be generated from the new isolated universe after lifecycle preflight passes.

Lifecycle scanner schema v3 uses the daily-snapshot `suspend_d` contract. Schema v4 preserves that
contract and additionally binds a separate official security-code transition graph. Boundary
findings are event-level and carry `INFO`, `WARNING`, or blocking `BOUNDARY_INCONSISTENCY` severity. Provider
shape diagnostics do not block execution by themselves. A recalibrated scan may explicitly name a
prior immutable manifest with `--supersedes-scan-manifest` and
`--supersession-reason SUSPEND_D_SEMANTICS_RECALIBRATION`; the new artifact publishes a hash-bound
old/new comparison without editing the prior scan.
Contract-v3 through contract-v6 evidence remain immutable and hash-readable as
`LEGACY_READ_ONLY`; they cannot be resumed as contract-v7 evidence.

## Fold Backend Benchmark

The existing backend benchmark supports either a legacy immutable model artifact or a current
schema-v3 walk-forward fold. Fold mode validates the entire walk-forward root, requires exact
governed feature provenance, and permits selection-period folds only. Omitting `--fold-id` chooses
the chronologically latest selection fold without consulting IC or return. It loads only that
fold's train and validation windows.

Benchmark identity binds actual row content, row order, dtypes, query groups, relevance labels,
ordered features, semantic parameters, horizon, seed, and the loaded LightGBM binary hash. CPU and
CUDA comparison rejects any mismatch, duplicate prediction key, non-finite prediction, missing
mandatory artifact, or path-escaping manifest entry. It reports pooled and per-date correlations
plus daily Top10/20/50 overlap; speed cannot override a correctness failure. CUDA remains optional
for ordinary CPU research and unavailable CUDA fails closed without publishing a CUDA result.

Negative performance does not make a fold technically invalid. Technical validity means chronology,
identity, OOS evaluation, source integrity, complete predictions, current accounting evidence when
required, and immutable publication all pass. The runner never selects the best fold, horizon,
feature subset, or hyperparameters.

## Commands

Create a governed feature-set artifact from one exact immutable diagnostics run:

```bash
ashare-quant --config config/default.yaml models feature-set-create \
  --diagnostics-dir reports/feature_diagnostics/RUN_ID \
  --name robust_20 --version v2 --created-by OPERATOR_ID
```

Create a horizon-safe plan (AUTO uses the configured H5/H10/H20/H60 scope):

```bash
ashare-quant --config config/default.yaml models walk-forward-plan \
  --start-date 20100101 --end-date 20260710 --scheme expanding \
  --feature-provenance reports/feature_selection/FEATURE_SET_ID/feature_set.json
```

Execute an exact experiment after creating a fully governed feature-set provenance artifact:

```bash
ashare-quant --config config/default.yaml models walk-forward-run \
  --experiment-id EXPERIMENT_ID \
  --experiment-manifest reports/horizon_experiments/RUN/experiment_manifest.json \
  --feature-provenance reports/feature_selection/FEATURE_SET_ID/feature_set.json
```

`--ranking-only` explicitly records executable evidence as not required. It must not be used where
the research contract requires executable accounting.

```bash
ashare-quant --config config/default.yaml models walk-forward-status \
  --experiment-id WALK_FORWARD_RUN_ID

ashare-quant --config config/default.yaml models walk-forward-recovery \
  --experiment-id WALK_FORWARD_RUN_ID
```

Status and recovery validate the complete root-to-leaf evidence chain. Recovery never repairs or
deletes artifacts.

Run a backend benchmark against a specific corrected walk-forward selection fold:

```bash
ashare-quant --config config/default.yaml models \
  --processed-root "$RESEARCH_PROCESSED" \
  --reports-root "$RESEARCH_REPORTS" \
  benchmark-training-backend \
  --backend cpu \
  --walk-forward-run-id "$H5_RUN_ID" \
  --fold-id FOLD_ID \
  --feature-provenance "$FEATURE_SET_JSON"
```

The same command with `--backend cuda` is valid only after the explicit CUDA capability probe
succeeds. Compare the resulting immutable artifacts with `compare-training-backends`.
Evidence-grade `walk-forward-run` requires a PASS lifecycle scan and one explicit identity
transition contract. Supply either `--identity-transition-artifact PATH` or, only when the scan
uses the same reviewed empty contract, `--no-identity-transitions`. The runner validates the
version/hash before fitting a fold and binds it into execution identity. Accounting schema v3 also
binds `corporate_action_execution_policy_v1`. A held predecessor may cross only a verified,
one-to-one `CODE_CHANGE_CONTINUITY` whose continuity is `SAME_LISTED_ENTITY` and whose finite
positive ratio is explicit. The engine records a zero-cash, zero-cost, zero-turnover position
transformation before same-day sell logic. Unknown ratios, restructuring, successor entities, and
ambiguous topology remain blocked with `CORPORATE_ACTION_EXECUTION_UNSUPPORTED`.

Training uses the mature, available labeled subset and publishes per-date selection counts.
Scoring remains label-free. Evaluation-label maturity requires a valid exchange-session
sequence, T+1 entry, the configured H-session endpoint, an actual exit no earlier than that
endpoint, and an exit no later than the governed cutoff.
