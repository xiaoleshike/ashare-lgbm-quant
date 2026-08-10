# Multi-Fold Research Validation

Phase 2.8.2I-B separates repeatable historical robustness analysis from genuinely prospective
evidence. It does not select a model automatically and does not mutate the Registry, Champion,
Promotion, Paper Trading, or production state.

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
Multi-fold evidence uses schema v2. Earlier feature provenance schema v1 remains readable, but a
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
selection and historical-holdout fold. Each fold uses the common `RankerDataLoader`, `fit_ranker`,
`evaluate_ranker`, backend resolver, and, when required, the shared Phase I-A
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

Ranking evidence includes Rank IC mean/median/std/ICIR, positive ratio, NDCG@10/50, coverage, date
count, and security count. Executable evidence requires accounting schema v2 and reports Top10,
Top20, and Top50 results. Aggregation reports distributions across folds: mean, median, standard
deviation, minimum, maximum, positive-fold ratio, and best/worst fold index. Feature-importance
rank dispersion is observational only.

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
