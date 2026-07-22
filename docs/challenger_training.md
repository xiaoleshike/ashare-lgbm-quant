# Challenger Training Framework

Phase 2.6.2D1 trains immutable LightGBM Ranker candidates from an existing
multi-horizon experiment plan. It does not compare models, read final-test
labels, promote a model, or generate predictions.

## Inputs and Selection Policy

The trainer requires a schema-v2 `experiment_manifest.json`, its referenced
schema-v2 walk-forward plan, and the current `features_daily` and
`universe_daily` manifests. Source hashes must still match. The ordered feature
list is inherited from the source model recorded by the horizon plan.

For each requested horizon, the trainer deterministically selects the last
mature fold in `selection_period`. It loads only that fold's train and
validation ranges. The fold's evaluation range and every `final_test_period`
range remain unread. Purge and embargo boundaries must satisfy the
horizon-specific maturity requirement.

The target is bound one-to-one: horizon `H` requires
`future_excess_ret_Hd`, holding period `H`, and `next_open` execution semantics.
The existing data loader internally stores the selected horizon's target in a
legacy column named `future_excess_ret_5d`; its query still filters physical
label rows by `H`.

## Commands

Train one horizon by stable alias:

```bash
ashare-quant --config config/default.yaml models train-challenger \
  --experiment-id experiment_c_h20
```

Train all configured horizons independently:

```bash
ashare-quant --config config/default.yaml models train-challenger \
  --all-horizons
```

Use `--experiment-manifest PATH` to pin a specific plan instead of the latest
available plan.

## Artifacts and Lifecycle

Each model is atomically published under
`models/challengers/experiment_c_<horizon-name>_<identity>/`. The directory
contains `model.txt`, `feature_list.json`, `metrics.json`, and `manifest.json`.
Existing directories are never replaced.

After publication, the registry records the model as `candidate`. Validation
metrics are stored, while test metrics remain empty. Promotion remains a
separate explicit operation and cannot pass the existing promotion gate until
a later phase performs the authorized final evaluation.

Fair final-test scoring and comparison are documented in
[`challenger_evaluation.md`](challenger_evaluation.md).
