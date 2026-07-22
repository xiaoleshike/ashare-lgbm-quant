# Challenger Evaluation Framework

Phase 2.6.2D2 compares one immutable challenger with the current Champion. It
does not retrain either model, use candidate-selection output, read production
observation logs, modify the registry, or promote a model.

## Prediction Contract

Challenger predictions are generated from the `final_test_period.folds` stored
in the source horizon experiment. Every range must end no later than the
horizon's `maximum_mature_evaluation_date`. Prediction reads only
`features_daily`, `universe_daily`, and the challenger artifact. It never reads
labels or future prices.

Production and challenger scoring share the same registered-model feature
identity checks, DuckDB column-pruned feature reads, universe-key validation,
`in_model_universe` filter, numeric conversion, and deterministic ranking.

```bash
ashare-quant --config config/default.yaml models predict-challenger \
  --model-id experiment_c_h5_<identity>
```

This atomically creates:

```text
reports/challenger_predictions/<model_id>/
  predictions.parquet
  manifest.json
```

The model-specific directory is immutable. Repeating an identical command
reuses it; changed input identity is rejected rather than overwritten.

## Fair Evaluation

Evaluation requires identical ordered feature hashes. The Champion is rescored
on exactly the Challenger prediction dates and universe keys. Both scores are
joined once to the same available `future_excess_ret` and
`benchmark_forward_ret` rows for the configured horizon.

```bash
ashare-quant --config config/default.yaml models evaluate-challenger \
  --model-id experiment_c_h5_<identity>
```

Outputs are immutable under `reports/challenger_evaluation/<run_id>/`:

```text
summary.json
evaluation_report.md
metrics.csv
manifest.json
```

`metrics.csv` contains overall, yearly, monthly, and bull/bear/neutral records.
Each record reports Rank IC, ICIR, positive-IC ratio, and equal-weight forward
excess return for Top 1%, 5%, 10%, 20%, and 50% score cohorts. These are
post-hoc ranking portfolio proxies, not executable backtests.

## Promotion Gate

`models.challenger_evaluation` configures transparent minimum observations,
Rank IC, Rank IC delta, positive-IC ratio, and Top-10 return delta. The result
only states whether the challenger is eligible for manual review. D2 never
calls `promote_model()` and leaves both model status and registry bytes
unchanged.
