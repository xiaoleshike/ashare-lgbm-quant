# Model Drift Diagnostics

Phase 2.6.2A provides read-only diagnostics for a registered Ranker and an existing immutable
historical prediction artifact. It does not load the LightGBM model, rescore stocks, fit a model,
select features, promote a model, or modify source data.

```bash
ashare-quant --config config/default.yaml models diagnostics drift \
  --model-id MODEL_ID \
  --start-date 20230101 \
  --end-date 20260709
```

The requested model must be registered. An OOS historical backtest for the same model and feature
hash must contain `predictions.parquet` and cover the complete requested range. Run the historical
backtest first when no suitable prediction artifact exists.

Outputs are published under `reports/model_diagnostics/<run_id>/`:

* `feature_drift.parquet`: monthly PSI, KS D statistic, finite-value coverage, missing ratio, and
  drift against the model training period;
* `score_drift.parquet`: monthly raw-score and percentile distributions, Top 1%/10% softmax
  concentration, score spreads, and effective breadth;
* `feature_response.parquet`: monthly daily Rank IC, reference sign, sign changes, and disjoint
  quintile/decile future-excess-return responses;
* `summary.json`, `diagnostics_report.md`, and `manifest.json`: aggregate findings and provenance.

Feature PSI bins are fitted only from a deterministic sample of the registered model's training
period. Exact coverage uses all eligible rows. Evaluation distribution statistics use deterministic
monthly samples. Score drift uses the earliest configured number of requested months as its
reference because existing model artifacts do not persist training-period scores; the selected
months are recorded in the manifest.

Labels are loaded only after prediction, feature-drift, and score-drift inputs are fixed. Only
available labels with `exit_date > trade_date` are accepted, and they are used solely for post-hoc
feature-response analysis. These diagnostics are model-health evidence and cannot directly select,
train, promote, or alter a model.
