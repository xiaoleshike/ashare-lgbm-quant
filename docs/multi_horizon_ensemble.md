# Multi-Horizon Ensemble Evaluation

Phase 2.6.2E evaluates a fixed ensemble of the independently trained 5, 10,
20, and 60 trading-day Challenger Rankers. It is a read-only research layer:
it does not retrain models, select weights, alter the model registry, promote a
model, or generate trading signals.

## Score Construction

The command requires one immutable Challenger prediction artifact for every
configured horizon. All artifacts must share the same feature hash, universe
manifest, feature manifest, source horizon experiment, and `next_open`
execution contract. Evaluation uses only dates common to all four artifacts
and requires identical `(trade_date, ts_code)` keys on every common date.

For each date and horizon, the LightGBM raw score is converted to a
cross-sectional percentile rank. Equal raw scores receive the same average
rank. The fixed first-stage ensemble is:

```text
ensemble_score = mean(h5_rank_pct, h10_rank_pct, h20_rank_pct, h60_rank_pct)
```

Raw LightGBM scores are never averaged because their scales are not comparable
across independently fitted models.

## Post-Hoc Evaluation

Ensemble scoring completes before labels are read. The frozen Champion, each
component Ranker, and the Ensemble are then evaluated separately against the
5, 10, 20, and 60-day mature excess-return labels. Reports include Rank IC,
ICIR, fixed Top-10/20/50 mean excess return, yearly stability, and bull, bear,
and neutral regime stability. Results across target horizons are reported
separately rather than used to choose a weight.

## CLI

Generate all Challenger prediction artifacts first, then run:

```bash
ashare-quant --config config/default.yaml models evaluate-ensemble \
  --model-id experiment_c_h5_<id> \
  --model-id experiment_c_h10_<id> \
  --model-id experiment_c_h20_<id> \
  --model-id experiment_c_h60_<id>
```

The immutable output is written under
`reports/ensemble_evaluation/<run_id>/` as `ensemble_predictions.parquet`,
`metrics.json`, `report.md`, and `manifest.json`.
