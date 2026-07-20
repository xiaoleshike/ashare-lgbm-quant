# Historical Backtest Diagnostics

The diagnostic layer evaluates one immutable, out-of-sample historical backtest. It never fits or
updates a model. It reads the run's frozen `predictions.parquet`, joins horizon-5 labels only after
scores and ranks exist, and reads the model's same-date feature columns only for attribution.

```bash
ashare-quant --config config/default.yaml backtest diagnostics --run-id <run_id>
```

Outputs are atomically published to `reports/backtest_diagnostics/<run_id>/`:

* `daily_ic.csv`: daily Pearson IC plus Spearman Rank IC (`rank_ic` and `spearman_ic`).
* `score_layer_returns.csv`: equal-weighted forward excess return for Top 1%, 5%, 10%, 20%, and
  Bottom 20% score cohorts.
* `monthly_stability.csv`: monthly cohort mean return, Rank IC, and win rate by score layer.
* `single_factor_groups.csv`: daily feature-quantile forward excess-return summaries.
* `summary.json`, `diagnostics_report.md`, and `manifest.json`: aggregate statistics and provenance.

Score-layer returns are post-hoc five-day forward excess-return cohorts, not an executable
portfolio backtest. Because daily five-day cohorts overlap, cumulative and annual returns use five
non-overlapping start-date vintages; the report takes the median terminal/annual return and the
worst vintage drawdown. Sharpe uses the five-day annualization scale `sqrt(252 / 5)`.

Factor attribution is limited to the frozen model feature list. Gain and split importance come from
the persisted LightGBM Booster. TreeSHAP uses a deterministic sample and must reproduce persisted
prediction scores within the configured tolerance. Single-factor groups are descriptive post-hoc
tests and must not be used to refit or promote the evaluated model without a new chronological
research cycle.

Older historical runs without `predictions.parquet` are intentionally rejected. Rerun the same
`backtest historical` command with current code; diagnostics never reconstruct missing scores from
holdings because that would exclude the unselected cross-section.
