# Feature Diagnostics and Selection

The diagnostics stage identifies robust predictors without using the final test period for
selection. It joins `features_daily`, available labels from `labels_forward`, and
`in_model_universe=true` rows on `trade_date, ts_code`.

## Chronological Contract

Every run requires non-overlapping train, validation, and test ranges. Training data determines
coverage gates, daily Pearson IC, daily Spearman Rank IC, annual stability, correlation pruning,
and the feature ordering from train-fitted LightGBM split/gain importance. Validation data is used
for permutation diagnostics, family ablation, incremental-family tests, and comparison of top
30/50/70/100/130/all-accepted sets. The recommended set is frozen before test data is loaded. Test
metrics are reported once and cannot alter the recommendation.
After the set is frozen, the final diagnostic model is refitted on train plus validation data and
evaluated once on the test period.

```bash
ashare-quant --config config/default.yaml diagnostics run \
  --train-start 20100101 --train-end 20191231 \
  --validation-start 20200101 --validation-end 20221231 \
  --test-start 20230101 --test-end 20260710 \
  --horizon 5
```

Use dates appropriate to the research protocol; do not repeatedly inspect and retune against the
test period. `diagnostics status` prints the latest report identity.

## Metrics and Outputs

IC is computed independently for each daily cross-section. ICIR is daily IC mean divided by daily
IC standard deviation. Coverage is measured over eligible, label-available model-universe rows.
Bull, bear, and neutral statistics are ex-post diagnostics based on benchmark forward return and
must not be used as contemporaneous features.

Greedy correlation pruning keeps the higher-ranked training feature when absolute sampled
correlation exceeds the configured threshold. LightGBM uses fixed diagnostic parameters and a
deterministic row sample. Permutation occurs within each validation date. Family ablation removes
one family at a time; incremental tests add families in training-strength order.

Each immutable run under `reports/feature_diagnostics/<run-id>/` contains coverage/IC summaries,
daily and yearly IC, regime results, pairwise correlations, pruning decisions, model and
permutation importance, family tests, candidate-count comparisons, a recommended feature JSON,
and a provenance manifest. Top-decile return, Sharpe, drawdown, and turnover are frictionless
diagnostic statistics, not strategy backtest results.
