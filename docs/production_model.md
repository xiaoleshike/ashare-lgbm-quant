# Production Ranker Training

Phase 7.5 trains the final LightGBM Ranker after the robust feature set has been approved.

This stage differs from the research baseline:

- it does not run train/validation/test evaluation;
- it does not tune parameters;
- it does not select features;
- it uses the frozen robust feature list from `config/feature_sets/robust_features.json`;
- it trains on the full approved history, default `20100101` through `20260710`.

Run:

```bash
ashare-quant --config config/default.yaml models train-production
```

The fixed LightGBM Ranker parameters are the same as the Phase 7 baseline `ranker` configuration.
The output is always written to `models/production/`:

- `model.txt`
- `feature_list.json`
- `metrics.json`
- `manifest.json`

`metrics.json` contains training provenance and feature importance only. It intentionally does not
contain validation or test performance. `manifest.json` records the training date range, feature
hash, Git revision, config hash, fixed parameters, and processed source manifests.
