# Production Inference

Production inference is a read-only scoring operation over completed, readiness-approved artifacts. It loads the explicitly promoted `lightgbm_ranker` champion from `models/registry.json`, verifies its ordered feature hash, reads only the requested session from `features_daily`, and filters rows using the same-date `in_model_universe` flag. It never reads labels, future prices, or trading execution data.

Run inference after the daily data pipeline succeeds:

```bash
ashare-quant --config config/default.yaml models champion
ashare-quant --config config/default.yaml pipeline readiness --as-of 20260717
ashare-quant --config config/default.yaml models predict --as-of 20260717
```

Results are published under `reports/YYYYMMDD/`:

* `predictions.parquet`: `trade_date`, `ts_code`, `prediction_score`, and `model_id`.
* `ranking.csv`: deterministic descending score rank with `ts_code` as the tie-breaker.
* `summary.json`: model, feature, universe, prediction, Git, and configuration summary.
* `manifest.json`: model identity, feature hash, input manifests, readiness results, and build provenance. It is published last as the completion marker.

This stage produces ranking scores only. It does not recommend trades, size positions, simulate execution, or account for costs.
