# Single-Stock Model Outlook

The standalone script scores one stock against the complete same-date `in_model_universe` using a
specified registered Ranker:

```bash
.venv/bin/python scripts/predict_stock_outlook.py \
  --config config/default.yaml \
  --model-id MODEL_ID \
  --ts-code 000001.SZ \
  --as-of 20260717 \
  --horizon 10
```

The model manifest `label_horizon` and target must exactly match the requested horizon. A 5-day
Ranker cannot produce a 10-day outlook. With the current 5-day champion, use `--horizon 5` to inspect
its actual signal, or train and register a separate `future_excess_ret_10d` Ranker before requesting
10 days.

Output fields include the model score, full-universe rank, percentile, relative-strength band, and
the T+1-open entry and horizon-consistent exit dates resolved from `trade_cal`. The output is a
cross-sectional relative outlook versus the eligible A-share universe. It is not an absolute price
forecast, a daily path, a target price, or a guaranteed return.

The script reads only the model registry/artifact, `features_daily`, `universe_daily`, and
`trade_cal`. It does not read labels, modify production inference, publish recommendations, or write
data unless `--output` explicitly names a JSON file.
