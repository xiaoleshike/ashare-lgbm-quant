# Production Candidate Selection

Candidate selection is an optional post-inference research layer. It reads same-date predictions and point-in-time universe, market, liquidity, and price-limit data. It does not create orders or make assumptions about next-session execution.

Rules are configured under `strategy.candidate_selection` in `config/default.yaml`. Defaults require model-universe membership; exclude ST, suspended, low-liquidity, newly listed, Beijing Stock Exchange, and sub-threshold market-cap stocks; require same-date `daily`, `daily_basic`, and `stk_limit` coverage; and reject invalid OHLC or price-limit relationships. STAR (`688xxx.SH`) and ChiNext (`300xxx.SZ`) remain enabled by default and have independent future-use exclusion switches.

Tushare units are preserved: `daily_basic.total_mv` is in CNY 10,000 and `daily.amount` is in CNY 1,000. Defaults therefore require `total_mv >= 500000` (CNY 5 billion) and `amount >= 30000` (CNY 30 million). Optional minimum and maximum `turnover_rate` bounds are disabled by default. An observed close at a valid limit is not itself abnormal and is not treated as a buy or sell decision.

```bash
ashare-quant --config config/default.yaml strategy candidates --as-of 20260717
```

The command writes `reports/YYYYMMDD/candidates.csv`, a compatibility `candidate.csv`, and `candidates_manifest.json`. It also extends the existing inference `summary.json` with candidate and exclusive filter-reason counts. Candidate `rank` is recomputed after filtering while preserving prediction-score order and using `ts_code` as the deterministic tie-breaker.
