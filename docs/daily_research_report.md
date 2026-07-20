# Daily Quantitative Research Report

The optional daily research layer converts production candidate artifacts into deterministic Markdown and JSON. It reads candidates and predictions for identity validation, then adds same-date universe metadata, market capitalization, liquidity, return, trailing volatility, and limit-up observations. It never changes prediction scores or candidate ranks.

```bash
ashare-quant --config config/default.yaml research report --as-of 20260717
```

Outputs are `reports/YYYYMMDD/daily_report.md` and `research_summary.json`. Thresholds and the Top-20 display limit are configured under `research.daily_report`. Recent volatility uses only available `daily.pct_chg` observations on or before the report date; missing observations are not filled. Missing optional metadata produces visible warnings rather than silently dropping candidates.

Industry is included only as descriptive universe metadata. The report is model research output, not investment advice, an order list, or a portfolio simulation.
