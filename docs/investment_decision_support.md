# Investment Decision Support

The decision-support layer converts existing model predictions, research candidates, local SHAP
explanations, same-date technical features, and `daily_basic` fields into a deterministic report for
human review. It does not change model scores or candidate ranks and does not create trading actions.

Run it after candidate selection and explainability:

```bash
ashare-quant --config config/default.yaml research decision --as-of 20260717
```

Outputs are `reports/YYYYMMDD/decision.json` and `decision_report.md`. Each candidate retains its
original model and candidate ranks and includes positive and negative SHAP factors, current technical
state, configurable observation conditions, and triggered risk observations.

All rules are configured under `research.decision_support`. The default technical inputs are
`gap_mean_1d`, `ma_ratio_20d`, `amount_ratio_20d`, `amihud_20d`, `ret_5d`, and
`realized_vol_20d`, plus same-date `daily_basic.turnover_rate` and `total_mv`. The amount condition
uses relative amount versus its trailing 20-session average; it is not an absolute execution-volume
estimate.

Only rows whose `trade_date` equals the requested date are read. Labels, future prices, orders,
position sizes, predicted prices, and fixed stop-loss or take-profit rules are outside this layer.
