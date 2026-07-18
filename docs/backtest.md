# Executable Portfolio Backtest

Phase 8 converts saved Ranker model scores into an executable daily portfolio simulation. It is a historical backtest framework, not a model-training or hyperparameter-tuning stage.

## Inputs

The backtest reads:

- `models/<experiment_id>/model.txt`
- `models/<experiment_id>/feature_list.json`
- processed `features_daily`
- processed `universe_daily`
- raw `daily`, `stk_limit`, `trade_cal`, and `index_daily`

The default model directory is the latest `models/experiment_b_robust_*`. Pass `--model-dir` to select a specific artifact.

## Execution Assumptions

Signals are generated after the close on signal date `T`. Buys are attempted at the next trading day's open. Positions are sold at the open after `backtest.holding_period_days`, default 5 trading days.

The simulator rejects buys when the entry date is suspended, ST, missing an open price, or opens at/above the limit-up price within tolerance. It rejects sells when the exit date is suspended, missing an open price, or opens at/below the limit-down price. Unsellable positions remain held and are retried.

## Costs

Default costs are configured in `config/default.yaml`:

- commission: `0.00025`
- stamp duty on sells: `0.001`
- slippage: `0.0005`

## Outputs

Run:

```bash
ashare-quant --config config/default.yaml backtest run \
  --model-dir models/<experiment_id> \
  --start-date 20230101 \
  --end-date 20260710
```

Outputs are written to `backtests/<experiment_id>_backtest_<timestamp>/`:

- `daily_returns.csv`
- `predictions.csv`
- `trades.csv`
- `holdings.csv`
- `metrics.json`
- `manifest.json`

`predictions.csv` is written before portfolio execution and contains `trade_date`, `ts_code`,
`prediction_score`, `rank`, and `selected_flag`. `selected_flag` marks stocks inside the largest
configured Top-N bucket for that signal date.

The default Top-N variants are `10`, `20`, and `50`. Override with `--top-n 20`.
