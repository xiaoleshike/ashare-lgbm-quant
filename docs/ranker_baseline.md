# LightGBM Ranker Baseline

Phase 7 runs two fixed-parameter `LGBMRanker` experiments. Experiment A consumes exactly the
`top_50` list from the latest feature diagnostics report. Experiment B consumes the manually
maintained subset in `config/feature_sets/robust_features.json`. The robust list is never rewritten
from validation or test results.

## Target and Groups

Each `trade_date` is one ranking group. Rows must be in `in_model_universe` and have an available
five-trading-day `future_excess_ret` label. LightGBM `lambdarank` requires non-negative integer
relevance labels, so continuous future excess returns are ranked within each date and mapped to
five deterministic relevance grades. Evaluation Rank IC and top-bucket returns always use the
original continuous `future_excess_ret_5d`, not the grade.

The fixed periods are:

- Train: `20100101` through `20191231`
- Validation: `20200101` through `20221231`
- Test: `20230101` through `20260710`

The model is fitted only on train data. Validation reports fixed-baseline results; there is no
hyperparameter search or early stopping. Test data is loaded only after fitting and validation.

Ranker training uses the explicit `ranker.training_backend` configuration. CPU is the default;
CUDA changes only LightGBM's execution backend and preserves all semantic parameters, including the
existing `max_bin` behavior. See `docs/training_compute_backend.md`. Diagnostics and inference stay
on CPU.

## Run

```bash
ashare-quant --config config/default.yaml models ranker-baseline
```

Use `--recommended-features` or `--robust-features` only to point at reviewed JSON artifacts. Use
`--processed-root` and `--output-root` before the subcommand to select alternate stores.

Each experiment is atomically published under `models/<experiment-id>/` with:

- `model.txt`: native LightGBM model;
- `feature_list.json`: ordered features and SHA256 hash;
- `metrics.json`: validation/test Rank IC, ICIR, NDCG@10/50, top 5%/10% mean future excess return,
  yearly stability, and gain/split importance;
- `manifest.json`: Git/config identity, fixed parameters, split dates, target semantics, source
  artifact manifests, and requested/effective training-compute provenance.

The top-bucket returns are equal-weighted cross-sectional ranking proxies. Five-day labels overlap,
and the report includes no costs, execution simulation, holdings, or portfolio accounting. These
metrics must not be presented as a production backtest.
