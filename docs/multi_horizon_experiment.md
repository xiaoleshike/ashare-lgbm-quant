# Multi-Horizon Experiment Planning

Multi-horizon challengers test whether alpha persists over distinct investment horizons instead of assuming that a five-day target describes every holding period. The configured 5d, 10d, 20d, and 60d experiments remain independent: each future model owns one label horizon, one matching holding period, one execution rule, and one experiment ID. Targets must never be mixed in one training dataset or selected using future test results.

This phase creates plans only. It inspects label availability counts and date coverage but never loads target values into the planner. It does not train a model, modify the Champion, produce predictions, or generate trading signals. All experiments inherit the Champion's frozen feature hash for comparability and record the current universe manifest hash.

The local `labels_forward` artifact must contain available 5d, 10d, 20d, and 60d rows before planning. Build those labels separately; the horizon planner never invokes the label builder.

```bash
ashare-quant --config config/default.yaml labels build \
  --start-date 20100101 \
  --end-date 20260710 \
  --horizons 5,10,20,60
```

## Walk-forward Integration

All horizons reference the same existing walk-forward fold plan so their later challenger results use identical evaluation dates. Because the 60d label enters at T+1 and matures 60 sessions later, the shared plan must have at least 61 purge and embargo sessions. Prepare that plan separately:

The fold manifest must use horizon-agnostic schema version 2. Older manifests containing `label_horizon` or `label_exit_lag_sessions` are rejected and must be regenerated.

```bash
ashare-quant --config config/default.yaml models walk-forward-plan \
  --start-date 20100101 \
  --end-date 20260717 \
  --scheme expanding \
  --purge-days 61 \
  --embargo-days 61
```

Then create the multi-horizon plan, either by automatic compatible-plan discovery or an explicit immutable reference:

```bash
ashare-quant --config config/default.yaml models horizon-plan

ashare-quant --config config/default.yaml models horizon-plan \
  --folds-manifest reports/walk_forward/<run_id>/manifest.json
```

The result is `reports/horizon_experiments/<run_id>/experiment_manifest.json`. Each horizon record contains its future label name, maturity and required gap sessions, holding period, execution rule, feature and universe hashes, and the referenced fold manifest. Evaluation ranges are clipped to the latest signal date whose label can mature under the authoritative `trade_cal`.

`selection_period` folds may be used to compare challengers. `final_test_period` folds are reserved for one final evaluation and explicitly carry `may_select_model=false`. A later challenger trainer must consume one horizon record at a time and must not combine targets or use final-test results for selection.

## Production Observation

Existing daily predictions and candidates can be recorded without trading logic:

```bash
ashare-quant --config config/default.yaml models observation-log --as-of 20260717
```

This writes `reports/production_observation/20260717.json` with the model ID, candidate count, Top-10/20/50 ranked candidates, source fingerprints, and empty future-return placeholders. It does not calculate returns, generate orders, or create new signals.

The downstream selection-fold-only training contract is documented in
[`challenger_training.md`](challenger_training.md). It consumes this plan
without rebuilding folds and leaves final-test periods unread.
