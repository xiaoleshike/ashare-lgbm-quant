# Governed Retraining Trigger

## Purpose

The trigger engine detects whether immutable production monitoring evidence justifies creating a
future training task. It does not train a model and it has no path to Registry, Champion,
Inference, Backtest, Candidate Selection, or Paper Trading state.

```text
Monitoring
    |
Retraining Request
    |
Human Review
    |
Training Orchestrator (Phase 2.8.2)
    |
Evaluation
    |
Promotion Governance
```

A training request is not a promotion request. `promotion_allowed` is always false, and no request
can alter a deployed model.

## Evidence

The engine reads only the completed date under `reports/model_monitor/YYYYMMDD/` and the immutable
performance-observation manifests already referenced by its performance monitor. It validates:

- monitor, performance, and alert artifact identities;
- file hashes recorded by their manifests;
- prospective-production observation lineage and maturity contract;
- unique `model_id + horizon` metric rows;
- model role and supported horizon.

It never reads `labels_forward`, `features_daily`, raw prices, model artifacts, or Registry state.
PSI and IC are consumed from existing diagnostics and monitoring output; they are not recomputed.

## Policy

Thresholds are versioned in `config/retraining_policy.yaml`. The default policy requires 60 mature
sessions for 5d and 10d models, 90 for 20d, and 120 for 60d. Supported reasons are:

- `alpha_decay`: monitored `alpha_decay_ratio` falls below its threshold;
- `ic_decline`: the configured rolling IC is below its threshold;
- `feature_drift`: existing maximum feature PSI exceeds its threshold;
- `critical_alert`: an unresolved model-specific CRITICAL alert exists;
- `manual_request`: an operator explicitly creates a request after the same maturity checks.

Feature drift and global operational alerts apply only to the monitored Champion identity. Each
model horizon is otherwise evaluated independently. An h10 failure never creates h5, h20, or h60
requests implicitly.

## Commands

Evaluate all monitored horizons:

```bash
ashare-quant --config config/default.yaml retraining evaluate --as-of YYYYMMDD
```

Create a manual governed request:

```bash
ashare-quant --config config/default.yaml retraining create-request \
  --model-id MODEL_ID --as-of YYYYMMDD
```

Inspect and validate requests:

```bash
ashare-quant --config config/default.yaml retraining status
ashare-quant --config config/default.yaml retraining validate --request-id REQUEST_ID
```

## Immutability and Cooldown

Requests are stored under `reports/retraining/requests/<request_id>/`. The request payload is
written before its commit manifest, and publication is transactional with
`reports/retraining/history/retraining_requests.parquet`. Logical identity is determined by model,
horizon, policy hash, evidence hash, and generation mode.

The same identity is idempotent. Changed evidence creates a different request only after the
configured cooldown. Cooldown returns `NO_ACTION_REQUIRED`/`COOLDOWN`; it never schedules delayed
training automatically.

## Production Integration

`pipeline production` invokes `retraining_evaluation` after Monitoring. It is a soft stage: source
or trigger failures are recorded as warnings and cannot change production success, candidates,
Paper Trading, Champion, or Registry state.
