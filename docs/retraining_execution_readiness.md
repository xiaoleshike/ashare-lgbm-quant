# Retraining Execution Readiness

Phase 2.8.2A is a read-only hard gate between an immutable governed training request and any
future training executor. It does not read features or labels and cannot start LightGBM,
inference, backtesting, paper trading, promotion, or Registry changes.

## Validation order

1. The configured systemd production service/timer and latest successful scheduler invocation.
2. The exact successful, non-dry-run production run and immutable closed-loop manifest.
3. The dated governance snapshot, current Registry hash, Champion assignment, recovery state,
   and promotion transaction state.
4. The current promotion policy against the policy frozen in governance lineage.
5. The retraining request, trigger policy, immutable evidence, model identity, horizon, and
   promotion policy binding.

Any failed check returns `FAILED` and later checks are `NOT_RUN`. Promotion policy drift is
reported explicitly as `FAILED_POLICY_DRIFT`.

## Usage

When the date has exactly one training request:

```bash
ashare-quant --config config/default.yaml retraining readiness --as-of 20260731
```

When multiple horizons generated requests on the same date, bind the intended request:

```bash
ashare-quant --config config/default.yaml retraining readiness \
  --as-of 20260731 \
  --request-id training_REQUEST_ID
```

Exit code `0` means `READY`; exit code `1` means the immutable readiness report was published
with `FAILED`. Execution errors return `2`.

Artifacts are written under `reports/retraining/readiness/YYYYMMDD/`. The JSON and Markdown are
staged and validated before `manifest.json` is written as the commit marker. A complete existing
artifact is idempotently reused only when its logical identity and file hashes match. A different
identity is never allowed to overwrite it.

## Policy lineage migration

New governance snapshots and retraining requests record the promotion policy version and hash.
Legacy artifacts without this binding are intentionally not execution-ready. Run a successful
production closed loop to publish a current governance snapshot, then evaluate retraining again
to create a policy-bound request. Do not edit old artifacts.

`READY` is only authorization for the future Phase 2.8.2 executor to begin its own validation. It
does not train, register, promote, or deploy a model.
