# Retrained Challenger Validation

Phase 2.8.2C validates a governed retraining artifact before it can be used as
promotion evidence. Validation is read-only and does not register, promote, or
deploy the model.

## Workflow

1. Validate the immutable model, dataset, execution, and candidate-registration
   manifests.
2. Resolve the selection-only walk-forward fold used by training. A fold that
   appears in the final-test period is rejected.
3. Score the fold's evaluation period and compute post-hoc Rank IC, ICIR,
   positive-IC ratio, Top-N observation metrics, yearly stability, and market
   regime metrics.
4. Apply the existing next-open portfolio simulator with the model horizon,
   configured commission, stamp duty, slippage, suspension, and price-limit
   constraints. Every signal date must have a complete executable exit.
5. Verify eligibility for a future prospective shadow-prediction adapter without
   generating a production prediction.
6. Atomically publish immutable evidence with the root manifest written last.

The offline evaluation may read mature `labels_forward` rows only after the
selection/final-test isolation checks pass. The executable validation is
label-free and does not modify paper-trading state.

## Commands

```bash
ashare-quant --config config/default.yaml retraining validate \
  --model-id MODEL_ID

ashare-quant --config config/default.yaml retraining validation-status \
  --run-id VALIDATION_RUN_ID
```

The existing request validation remains available:

```bash
ashare-quant --config config/default.yaml retraining validate \
  --request-id REQUEST_ID
```

Outputs are stored below
`reports/retraining_validation/<validation_run_id>/`. `promotion_ready` means
the validation evidence is complete enough for later governance review. It does
not create a promotion request or change model status.
