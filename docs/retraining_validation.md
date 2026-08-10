# Retrained Challenger Validation

Phase 2.8.2C validates a governed retraining artifact before it can be used as
promotion evidence. Validation is read-only and does not register, promote, or
deploy the model.

Qualification validation reuses the same artifact, offline, executable, leakage, and Shadow
eligibility validators. It requires exact model/registration `qualification_run_id` lineage and
publishes qualification-only evidence without creating a Promotion Request.

After validation, Shadow enrollment requires a separate authorization bound to the exact validation
snapshot. Validation itself remains explicitly advanced but does not consume a privileged-stage
authorization.

## Workflow

1. Validate the immutable model, dataset, execution, and candidate-registration
   manifests.
2. Resolve the selection-only walk-forward fold used by training. A fold that
   appears in the final-test period is rejected.
3. Score the fold's evaluation period and compute post-hoc Rank IC, ICIR,
   positive-IC ratio, Top-N observation metrics, yearly stability, and market
   regime metrics.
4. Apply the shared accounting-schema-v2 next-open portfolio simulator with the model horizon,
   effective-dated cost policy, suspension valuation, and price-limit constraints. Candidate
   evaluation must begin after its immutable selection fold; every position must have a complete
   executable or explicitly terminal lifecycle.
5. Verify eligibility for a future prospective shadow-prediction adapter without
   generating a production prediction.
6. Atomically publish immutable evidence with the root manifest written last.

The offline evaluation may read mature `labels_forward` rows only after the
selection/historical-holdout isolation checks pass. Older immutable manifests retain the
`final_test_period` field, but current research policy does not classify that repeatedly inspected
history as a pristine lockbox. The executable validation is
label-free and does not modify paper-trading state.

Executable evidence freezes `accounting_schema_version=2`, the complete execution-cost schedule,
`cost_policy_hash`, accounting diagnostics, and corrected compounded metrics. Unexpected market-data
gaps, unresolved positions, invalid costs, unsupported execution modes, and accounting invariant
failures block validation. Old executable evidence remains immutable but is not accepted as current
Promotion evidence.

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
