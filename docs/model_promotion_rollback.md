# Governed Champion Rollback

Phase 2.7.5E restores a historical Champion only after an immutable request,
validation, and authorized human approval. Monitoring metrics and alerts never
trigger rollback automatically.

## Preconditions

The target model must:

- be registered with status `retired`;
- appear in `models/champion_history/` for the requested deployment slot;
- retain all required immutable model artifacts;
- match its registered feature hash;
- use the same horizon, holding period, and execution rule as the current Champion.

Legacy model manifests may provide `label_horizon` instead of `horizon`. When a
legacy manifest omits a duplicate holding-period field, holding is constrained to
that horizon. An omitted execution rule may only inherit the current Champion's
explicit contract for the same deployment slot; if neither artifact declares one,
validation fails rather than using a configuration default.

The request freezes the Registry hash and hashes of `model.txt`,
`feature_list.json`, `manifest.json`, and `metrics.json`.

## Workflow

Create a reason file, then create and validate the request:

```bash
ashare-quant --config config/promotion_review.yaml models promotion rollback-create \
  --model-id HISTORICAL_MODEL_ID \
  --reason-file reason.txt

ashare-quant --config config/promotion_review.yaml models promotion rollback-validate \
  --request-id ROLLBACK_REQUEST_ID
```

Use the existing OS-identity human review commands:

```bash
ashare-quant --config config/promotion_review.yaml models promotion review \
  --request-id ROLLBACK_REQUEST_ID

ashare-quant --config config/promotion_review.yaml models promotion approve \
  --request-id ROLLBACK_REQUEST_ID \
  --comments-file review.txt
```

Apply explicitly:

```bash
ashare-quant --config config/promotion_review.yaml models promotion rollback-apply \
  --request-id ROLLBACK_REQUEST_ID
```

Apply acquires `runs/.production.lock` before `models/.registry.lock`. It creates
a new immutable Registry version, retires the current Champion, restores the
target as Champion, appends Champion history, atomically switches
`models/registry.json`, and writes `rollback_apply_manifest.json` last.

The original request, validation, approval, Registry versions, Champion history,
and all model artifacts remain immutable. Repeating the same completed apply is
idempotent. Registry, review policy, approval, or model artifact changes invalidate
the operation before state transition.
