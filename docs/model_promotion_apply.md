# Controlled Model Promotion Apply

Phase 2.7.5D is the only promotion-governance operation that changes the current model
registry. It does not train, score, backtest, or automatically select a model.

## Preconditions

An apply requires all of the following to remain unchanged:

- the immutable promotion request and deployment contract;
- a `PASS` or human-approved `REVIEW_REQUIRED` gate result;
- one unexpired `APPROVED` review event;
- the model registry and current Champion assignment;
- the candidate's `model.txt`, `feature_list.json`, `manifest.json`, and `metrics.json`.

Promotion requests created before model artifact hashes were added to the deployment contract
cannot be applied safely. Create a new request, rerun its gate, and obtain a new approval.

## Apply

```bash
ashare-quant --config config/promotion_review.yaml models promotion apply \
  --request-id REQUEST_ID
```

The command acquires `runs/.production.lock` before `models/.registry.lock`. It writes an
immutable old-registry snapshot and new registry version, atomically switches
`models/registry.json`, records Champion assignment history, and writes the apply manifest
last. The previous Champion artifact is never changed or removed.

Inspect status without changing state:

```bash
ashare-quant --config config/promotion_review.yaml models promotion apply-status \
  --request-id REQUEST_ID
```

Apply is idempotent for an already committed request. A changed Registry, approval, request,
gate result, deployment contract, or model artifact causes a hard failure before publication.
An interrupted pending switch is restored to its approved parent Registry before a retry.

Rollback is intentionally not implemented in this phase.
