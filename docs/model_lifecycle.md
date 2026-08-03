# Model Lifecycle Operations

## Promotion Procedure

The governed lifecycle is:

```text
Candidate
  -> Evidence Freeze
  -> Promotion Request
  -> Gate Validation
  -> Human Review
  -> Approval
  -> Apply
  -> Monitoring
```

Create a request by binding all required immutable evidence:

```bash
ashare-quant --config config/default.yaml models promotion create \
  --model-id MODEL_ID \
  --evidence-cutoff-date YYYYMMDD \
  --challenger-evaluation PATH/manifest.json \
  --executable-validation PATH/manifest.json \
  --shadow-prediction PATH/manifest.json \
  --performance-observation PATH/manifest.json \
  --monitoring-summary PATH/monitor_summary.json \
  --alerts PATH/manifest.json
```

Evaluate the gate and inspect its evidence:

```bash
ashare-quant --config config/default.yaml models promotion validate --request-id REQUEST_ID
ashare-quant --config config/promotion_review.yaml models promotion review \
  --request-id REQUEST_ID
```

Approve only after the reviewer verifies evidence cutoff, prospective observation coverage,
deployment compatibility, alerts, and operational readiness:

```bash
ashare-quant --config config/promotion_review.yaml models promotion approve \
  --request-id REQUEST_ID --comments-file review.txt
ashare-quant --config config/promotion_review.yaml models promotion apply \
  --request-id REQUEST_ID
```

Reject a request when evidence is incomplete, hashes or lineage differ, observations are immature,
the deployment contract is incompatible, the gate fails, or unresolved critical alerts exist:

```bash
ashare-quant --config config/promotion_review.yaml models promotion reject \
  --request-id REQUEST_ID --comments-file reason.txt
```

Approval and apply are separate. Neither gate metrics nor monitoring can promote a model
automatically. Applying preserves the prior Champion artifact and writes a new registry version and
Champion assignment.

## Rollback Procedure

Rollback is a separate governed transition and only targets a historical Champion in the same
deployment slot:

```bash
ashare-quant --config config/promotion_review.yaml models promotion rollback-create \
  --model-id HISTORICAL_CHAMPION_ID --reason-file rollback_reason.txt
ashare-quant --config config/promotion_review.yaml models promotion rollback-validate \
  --request-id REQUEST_ID
ashare-quant --config config/promotion_review.yaml models promotion review \
  --request-id REQUEST_ID
ashare-quant --config config/promotion_review.yaml models promotion approve \
  --request-id REQUEST_ID --comments-file rollback_review.txt
ashare-quant --config config/promotion_review.yaml models promotion rollback-apply \
  --request-id REQUEST_ID
```

Rollback is never automatic and cannot target an arbitrary candidate. Registry, artifact, policy,
or approval hash changes invalidate the transition.
