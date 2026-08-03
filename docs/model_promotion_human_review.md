# Model Promotion Human Review

Phase 2.7.5C records an immutable human approval or rejection. It does not apply a
promotion or modify the model registry.

## Reviewer policy

The default reviewer allowlist is empty. Configure a dedicated review config to avoid
changing the hash of `config/default.yaml` used by production artifacts:

```yaml
promotion:
  reviewer_allowlist:
    - reviewer_linux_user
  allow_requester_as_reviewer: false
  review_expire_hours: 72
```

The reviewer is always derived from the current Linux account. There is no CLI option
that can override reviewer identity. New promotion requests record their OS requester;
legacy requests without requester identity cannot be approved.

## Workflow

```bash
ashare-quant --config config/promotion_review.yaml models promotion review \
  --request-id REQUEST_ID

ashare-quant --config config/promotion_review.yaml models promotion approve \
  --request-id REQUEST_ID \
  --comments-file review.txt

ashare-quant --config config/promotion_review.yaml models promotion reject \
  --request-id REQUEST_ID \
  --comments-file reason.txt

ashare-quant --config config/promotion_review.yaml models promotion review-status \
  --request-id REQUEST_ID
```

Events are stored under
`models/promotion_requests/<request_id>/approval_events/`. The event JSON is written
before its completion manifest. A request can have only one terminal review event.

An approval becomes invalid when its request, gate result, registry snapshot, or review
policy changes. Approved events expire after the configured TTL. Rejected events remain
as permanent audit records.
