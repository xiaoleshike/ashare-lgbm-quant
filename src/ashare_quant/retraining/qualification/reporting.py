"""Human-readable qualification checkpoint report."""

from ashare_quant.retraining.qualification.schemas import QualificationSnapshot


def render_qualification_report(snapshot: QualificationSnapshot) -> str:
    summary = snapshot.summary
    lines = [
        "# Controlled Operational Qualification",
        "",
        f"- Run: {summary.qualification_run_id}",
        f"- Request: {summary.request_id}",
        f"- State: {summary.current_state}",
        f"- As-of: {summary.as_of}",
        "- Qualification only: true",
        "- Promotion forbidden: true",
        "- Trading forbidden: true",
        "",
        "## Checkpoints",
        "",
    ]
    for name, checkpoint in sorted(snapshot.checkpoints.items()):
        lines.append(f"- {name}: {checkpoint.status}")
        lines.extend(f"  - warning: {warning}" for warning in checkpoint.warnings)
        if checkpoint.error:
            lines.append(f"  - error: {checkpoint.error}")
    lines.extend(
        [
            "",
            "QUALIFIED confirms controlled lifecycle integration only. It does not imply "
            "Promotion eligibility, approval, Champion replacement, or trading readiness.",
            "",
        ]
    )
    return "\n".join(lines)
