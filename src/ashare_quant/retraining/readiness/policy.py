"""Conservative execution-readiness policy."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RetrainingReadinessPolicy(BaseModel):
    """Operational freshness bounds applied before any future training job."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_scheduler_age_hours: int = Field(default=72, gt=0)
