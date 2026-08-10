"""Immutable research-access policy and prospective lockbox enforcement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ashare_quant.data.exceptions import DataValidationError

type ResearchConsumer = Literal[
    "feature_selection",
    "hyperparameter_tuning",
    "model_family_selection",
    "horizon_selection",
    "fold_selection",
    "research_threshold_tuning",
    "walk_forward_evaluation",
    "production_shadow",
    "performance_observation",
    "challenger_paper_trading",
    "promotion_evidence",
]


class HistoricalHoldoutPolicy(BaseModel):
    """Classification for historical data already exposed to research."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    classification: Literal["HISTORICAL_HOLDOUT"]
    start_date: str = Field(pattern=r"^\d{8}$")
    end_date: str = Field(pattern=r"^\d{8}$")


class ProspectiveLockboxPolicy(BaseModel):
    """Allowed and forbidden uses of naturally arriving future evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    start_date: str = Field(pattern=r"^\d{8}$")
    allowed_consumers: tuple[ResearchConsumer, ...]
    forbidden_consumers: tuple[ResearchConsumer, ...]


class ResearchPolicy(BaseModel):
    """Versioned research-validity contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1]
    policy_version: str = Field(min_length=1)
    historical_holdout: HistoricalHoldoutPolicy
    prospective_lockbox: ProspectiveLockboxPolicy
    governed_training_access: Literal["NOT_YET_ENABLED"]

    @property
    def policy_hash(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(encoded.encode()).hexdigest()


def load_research_policy(path: Path) -> ResearchPolicy:
    """Load and validate one explicit research policy file."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise DataValidationError(f"cannot read research policy {path}: {error}") from error
    try:
        policy = ResearchPolicy.model_validate(payload)
    except ValueError as error:
        raise DataValidationError(f"invalid research policy {path}: {error}") from error
    if policy.historical_holdout.end_date >= policy.prospective_lockbox.start_date:
        raise DataValidationError("historical holdout must end before prospective lockbox starts")
    if set(policy.prospective_lockbox.allowed_consumers) & set(
        policy.prospective_lockbox.forbidden_consumers
    ):
        raise DataValidationError("research policy consumer cannot be both allowed and forbidden")
    return policy


def enforce_research_window(
    policy: ResearchPolicy,
    *,
    consumer: ResearchConsumer,
    start_date: str,
    end_date: str,
) -> None:
    """Fail closed when a forbidden research consumer reaches the lockbox."""

    if start_date > end_date:
        raise DataValidationError("research window is reversed")
    lockbox = policy.prospective_lockbox
    if consumer in lockbox.forbidden_consumers and end_date >= lockbox.start_date:
        raise DataValidationError(
            "RESEARCH_LOCKBOX_VIOLATION: "
            f"consumer={consumer} window={start_date}..{end_date} "
            f"lockbox_start={lockbox.start_date}"
        )
