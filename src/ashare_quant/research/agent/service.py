"""Isolated research-agent orchestration over immutable report artifacts."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ashare_quant.config.settings import ResearchAgentSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import canonical_payload_hash
from ashare_quant.research.agent.adapters import LLMAdapter, build_adapter
from ashare_quant.research.agent.collector import collect_artifacts
from ashare_quant.research.agent.configuration import (
    ProviderConfiguration,
    provider_configuration,
)
from ashare_quant.research.agent.context import build_research_context, context_hash
from ashare_quant.research.agent.fallback import deterministic_fallback
from ashare_quant.research.agent.prompts import build_prompts, prompt_hash
from ashare_quant.research.agent.rendering import render_daily_research
from ashare_quant.research.agent.schemas import (
    ResearchAgentResult,
    ResearchAgentSummary,
    ResearchAgentValidationResult,
    ResearchContext,
)
from ashare_quant.research.agent.storage import (
    load_summary_payload,
    publish_research_agent,
    read_complete_output,
)
from ashare_quant.research.agent.validation import (
    parse_and_validate_summary,
    validate_summary,
)
from ashare_quant.utils.manifest import config_hash, current_git_info, utc_now_iso

AdapterFactory = Callable[[ProviderConfiguration], LLMAdapter]
Sleeper = Callable[[float], None]


class ResearchAgentService:
    """Generate grounded research summaries without model or trading access."""

    def __init__(
        self,
        *,
        settings: ResearchAgentSettings,
        config_path: Path,
        reports_root: Path,
        adapter_factory: AdapterFactory = build_adapter,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        self.settings = settings
        self.config_path = config_path
        self.reports_root = reports_root
        self.adapter_factory = adapter_factory
        self.sleeper = sleeper

    def generate(self, as_of: str) -> ResearchAgentResult:
        """Collect, summarize, validate, and atomically publish one report."""

        collected = collect_artifacts(self.reports_root, as_of)
        context = build_research_context(
            collected,
            top_candidates=self.settings.top_candidates,
        )
        system_prompt, user_prompt = build_prompts(
            context,
            self.settings.prompt_version,
            allow_advisory_language=self.settings.allow_advisory_language,
        )
        provider = provider_configuration(self.settings)
        identity = self._identity(
            collected.source_hashes,
            context,
            provider,
            system_prompt,
            user_prompt,
        )
        run_id = f"research_agent_{as_of}_{identity[:16]}"
        output_dir = self.output_dir(as_of)
        existing = read_complete_output(output_dir)
        if existing is not None:
            if existing.get("identity_hash") != identity:
                raise DataValidationError(
                    "existing research-agent output has different immutable source identity"
                )
            return ResearchAgentResult(
                as_of,
                output_dir,
                str(existing["generation_mode"]),
                str(existing["run_id"]),
                idempotent=True,
            )
        if output_dir.exists():
            raise DataValidationError(f"incomplete research-agent output exists: {output_dir}")
        summary, mode, fallback_reason, response_hash = self._generate_summary(
            context,
            provider,
            system_prompt,
            user_prompt,
        )
        validate_summary(
            summary,
            context,
            allow_advisory_language=self.settings.allow_advisory_language,
        )
        summary_payload = {
            "schema_version": 1,
            "artifact_name": "daily_llm_research_summary",
            "as_of": as_of,
            "generation_mode": mode,
            "summary": summary.model_dump(mode="json"),
        }
        markdown = render_daily_research(as_of, summary, mode)
        git = current_git_info()
        manifest = {
            "schema_version": 1,
            "artifact_name": "llm_research_agent",
            "as_of": as_of,
            "run_id": run_id,
            "identity_hash": identity,
            "source_artifacts": [
                {
                    "name": name,
                    "path": collected.source_paths[name],
                    "sha256": digest,
                }
                for name, digest in sorted(collected.source_hashes.items())
            ],
            "source_hash": canonical_payload_hash(collected.source_hashes),
            "context_hash": context_hash(context),
            "prompt_hash": prompt_hash(
                self.settings.prompt_version,
                allow_advisory_language=self.settings.allow_advisory_language,
            ),
            "request_hash": canonical_payload_hash({"system": system_prompt, "user": user_prompt}),
            "response_hash": response_hash,
            "provider": provider.provider,
            "model": provider.model,
            "generation_mode": mode,
            "fallback_reason": fallback_reason,
            "generated_at": utc_now_iso(),
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "config_hash": config_hash(self.config_path),
            "agent_config_hash": canonical_payload_hash(self.settings.model_dump(mode="json")),
            "labels_read": False,
            "inference_called": False,
            "models_modified": False,
            "registry_modified": False,
            "trading_modified": False,
            "advisory_language_allowed": self.settings.allow_advisory_language,
            "automatic_execution_allowed": False,
            "status": "success" if mode == "llm" else "success_with_fallback",
        }
        publish_research_agent(
            output_dir=output_dir,
            summary_payload=summary_payload,
            markdown=markdown,
            manifest=manifest,
        )
        return ResearchAgentResult(as_of, output_dir, mode, run_id)

    def validate(self, as_of: str) -> ResearchAgentValidationResult:
        """Validate current sources and an existing output without generation."""

        output_dir = self.output_dir(as_of)
        try:
            collected = collect_artifacts(self.reports_root, as_of)
            context = build_research_context(
                collected,
                top_candidates=self.settings.top_candidates,
            )
            manifest = read_complete_output(output_dir)
            if manifest is None:
                return ResearchAgentValidationResult(
                    as_of, False, output_dir.exists(), error="research-agent output is missing"
                )
            _validate_published_sources(manifest, collected.source_hashes, as_of)
            payload = load_summary_payload(output_dir)
            _validate_summary_payload(
                payload,
                context,
                as_of,
                allow_advisory_language=self.settings.allow_advisory_language,
            )
        except (DataValidationError, OSError, ValueError, ValidationError) as error:
            return ResearchAgentValidationResult(
                as_of,
                False,
                output_dir.exists(),
                error=str(error),
            )
        return ResearchAgentValidationResult(
            as_of,
            True,
            True,
            str(manifest["generation_mode"]),
        )

    def status(self, as_of: str) -> ResearchAgentValidationResult:
        """Validate only the published output and its physical hashes."""

        output_dir = self.output_dir(as_of)
        try:
            manifest = read_complete_output(output_dir)
        except (DataValidationError, OSError, ValueError) as error:
            return ResearchAgentValidationResult(
                as_of, False, output_dir.exists(), error=str(error)
            )
        if manifest is None:
            return ResearchAgentValidationResult(
                as_of, False, output_dir.exists(), error="research-agent output is missing"
            )
        return ResearchAgentValidationResult(as_of, True, True, str(manifest["generation_mode"]))

    def output_dir(self, as_of: str) -> Path:
        return self.reports_root / "research_agent" / as_of

    def _generate_summary(
        self,
        context: ResearchContext,
        provider: ProviderConfiguration,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[ResearchAgentSummary, str, str | None, str | None]:
        if not self.settings.enabled:
            return deterministic_fallback(context), "deterministic_fallback", "disabled", None
        if not provider.api_key:
            return (
                deterministic_fallback(context),
                "deterministic_fallback",
                "api_key_unavailable",
                None,
            )
        error_name = "llm_failure"
        for attempt in range(provider.max_retries + 1):
            try:
                response = self.adapter_factory(provider).generate(system_prompt, user_prompt)
                summary = parse_and_validate_summary(
                    response,
                    context,
                    allow_advisory_language=self.settings.allow_advisory_language,
                )
                return summary, "llm", None, canonical_payload_hash(response)
            except Exception as error:  # provider and output failures use the safe fallback
                error_name = type(error).__name__
                if attempt < provider.max_retries:
                    self.sleeper(float(2**attempt))
        return (
            deterministic_fallback(context),
            "deterministic_fallback",
            error_name,
            None,
        )

    def _identity(
        self,
        source_hashes: dict[str, str],
        context: ResearchContext,
        provider: ProviderConfiguration,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        return canonical_payload_hash(
            {
                "as_of": context.as_of,
                "source_hashes": source_hashes,
                "context_hash": context_hash(context),
                "prompt_hash": prompt_hash(
                    self.settings.prompt_version,
                    allow_advisory_language=self.settings.allow_advisory_language,
                ),
                "request_hash": canonical_payload_hash(
                    {"system": system_prompt, "user": user_prompt}
                ),
                "provider": provider.provider,
                "model": provider.model,
                "agent_config_hash": canonical_payload_hash(self.settings.model_dump(mode="json")),
            }
        )


def _validate_summary_payload(
    payload: dict[str, Any],
    context: ResearchContext,
    as_of: str,
    *,
    allow_advisory_language: bool,
) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_name") != "daily_llm_research_summary"
        or payload.get("as_of") != as_of
    ):
        raise DataValidationError("invalid published research-agent summary identity")
    try:
        summary = ResearchAgentSummary.model_validate(payload.get("summary"))
    except ValidationError as error:
        raise DataValidationError(f"invalid published research-agent summary: {error}") from error
    validate_summary(
        summary,
        context,
        allow_advisory_language=allow_advisory_language,
    )


def _validate_published_sources(
    manifest: dict[str, Any],
    current_hashes: dict[str, str],
    as_of: str,
) -> None:
    if manifest.get("as_of") != as_of:
        raise DataValidationError("research-agent manifest date differs from requested date")
    sources = manifest.get("source_artifacts")
    if not isinstance(sources, list):
        raise DataValidationError("research-agent manifest lacks source artifacts")
    recorded = {
        str(item.get("name")): str(item.get("sha256")) for item in sources if isinstance(item, dict)
    }
    if recorded != current_hashes:
        raise DataValidationError("research-agent source artifacts changed after publication")
