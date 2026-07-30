"""Provider configuration without secret persistence."""

from __future__ import annotations

import os
from dataclasses import dataclass

from ashare_quant.config.settings import ResearchAgentSettings

_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


@dataclass(frozen=True, slots=True)
class ProviderConfiguration:
    """Runtime provider settings with an in-memory API key."""

    provider: str
    model: str
    api_key: str | None
    temperature: float
    timeout_seconds: int
    max_retries: int
    max_output_tokens: int


def provider_configuration(settings: ResearchAgentSettings) -> ProviderConfiguration:
    """Read only the selected provider's key from its environment variable."""

    return ProviderConfiguration(
        provider=settings.provider,
        model=settings.model,
        api_key=os.environ.get(_KEY_ENV[settings.provider]),
        temperature=settings.temperature,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
        max_output_tokens=settings.max_output_tokens,
    )


def provider_key_environment(provider: str) -> str:
    """Return the documented environment variable name, never its value."""

    return _KEY_ENV[provider]
