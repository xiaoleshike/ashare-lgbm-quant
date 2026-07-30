"""Provider-neutral structured LLM adapters using the standard library."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Protocol

from ashare_quant.research.agent.configuration import ProviderConfiguration


class LLMAdapter(Protocol):
    """Minimal provider-neutral generation contract."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the provider's response text."""


def build_adapter(configuration: ProviderConfiguration) -> LLMAdapter:
    """Build an adapter without exposing the API key to artifacts or logs."""

    if not configuration.api_key:
        raise ValueError("provider API key is unavailable")
    if configuration.provider in {"openai", "deepseek"}:
        endpoint = (
            "https://api.openai.com/v1/chat/completions"
            if configuration.provider == "openai"
            else "https://api.deepseek.com/chat/completions"
        )
        return OpenAICompatibleAdapter(configuration, endpoint)
    if configuration.provider == "claude":
        return ClaudeAdapter(configuration)
    if configuration.provider == "gemini":
        return GeminiAdapter(configuration)
    raise ValueError(f"unsupported research-agent provider: {configuration.provider}")


class OpenAICompatibleAdapter:
    """OpenAI and DeepSeek chat-completions transport."""

    def __init__(self, configuration: ProviderConfiguration, endpoint: str) -> None:
        self.configuration = configuration
        self.endpoint = endpoint

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.configuration.model,
            "temperature": self.configuration.temperature,
            "max_tokens": self.configuration.max_output_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = _post_json(
            self.endpoint,
            payload,
            {
                "Authorization": f"Bearer {self.configuration.api_key}",
                "Content-Type": "application/json",
            },
            self.configuration.timeout_seconds,
        )
        return str(response["choices"][0]["message"]["content"])


class ClaudeAdapter:
    """Anthropic Messages API transport."""

    def __init__(self, configuration: ProviderConfiguration) -> None:
        self.configuration = configuration

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = _post_json(
            "https://api.anthropic.com/v1/messages",
            {
                "model": self.configuration.model,
                "system": system_prompt,
                "temperature": self.configuration.temperature,
                "max_tokens": self.configuration.max_output_tokens,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            {
                "x-api-key": str(self.configuration.api_key),
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            self.configuration.timeout_seconds,
        )
        return "".join(str(item["text"]) for item in response["content"] if "text" in item)


class GeminiAdapter:
    """Google Gemini generateContent transport."""

    def __init__(self, configuration: ProviderConfiguration) -> None:
        self.configuration = configuration

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        model = urllib.parse.quote(self.configuration.model, safe="")
        key = urllib.parse.quote(str(self.configuration.api_key), safe="")
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:"
            f"generateContent?key={key}"
        )
        response = _post_json(
            endpoint,
            {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": self.configuration.temperature,
                    "maxOutputTokens": self.configuration.max_output_tokens,
                    "responseMimeType": "application/json",
                },
            },
            {"Content-Type": "application/json"},
            self.configuration.timeout_seconds,
        )
        return "".join(
            str(part["text"])
            for part in response["candidates"][0]["content"]["parts"]
            if "text" in part
        )


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: int,
) -> dict[str, Any]:
    request = urllib.request.Request(  # noqa: S310 -- fixed HTTPS provider endpoints
        url,
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("LLM provider response must be a JSON object")
    return value
