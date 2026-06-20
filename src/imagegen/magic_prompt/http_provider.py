"""HTTP magic-prompt providers (openai / anthropic / openrouter) via stdlib urllib."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ._caption_prompt import run_magic_prompt


def _bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _anthropic_auth(key: str) -> dict[str, str]:
    return {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }


def _messages(instruction: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": instruction}]


def _openai_body(model: str, instruction: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": _messages(instruction),
        "response_format": {"type": "json_object"},
    }


def _openrouter_body(model: str, instruction: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "messages": _messages(instruction),
        "response_format": {"type": "json_object"},
    }
    if "," in model:
        body["models"] = [m.strip() for m in model.split(",")]
    else:
        body["model"] = model
    return body


def _anthropic_body(model: str, instruction: str) -> dict[str, Any]:
    return {"model": model, "max_tokens": 4096, "messages": _messages(instruction)}


def _openai_extract(payload: dict[str, Any]) -> str:
    return str(payload["choices"][0]["message"]["content"])


def _anthropic_extract(payload: dict[str, Any]) -> str:
    for block in payload.get("content", []):
        if block.get("type") == "text":
            return str(block["text"])
    raise ValueError("no text block in Anthropic response")


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    url: str
    env_var: str
    default_model: str
    auth: Callable[[str], dict[str, str]]
    build_body: Callable[[str, str], dict[str, Any]]
    extract: Callable[[dict[str, Any]], str]


PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        "openai",
        "https://api.openai.com/v1/chat/completions",
        "OPENAI_API_KEY",
        "gpt-4o-mini",
        _bearer,
        _openai_body,
        _openai_extract,
    ),
    "openrouter": ProviderSpec(
        "openrouter",
        "https://openrouter.ai/api/v1/chat/completions",
        "OPENROUTER_API_KEY",
        "openrouter/free",
        _bearer,
        _openrouter_body,
        _openai_extract,
    ),
    "anthropic": ProviderSpec(
        "anthropic",
        "https://api.anthropic.com/v1/messages",
        "ANTHROPIC_API_KEY",
        "claude-haiku-4-5",
        _anthropic_auth,
        _anthropic_body,
        _anthropic_extract,
    ),
}


class HttpMagicPromptProvider:
    """Calls an OpenAI/Anthropic-style chat API and parses the JSON caption."""

    def __init__(
        self,
        spec: ProviderSpec,
        model: str,
        api_key: str,
        timeout_s: int = 120,
        transport: Callable[..., Any] = urllib.request.urlopen,
    ):
        self.spec = spec
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._transport = transport

    def expand(
        self, prompt: str, *, width: int, height: int, target_elements: int = 0
    ) -> dict[str, Any]:
        return run_magic_prompt(
            self._call_model,
            prompt=prompt,
            width=width,
            height=height,
            target_elements=target_elements,
        )

    def _call_model(self, instruction: str) -> str:
        data = json.dumps(self.spec.build_body(self.model, instruction)).encode()
        req = urllib.request.Request(
            self.spec.url, data=data, headers=self.spec.auth(self.api_key), method="POST"
        )
        try:
            with self._transport(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise RuntimeError(f"invalid API key for {self.spec.name}") from exc
            if exc.code == 429:
                raise RuntimeError(f"rate-limited by {self.spec.name}; retry shortly") from exc
            detail = exc.read().decode(errors="replace")[:300] if hasattr(exc, "read") else ""
            raise RuntimeError(f"{self.spec.name} HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"network error calling {self.spec.name}: {exc.reason}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{self.spec.name} returned a non-object response")
        return self.spec.extract(payload)
