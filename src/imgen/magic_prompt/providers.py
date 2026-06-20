"""Magic-prompt provider factory + CLI-flag resolution."""

from __future__ import annotations

import os

import click

from ..config import Config, Secrets
from .base import MagicPromptProvider
from .cli_provider import CliMagicPromptProvider
from .http_provider import PROVIDERS, HttpMagicPromptProvider

CLI_PROVIDERS = {"codex", "claude", "pi"}
HTTP_PROVIDERS = set(PROVIDERS)  # {"openai", "anthropic", "openrouter"}
ALL_PROVIDERS = CLI_PROVIDERS | HTTP_PROVIDERS

DEFAULT_MODELS = {
    "codex": "gpt-5.5",
    "claude": "sonnet",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "openrouter": "openrouter/free",
}  # pi has no default — it needs an explicit "<pi-provider>/<model>".


def resolve_magic_provider(config: Config) -> tuple[str, str]:
    """Effective (provider, model) from persisted [magic_prompt] config, else defaults."""
    provider = config.magic_prompt_provider() or "codex"
    model = config.magic_prompt_model()
    if not model or config.magic_prompt_provider() != provider:
        model = DEFAULT_MODELS.get(provider)
    if model is None:
        raise click.ClickException(f"no magic-prompt model for {provider!r}; set one with config")
    return provider, model


def make_magic_provider(provider: str, model: str, *, secrets: Secrets) -> MagicPromptProvider:
    if provider in CLI_PROVIDERS:
        return CliMagicPromptProvider(provider, model)
    if provider in HTTP_PROVIDERS:
        spec = PROVIDERS[provider]
        key = os.environ.get(spec.env_var) or secrets.api_key(provider)
        if not key:
            raise click.ClickException(
                f"no API key for {provider}; set {spec.env_var} or run "
                f"`ig gen ... ideogram4 --mp {provider} --set-mk <key>`"
            )
        return HttpMagicPromptProvider(spec, model, api_key=key)
    raise click.ClickException(
        f"unknown magic-prompt provider {provider!r}; choose from: {', '.join(sorted(ALL_PROVIDERS))}"
    )
