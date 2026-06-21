"""Magic-prompt provider factory + CLI-flag resolution."""

from __future__ import annotations

import os

import click

from ..config import Config, MAGIC_MODEL_ENV, MAGIC_PROVIDER_ENV, Secrets
from .base import MagicPromptProvider
from .cli_provider import CliMagicPromptProvider
from .http_provider import PROVIDERS, HttpMagicPromptProvider
from .ideogram_provider import IDEOGRAM_API_KEY_ENV, IdeogramMagicPromptProvider

CLI_PROVIDERS = {"codex", "claude", "pi"}
HTTP_PROVIDERS = set(PROVIDERS) | {"ideogram"}  # all providers that need an API key
ALL_PROVIDERS = CLI_PROVIDERS | HTTP_PROVIDERS

DEFAULT_MODELS = {
    "codex": "gpt-5.5",
    "claude": "sonnet",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "openrouter": "openrouter/free",
    "ideogram": "v4",  # ignored — the hosted magic-prompt endpoint has no model selector
}  # pi has no default — it needs an explicit "<pi-provider>/<model>".


def resolve_magic_provider(config: Config) -> tuple[str, str]:
    """Daemon-default (provider, model): IG_MAGIC_* > config > codex/<provider default>."""
    cfg_provider = config.magic_prompt_provider()
    provider = os.environ.get(MAGIC_PROVIDER_ENV) or cfg_provider or "codex"
    model = os.environ.get(MAGIC_MODEL_ENV)
    if not model:
        model = config.magic_prompt_model() if cfg_provider == provider else None
    if not model:
        model = DEFAULT_MODELS.get(provider)
    if model is None:
        raise click.ClickException(f"no magic-prompt model for {provider!r}; set one with config")
    return provider, model


def effective_magic(
    default_provider: str, default_model: str, *, provider: str | None, model: str | None
) -> tuple[str, str]:
    """Per-request (provider, model) given the daemon defaults and request overrides."""
    p = provider or default_provider
    if model:
        m: str | None = model
    elif p == default_provider:
        m = default_model
    else:
        m = DEFAULT_MODELS.get(p)
    if m is None:
        raise click.ClickException(f"no magic-prompt model for {p!r}; pass --magic-model")
    return p, m


def make_magic_provider(provider: str, model: str, *, secrets: Secrets) -> MagicPromptProvider:
    if provider in CLI_PROVIDERS:
        return CliMagicPromptProvider(provider, model)
    if provider == "ideogram":
        key = os.environ.get(IDEOGRAM_API_KEY_ENV) or secrets.api_key("ideogram")
        if not key:
            raise click.ClickException(
                f"no API key for ideogram; set {IDEOGRAM_API_KEY_ENV} or run "
                f"`ig ideogram4 config set-key ideogram <key>`"
            )
        return IdeogramMagicPromptProvider(model, api_key=key)
    if provider in PROVIDERS:
        spec = PROVIDERS[provider]
        key = os.environ.get(spec.env_var) or secrets.api_key(provider)
        if not key:
            raise click.ClickException(
                f"no API key for {provider}; set {spec.env_var} or run "
                f"`ig ideogram4 config set-key {provider} <key>`"
            )
        return HttpMagicPromptProvider(spec, model, api_key=key)
    raise click.ClickException(
        f"unknown magic-prompt provider {provider!r}; choose from: {', '.join(sorted(ALL_PROVIDERS))}"
    )
