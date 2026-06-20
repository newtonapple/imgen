"""Magic-prompt provider factory + CLI-flag resolution."""

from __future__ import annotations

import os
from typing import Any

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


def resolve_magic_settings(
    opts: dict[str, Any], *, config: Config, secrets: Secrets
) -> tuple[str, str]:
    """Read --mp/--mm/--set-* from opts, persist --set-*, return (provider, model)."""
    set_provider = opts.get("set_magic_prompt_provider")
    set_model = opts.get("set_magic_model")
    set_key = opts.get("set_magic_prompt_api_key")

    if set_provider:
        config.set_magic_prompt_provider(set_provider)
        config.save()
    if set_model:
        config.set_magic_prompt_model(set_model)
        config.save()

    provider = (
        set_provider
        or opts.get("magic_prompt_provider")
        or config.magic_prompt_provider()
        or "codex"
    )

    model = set_model or opts.get("magic_model")
    if model is None:
        cfg_model = config.magic_prompt_model()
        if cfg_model and config.magic_prompt_provider() == provider:
            model = cfg_model
        else:
            model = DEFAULT_MODELS.get(provider)

    if set_key:
        secrets.set_api_key(provider, set_key)
        secrets.save()

    if model is None:
        if provider == "pi":
            raise click.ClickException('pi requires --mm "<pi-provider>/<model>"')
        raise click.ClickException(f"no model for magic-prompt provider {provider!r}")

    return provider, model
