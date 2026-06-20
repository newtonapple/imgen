import pytest

from imgen.config import Config, Secrets
from imgen.magic_prompt import providers as P
from imgen.magic_prompt.cli_provider import CliMagicPromptProvider
from imgen.magic_prompt.http_provider import HttpMagicPromptProvider


def test_make_provider_cli_no_key():
    p = P.make_magic_provider("codex", "gpt-5.5", secrets=Secrets({}))
    assert isinstance(p, CliMagicPromptProvider)
    assert p.provider == "codex" and p.model == "gpt-5.5"


def test_make_provider_http_resolves_key_from_secrets(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    secrets = Secrets({"openrouter": {"api_key": "sk-from-secrets"}})
    p = P.make_magic_provider("openrouter", "openrouter/free", secrets=secrets)
    assert isinstance(p, HttpMagicPromptProvider)
    assert p.api_key == "sk-from-secrets"


def test_make_provider_http_env_beats_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    p = P.make_magic_provider(
        "openai", "gpt-4o-mini", secrets=Secrets({"openai": {"api_key": "sk-sec"}})
    )
    assert isinstance(p, HttpMagicPromptProvider) and p.api_key == "sk-env"


def test_make_provider_http_missing_key_errors(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import click

    with pytest.raises(click.ClickException, match="no API key"):
        P.make_magic_provider("anthropic", "claude-haiku-4-5", secrets=Secrets({}))


def test_make_provider_unknown_errors():
    import click

    with pytest.raises(click.ClickException, match="unknown magic-prompt provider"):
        P.make_magic_provider("nope", "x", secrets=Secrets({}))


def test_resolve_magic_provider_reads_config_then_defaults():
    assert P.resolve_magic_provider(Config({})) == ("codex", "gpt-5.5")
    cfg = Config({"magic_prompt": {"provider": "openrouter", "model": "openrouter/free"}})
    assert P.resolve_magic_provider(cfg) == ("openrouter", "openrouter/free")
    # provider set but model missing -> per-provider default
    assert P.resolve_magic_provider(Config({"magic_prompt": {"provider": "openai"}})) == (
        "openai",
        "gpt-4o-mini",
    )
