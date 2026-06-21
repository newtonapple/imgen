import pytest

from imgen.config import Config, Secrets
from imgen.magic_prompt import providers as P
from imgen.magic_prompt.cli_provider import CliMagicPromptProvider
from imgen.magic_prompt.http_provider import HttpMagicPromptProvider
from imgen.magic_prompt.ideogram_provider import IdeogramMagicPromptProvider


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


def test_make_provider_ideogram_resolves_key_from_secrets(monkeypatch):
    monkeypatch.delenv("IDEOGRAM_API_KEY", raising=False)
    secrets = Secrets({"ideogram": {"api_key": "ig-from-secrets"}})
    p = P.make_magic_provider("ideogram", "v4", secrets=secrets)
    assert isinstance(p, IdeogramMagicPromptProvider)
    assert p.api_key == "ig-from-secrets"


def test_make_provider_ideogram_env_beats_secrets(monkeypatch):
    monkeypatch.setenv("IDEOGRAM_API_KEY", "ig-env")
    p = P.make_magic_provider(
        "ideogram", "v4", secrets=Secrets({"ideogram": {"api_key": "ig-sec"}})
    )
    assert isinstance(p, IdeogramMagicPromptProvider) and p.api_key == "ig-env"


def test_make_provider_ideogram_missing_key_errors(monkeypatch):
    monkeypatch.delenv("IDEOGRAM_API_KEY", raising=False)
    import click

    with pytest.raises(click.ClickException, match="no API key for ideogram"):
        P.make_magic_provider("ideogram", "v4", secrets=Secrets({}))


def test_resolve_magic_provider_reads_config_then_defaults():
    assert P.resolve_magic_provider(Config({})) == ("codex", "gpt-5.5")
    cfg = Config({"magic_prompt": {"provider": "openrouter", "model": "openrouter/free"}})
    assert P.resolve_magic_provider(cfg) == ("openrouter", "openrouter/free")
    # provider set but model missing -> per-provider default
    assert P.resolve_magic_provider(Config({"magic_prompt": {"provider": "openai"}})) == (
        "openai",
        "gpt-4o-mini",
    )


def test_resolve_magic_provider_env_wins(monkeypatch):
    from imgen.config import Config
    from imgen.magic_prompt import providers as P

    cfg = Config({"magic_prompt": {"provider": "codex", "model": "gpt-5.5"}})
    monkeypatch.setenv("IG_MAGIC_PROVIDER", "openrouter")
    monkeypatch.setenv("IG_MAGIC_MODEL", "openrouter/free")
    assert P.resolve_magic_provider(cfg) == ("openrouter", "openrouter/free")


def test_resolve_magic_provider_env_provider_only_uses_provider_default(monkeypatch):
    from imgen.config import Config
    from imgen.magic_prompt import providers as P

    cfg = Config({"magic_prompt": {"provider": "codex", "model": "gpt-5.5"}})
    monkeypatch.setenv("IG_MAGIC_PROVIDER", "anthropic")  # no IG_MAGIC_MODEL
    provider, model = P.resolve_magic_provider(cfg)
    assert provider == "anthropic" and model == P.DEFAULT_MODELS["anthropic"]


def test_effective_magic_per_request():
    from imgen.magic_prompt.providers import effective_magic

    assert effective_magic("codex", "gpt-5.5", provider=None, model="gpt-5.4") == (
        "codex",
        "gpt-5.4",
    )
    assert effective_magic("codex", "gpt-5.5", provider=None, model=None) == ("codex", "gpt-5.5")
    p, m = effective_magic("codex", "gpt-5.5", provider="openrouter", model=None)
    assert p == "openrouter" and m == "openrouter/free"
