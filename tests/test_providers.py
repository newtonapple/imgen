import pytest

from imagegen.config import Config, Secrets
from imagegen.magic_prompt import providers as P
from imagegen.magic_prompt.cli_provider import CliMagicPromptProvider
from imagegen.magic_prompt.http_provider import HttpMagicPromptProvider


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


def test_resolve_defaults_to_codex_when_nothing_set():
    provider, model = P.resolve_magic_settings({}, config=Config({}), secrets=Secrets({}))
    assert provider == "codex" and model == "gpt-5.5"


def test_resolve_transient_flags_win_without_persisting(tmp_path, monkeypatch):
    monkeypatch.setenv("IG_CONFIG_DIR", str(tmp_path))
    cfg = Config.load()
    provider, model = P.resolve_magic_settings(
        {"magic_prompt_provider": "openrouter", "magic_model": "openrouter/free"},
        config=cfg,
        secrets=Secrets.load(),
    )
    assert (provider, model) == ("openrouter", "openrouter/free")
    assert not (tmp_path / "config.toml").exists()  # transient, not persisted


def test_resolve_set_flags_persist_and_apply(tmp_path, monkeypatch):
    monkeypatch.setenv("IG_CONFIG_DIR", str(tmp_path))
    cfg = Config.load()
    secrets = Secrets.load()
    provider, model = P.resolve_magic_settings(
        {
            "set_magic_prompt_provider": "openrouter",
            "set_magic_model": "openrouter/free",
            "set_magic_prompt_api_key": "sk-persist",
        },
        config=cfg,
        secrets=secrets,
    )
    assert (provider, model) == ("openrouter", "openrouter/free")
    assert Config.load().magic_prompt_provider() == "openrouter"
    assert Config.load().magic_prompt_model() == "openrouter/free"
    assert Secrets.load().api_key("openrouter") == "sk-persist"


def test_resolve_pi_without_model_errors():
    import click

    with pytest.raises(click.ClickException, match="pi requires"):
        P.resolve_magic_settings(
            {"magic_prompt_provider": "pi"}, config=Config({}), secrets=Secrets({})
        )
