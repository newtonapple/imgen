import importlib
import stat


def _fresh_config(tmp_path, monkeypatch):
    monkeypatch.setenv("IG_CONFIG_DIR", str(tmp_path))
    import imgen.config as c

    importlib.reload(c)
    return c


def test_secrets_set_get_roundtrip_and_mode_600(tmp_path, monkeypatch):
    c = _fresh_config(tmp_path, monkeypatch)
    s = c.Secrets.load()
    s.set_api_key("openrouter", "sk-test")
    s.save()
    path = tmp_path / "secrets.toml"
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert c.Secrets.load().api_key("openrouter") == "sk-test"


def test_secrets_missing_provider_returns_none(tmp_path, monkeypatch):
    c = _fresh_config(tmp_path, monkeypatch)
    assert c.Secrets.load().api_key("openai") is None
