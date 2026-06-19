from pathlib import Path

import pytest

from imagegen import config as cfg_mod
from imagegen.config import Config, resolve_weights_path


def _cfg(tmp_path):
    return Config.load(tmp_path / "config.toml")


def test_set_and_read_model_path_roundtrip(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.model_path("ideogram4") is None
    cfg.set_model_path("ideogram4", "/weights/ig4")
    cfg.save()
    reloaded = Config.load(tmp_path / "config.toml")
    assert reloaded.model_path("ideogram4") == Path("/weights/ig4")


def test_resolve_prefers_override(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.set_model_path("ideogram4", "/from/config")
    p = resolve_weights_path("ideogram4", override="/from/flag", cfg=cfg)
    assert p == Path("/from/flag")


def test_resolve_uses_config_then_env(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.set_model_path("ideogram4", "/from/config")
    assert resolve_weights_path("ideogram4", override=None, cfg=cfg) == Path("/from/config")
    empty = _cfg(tmp_path / "x")
    monkeypatch.setenv("IMAGEGEN_WEIGHTS_ROOT", "/root")
    assert resolve_weights_path("ideogram4", override=None, cfg=empty) == Path("/root/ideogram4")


def test_resolve_errors_when_nothing_set(tmp_path, monkeypatch):
    monkeypatch.delenv("IMAGEGEN_WEIGHTS_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="ig model set-path"):
        resolve_weights_path("ideogram4", override=None, cfg=_cfg(tmp_path))


def test_config_dir_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("IG_CONFIG_DIR", str(tmp_path / "cfgdir"))
    import importlib

    importlib.reload(cfg_mod)
    assert cfg_mod.CONFIG_PATH == tmp_path / "cfgdir" / "config.toml"
    monkeypatch.delenv("IG_CONFIG_DIR")
    importlib.reload(cfg_mod)
