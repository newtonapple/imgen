import importlib
from pathlib import Path

import pytest


def _fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("IG_RUNTIME_DIR", str(tmp_path))
    import imagegen.config as c

    importlib.reload(c)
    return c


def test_runtime_paths_honor_env(tmp_path, monkeypatch):
    c = _fresh(tmp_path, monkeypatch)
    assert c.runtime_dir() == tmp_path
    assert c.daemon_socket_path("ideogram4") == tmp_path / "daemons" / "ideogram4.sock"
    assert c.daemon_record_path("ideogram4") == tmp_path / "daemons" / "ideogram4.json"
    assert c.daemon_log_path("ideogram4") == tmp_path / "logs" / "ideogram4.log"
    importlib.reload(c)


def test_validate_socket_path_ok(monkeypatch):
    # Use a short, explicit path so the socket path stays under the sun_path limit on all platforms.
    short_dir = Path("/tmp/ig_test")
    monkeypatch.setenv("IG_RUNTIME_DIR", str(short_dir))
    import imagegen.config as c

    importlib.reload(c)
    c.validate_socket_path(c.daemon_socket_path("ideogram4"))  # short path, no raise
    importlib.reload(c)


def test_validate_socket_path_too_long(tmp_path, monkeypatch):
    monkeypatch.setenv("IG_RUNTIME_DIR", str(tmp_path / ("x" * 200)))
    import imagegen.config as c

    importlib.reload(c)
    with pytest.raises(RuntimeError, match="socket path too long"):
        c.validate_socket_path(c.daemon_socket_path("ideogram4"))
    importlib.reload(c)
