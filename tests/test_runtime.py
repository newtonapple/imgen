import importlib
import shutil
import tempfile

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
    # A short, unique /tmp dir keeps the socket path under the sun_path limit on all
    # platforms (pytest's tmp_path on macOS is /var/folders/... which is already too long).
    short_dir = tempfile.mkdtemp(dir="/tmp", prefix="igrt")
    monkeypatch.setenv("IG_RUNTIME_DIR", short_dir)
    import imagegen.config as c

    importlib.reload(c)
    try:
        c.validate_socket_path(c.daemon_socket_path("ideogram4"))  # short path, no raise
    finally:
        shutil.rmtree(short_dir, ignore_errors=True)
        importlib.reload(c)


def test_validate_socket_path_too_long(tmp_path, monkeypatch):
    monkeypatch.setenv("IG_RUNTIME_DIR", str(tmp_path / ("x" * 200)))
    import imagegen.config as c

    importlib.reload(c)
    with pytest.raises(RuntimeError, match="socket path too long"):
        c.validate_socket_path(c.daemon_socket_path("ideogram4"))
    importlib.reload(c)
