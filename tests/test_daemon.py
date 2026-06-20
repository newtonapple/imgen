import importlib
import os


def _fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("IG_RUNTIME_DIR", str(tmp_path))
    import imagegen.config as c
    import imagegen.daemon as d

    importlib.reload(c)
    importlib.reload(d)
    return d


def test_record_roundtrip_and_remove(tmp_path, monkeypatch):
    d = _fresh(tmp_path, monkeypatch)
    d.write_record("ideogram4", socket="/tmp/x.sock", pid=os.getpid(), backend="mlx", quantize=None)
    rec = d.read_record("ideogram4")
    assert rec["pid"] == os.getpid() and rec["state"] == "idle"
    d.remove_record("ideogram4")
    assert d.read_record("ideogram4") is None


def test_live_record_prunes_dead_pid(tmp_path, monkeypatch):
    d = _fresh(tmp_path, monkeypatch)
    # pid 999999 is (almost certainly) not running
    d.write_record("ideogram4", socket="/tmp/x.sock", pid=999999, backend="mlx", quantize=None)
    assert d.live_record("ideogram4") is None
    assert d.read_record("ideogram4") is None  # pruned


def test_set_state(tmp_path, monkeypatch):
    d = _fresh(tmp_path, monkeypatch)
    d.write_record("ideogram4", socket="/tmp/x.sock", pid=os.getpid(), backend="mlx", quantize=None)
    d.set_state("ideogram4", "busy")
    assert d.read_record("ideogram4")["state"] == "busy"


def test_stop_sigterms_pid(tmp_path, monkeypatch):
    d = _fresh(tmp_path, monkeypatch)
    sent: dict[str, int] = {}
    monkeypatch.setattr(d.os, "kill", lambda pid, sig: sent.update(pid=pid, sig=sig))
    d.write_record("ideogram4", socket="/tmp/x.sock", pid=4123, backend="mlx", quantize=None)
    assert d.stop("ideogram4") is True
    assert sent["pid"] == 4123
    assert d.stop("ideogram4") is False  # no record now? (record removed by stop) -> see impl
