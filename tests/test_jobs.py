import importlib


def _fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("IG_RUNTIME_DIR", str(tmp_path))
    import imagegen.config as c
    import imagegen.jobs as j

    importlib.reload(c)
    importlib.reload(j)
    return j


def test_new_job_id_is_6_hex(tmp_path, monkeypatch):
    j = _fresh(tmp_path, monkeypatch)
    jid = j.new_job_id()
    assert len(jid) == 6 and all(ch in "0123456789abcdef" for ch in jid)


def test_create_read_job(tmp_path, monkeypatch):
    j = _fresh(tmp_path, monkeypatch)
    rec = j.create_job("ab12cd", model="ideogram4", out="/tmp/o.png", request={"op": "run"})
    assert rec["status"] == "queued" and rec["request"]["op"] == "run"
    again = j.read_job("ab12cd")
    assert again is not None and again["model"] == "ideogram4" and again["out"] == "/tmp/o.png"
    assert again["finished_at"] is None and again["error"] is None


def test_set_job_status_merges(tmp_path, monkeypatch):
    j = _fresh(tmp_path, monkeypatch)
    j.create_job("ab12cd", model="ideogram4", out="/tmp/o.png", request={})
    j.set_job_status("ab12cd", "running", pid=4321)
    rec = j.read_job("ab12cd")
    assert rec["status"] == "running" and rec["pid"] == 4321
    j.set_job_status("ab12cd", "failed", error="OOM", finished_at=1.0)
    rec = j.read_job("ab12cd")
    assert rec["status"] == "failed" and rec["error"] == "OOM" and rec["finished_at"] == 1.0


def test_list_jobs_newest_first(tmp_path, monkeypatch):
    j = _fresh(tmp_path, monkeypatch)
    j.create_job("aaaaaa", model="m", out="/a", request={})
    j.set_job_status("aaaaaa", "queued", started_at=1.0)
    j.create_job("bbbbbb", model="m", out="/b", request={})
    j.set_job_status("bbbbbb", "queued", started_at=2.0)
    ids = [r["id"] for r in j.list_jobs()]
    assert ids[0] == "bbbbbb" and ids[1] == "aaaaaa"


def test_spawn_runner_detaches(tmp_path, monkeypatch):
    j = _fresh(tmp_path, monkeypatch)
    calls = {}

    class FakePopen:
        def __init__(self, argv, **kw):
            calls["argv"] = argv
            calls["kw"] = kw
            self.pid = 5555

    monkeypatch.setattr(j.subprocess, "Popen", FakePopen)
    j.create_job("ab12cd", model="ideogram4", out="/tmp/o.png", request={})
    pid = j.spawn_runner("ab12cd")
    assert pid == 5555
    assert calls["argv"][1:] == ["-m", "imagegen.jobs", "ab12cd"]
    assert calls["kw"]["start_new_session"] is True


def test_run_job_success_writes_sidecar_and_done(tmp_path, monkeypatch):
    j = _fresh(tmp_path, monkeypatch)
    out = tmp_path / "o.png"
    j.create_job(
        "ab12cd",
        model="ideogram4",
        out=str(out),
        request={
            "op": "run",
            "prompt": "a cat",
            "width": 64,
            "height": 64,
            "output_path": str(out),
        },
    )
    monkeypatch.setattr(j.daemon, "ensure_daemon", lambda name: "/tmp/fake.sock")
    monkeypatch.setattr(
        j,
        "stream_request",
        lambda sock, req, on_progress=None: {
            "ok": True,
            "seed": 1,
            "width": 64,
            "height": 64,
            "preset": "V4_TURBO_12",
            "backend": "mlx",
            "duration_s": 0.1,
            "caption": {"high_level_description": "a cat"},
        },
    )
    rc = j.run_job("ab12cd")
    assert rc == 0
    rec = j.read_job("ab12cd")
    assert rec["status"] == "done" and rec["finished_at"] is not None
    meta = __import__("json").loads((tmp_path / "o.png.json").read_text())
    assert meta["caption"]["high_level_description"] == "a cat"


def test_run_job_failure_records_error(tmp_path, monkeypatch):
    j = _fresh(tmp_path, monkeypatch)
    out = tmp_path / "o.png"
    j.create_job(
        "ab12cd", model="ideogram4", out=str(out), request={"op": "run", "output_path": str(out)}
    )
    monkeypatch.setattr(j.daemon, "ensure_daemon", lambda name: "/tmp/fake.sock")
    monkeypatch.setattr(
        j, "stream_request", lambda sock, req, on_progress=None: {"ok": False, "error": "OOM"}
    )
    rc = j.run_job("ab12cd")
    assert rc == 1
    rec = j.read_job("ab12cd")
    assert rec["status"] == "failed" and rec["error"] == "OOM"


def test_clean_removes_finished_keeps_active(tmp_path, monkeypatch):
    j = _fresh(tmp_path, monkeypatch)
    monkeypatch.setattr(j.daemon, "list_daemons", lambda: [])
    j.create_job("done01", model="m", out="/o.png", request={})
    j.set_job_status("done01", "done", finished_at=1.0)
    j.config.job_log_path("done01").write_text("log")
    j.create_job("run001", model="m", out="/o.png", request={})
    j.set_job_status("run001", "running")
    j.create_job("queue1", model="m", out="/o.png", request={})  # queued
    stats = j.clean()
    assert stats["jobs"] == 1
    assert j.read_job("done01") is None
    assert not j.config.job_log_path("done01").exists()
    assert j.read_job("run001") is not None and j.read_job("queue1") is not None


def test_clean_older_than(tmp_path, monkeypatch):
    import time as _t

    j = _fresh(tmp_path, monkeypatch)
    monkeypatch.setattr(j.daemon, "list_daemons", lambda: [])
    j.create_job("oldjob", model="m", out="/o", request={})
    j.set_job_status("oldjob", "done", finished_at=_t.time() - 10 * 86400)
    j.create_job("newjob", model="m", out="/o", request={})
    j.set_job_status("newjob", "done", finished_at=_t.time())
    stats = j.clean(older_than_days=7)
    assert stats["jobs"] == 1
    assert j.read_job("oldjob") is None and j.read_job("newjob") is not None


def test_clean_dead_daemon_logs(tmp_path, monkeypatch):
    j = _fresh(tmp_path, monkeypatch)
    monkeypatch.setattr(
        j.daemon,
        "list_daemons",
        lambda: [{"model": "live1", "live": True}, {"model": "dead1", "live": False}],
    )
    j.config.logs_dir().mkdir(parents=True, exist_ok=True)
    (j.config.logs_dir() / "live1.log").write_text("x")
    (j.config.logs_dir() / "dead1.log").write_text("x")
    stats = j.clean()
    assert stats["logs"] == 1
    assert (j.config.logs_dir() / "live1.log").exists()
    assert not (j.config.logs_dir() / "dead1.log").exists()
    # --all truncates the live one
    stats2 = j.clean(truncate_running=True)
    assert stats2["truncated"] == 1
    assert (j.config.logs_dir() / "live1.log").read_text() == ""
