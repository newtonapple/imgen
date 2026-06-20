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
