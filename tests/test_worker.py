# tests/test_worker.py
from __future__ import annotations


def _fake_pipeline():
    class FakeImg:
        def save(self, p):
            open(p, "wb").close()

    class FakeResult:
        def __init__(self, seed):
            self.image = FakeImg()
            self.seed = seed or 1
            self.width = 64
            self.height = 64
            self.preset = "V4_TURBO_12"
            self.caption = {"high_level_description": "x"}
            self.backend = "fake"
            self.duration_s = 0.1

    class FakePipeline:
        def magic(self, prompt, *, width, height, target_elements=0):
            return {"high_level_description": prompt}

        def generate(self, caption, *, width, height, preset, seed):
            return FakeResult(seed)

    return FakePipeline()


def test_handle_run_emits_progress_and_result(tmp_path):
    from imgen.worker import handle_request

    out = tmp_path / "o.png"
    events: list[dict[str, object]] = []
    res = handle_request(
        _fake_pipeline(),
        {
            "op": "run",
            "prompt": "a cat",
            "width": 64,
            "height": 64,
            "seed": 9,
            "preset": "V4_TURBO_12",
            "output_path": str(out),
        },
        events.append,
    )
    phases = [e["phase"] for e in events if e.get("type") == "progress"]
    assert "magic-prompt" in phases and "sampling" in phases and "saving" in phases
    assert res["ok"] and res["seed"] == 9 and res["caption"]["high_level_description"] == "a cat"
    assert out.exists()


def test_handle_unknown_op_returns_error():
    from imgen.worker import handle_request

    resp = handle_request(_fake_pipeline(), {"op": "nope"}, lambda _d: None)
    assert resp["ok"] is False and "op" in resp["error"].lower()


def test_socket_roundtrip(tmp_path):
    import os
    import tempfile
    import threading
    import time
    from imgen.worker import send_request, serve

    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        sock = os.path.join(td, "w.sock")
        out = str(tmp_path / "o.png")
        t = threading.Thread(target=serve, args=(sock, _fake_pipeline()), daemon=True)
        t.start()
        for _ in range(50):
            if os.path.exists(sock):
                break
            time.sleep(0.02)
        resp = send_request(
            sock,
            {
                "op": "run",
                "prompt": "x",
                "width": 64,
                "height": 64,
                "seed": 3,
                "preset": "V4_TURBO_12",
                "output_path": out,
            },
        )
        assert resp["ok"] and resp["seed"] == 3


def test_serve_invokes_job_hooks(tmp_path):
    import os
    import tempfile
    import threading
    import time
    from imgen.worker import send_request, serve

    events: list[str] = []
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        sock = os.path.join(td, "w.sock")
        out = str(tmp_path / "o.png")
        t = threading.Thread(
            target=serve,
            args=(sock, _fake_pipeline()),
            kwargs={
                "on_job_start": lambda: events.append("start"),
                "on_job_end": lambda: events.append("end"),
            },
            daemon=True,
        )
        t.start()
        for _ in range(50):
            if os.path.exists(sock):
                break
            time.sleep(0.02)
        resp = send_request(
            sock,
            {
                "op": "run",
                "prompt": "x",
                "width": 64,
                "height": 64,
                "seed": 3,
                "preset": "V4_TURBO_12",
                "output_path": out,
            },
        )
        assert resp["ok"]
        for _ in range(50):
            if events == ["start", "end"]:
                break
            time.sleep(0.02)
    assert events == ["start", "end"]
