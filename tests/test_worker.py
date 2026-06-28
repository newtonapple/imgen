# tests/test_worker.py
from __future__ import annotations


def _fake_pipeline():
    class FakeImg:
        def save(self, p):
            open(p, "wb").close()

    class FakeResult:
        def __init__(self, seed):
            self.image = FakeImg(); self.seed = seed or 1
            self.width = 64; self.height = 64; self.preset = "V4_TURBO_12"
            self.caption = {"high_level_description": "x"}
            self.backend = "fake"; self.duration_s = 0.1

    class FakePipeline:
        def magic(self, prompt, *, width, height, target_elements=0,
                  magic_provider=None, magic_model=None):
            return {"high_level_description": prompt}

        def generate(self, caption, *, width, height, preset, seed, progress=None):
            if progress is not None:
                for k in range(1, 3):  # 2 fake steps
                    progress(k, 2)
            return FakeResult(seed)

    return FakePipeline()


def test_run_emits_caption_then_per_step_then_saving():
    from imgen.worker import handle_request

    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "o.png")
        events = []
        res = handle_request(
            _fake_pipeline(),
            {"op": "run", "prompt": "a cat", "width": 64, "height": 64,
             "seed": 9, "preset": "V4_TURBO_12", "output_path": out},
            events.append,
        )
    seq = [(e.get("phase"), e.get("step"), e.get("total")) for e in events if e.get("type") == "progress"]
    assert seq == [
        ("magic-prompt", None, None),
        ("caption", None, None),
        ("sampling", 1, 2),
        ("sampling", 2, 2),
        ("saving", None, None),
    ]
    caption_evt = next(e for e in events if e.get("phase") == "caption")
    assert caption_evt["caption"] == {"high_level_description": "a cat"}
    assert res["ok"] and res["seed"] == 9


def test_generate_op_has_no_magic_or_caption_events():
    from imgen.worker import handle_request

    import tempfile, os
    cap = {"high_level_description": "x"}
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "o.png")
        events = []
        handle_request(
            _fake_pipeline(),
            {"op": "generate", "caption": cap, "width": 64, "height": 64,
             "seed": 1, "preset": "V4_TURBO_12", "output_path": out},
            events.append,
        )
    phases = [e.get("phase") for e in events if e.get("type") == "progress"]
    assert "magic-prompt" not in phases and "caption" not in phases
    assert phases == ["sampling", "sampling", "saving"]


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


def test_handle_run_passes_magic_overrides(tmp_path):
    from imgen.worker import handle_request

    seen = {}

    class FakeImg:
        def save(self, p):
            open(p, "wb").close()

    class FakeResult:
        image = FakeImg()
        seed = 1
        width = 64
        height = 64
        preset = "V4_TURBO_12"
        caption = {}
        backend = "fake"
        duration_s = 0.1

    class FakePipeline:
        def magic(
            self, prompt, *, width, height, target_elements=0, magic_provider=None, magic_model=None
        ):
            seen["provider"] = magic_provider
            seen["model"] = magic_model
            return {"high_level_description": prompt}

        def generate(self, caption, *, width, height, preset, seed):
            return FakeResult()

    handle_request(
        FakePipeline(),
        {
            "op": "run",
            "prompt": "x",
            "width": 64,
            "height": 64,
            "output_path": str(tmp_path / "o.png"),
            "magic_provider": "openrouter",
            "magic_model": "m",
        },
        lambda d: None,
    )
    assert seen == {"provider": "openrouter", "model": "m"}
