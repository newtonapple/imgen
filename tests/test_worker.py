# tests/test_worker.py
from __future__ import annotations

import os
import tempfile
import threading
import time

from imagegen.engine.base import GenerationResult
from imagegen.worker import handle_request, send_request, serve


class FakeImg:
    size = (512, 512)

    def save(self, path):
        open(path, "wb").close()


class FakePipeline:
    def run(self, prompt, *, width, height, preset, seed, target_elements=0):
        return GenerationResult(
            image=FakeImg(),  # type: ignore[arg-type]
            seed=seed or 1,
            width=width,
            height=height,
            preset=preset,
            caption={"high_level_description": prompt},
            backend="fake",
            duration_s=0.1,
        )


def test_handle_run_writes_image_and_returns_params(tmp_path):
    out = tmp_path / "o.png"
    resp = handle_request(
        FakePipeline(),
        {
            "op": "run",
            "prompt": "a cat",
            "width": 512,
            "height": 512,
            "preset": "V4_DEFAULT_20",
            "seed": 9,
            "output_path": str(out),
        },
    )
    assert resp["ok"] and resp["seed"] == 9 and out.exists()


def test_handle_unknown_op_returns_error():
    resp = handle_request(FakePipeline(), {"op": "nope"})
    assert resp["ok"] is False and "op" in resp["error"].lower()


def test_socket_roundtrip(tmp_path):
    # Use a short path under /tmp to stay within AF_UNIX's 104-char limit on macOS.
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        sock = os.path.join(td, "w.sock")
        out = str(tmp_path / "o.png")
        t = threading.Thread(target=serve, args=(sock, FakePipeline()), daemon=True)
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
                "preset": "V4_TURBO_12",
                "seed": 3,
                "output_path": out,
            },
        )
        assert resp["ok"] and resp["seed"] == 3
