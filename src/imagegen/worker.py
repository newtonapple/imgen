"""Warm worker: one engine held in memory, one job at a time over a Unix socket (NDJSON)."""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Callable
from typing import Any


def handle_request(
    pipeline: Any, req: dict[str, Any], emit: Callable[[dict[str, Any]], None]
) -> dict[str, Any]:
    op = req.get("op")
    try:
        if op == "magic_prompt":
            cap = pipeline.magic(
                req["prompt"],
                width=req["width"],
                height=req["height"],
                target_elements=req.get("target_elements", 0),
            )
            return {"ok": True, "caption": cap}
        if op in ("generate", "run"):
            if op == "run":
                emit({"type": "progress", "phase": "magic-prompt"})
                caption = pipeline.magic(
                    req["prompt"],
                    width=req["width"],
                    height=req["height"],
                    target_elements=req.get("target_elements", 0),
                )
            else:
                caption = req["caption"]
            emit({"type": "progress", "phase": "sampling"})
            r = pipeline.generate(
                caption,
                width=req["width"],
                height=req["height"],
                preset=req.get("preset", "V4_DEFAULT_20"),
                seed=req.get("seed"),
            )
            emit({"type": "progress", "phase": "saving"})
            r.image.save(req["output_path"])
            return {
                "ok": True,
                "seed": r.seed,
                "width": r.width,
                "height": r.height,
                "preset": r.preset,
                "caption": caption,
                "backend": r.backend,
                "output_path": req["output_path"],
                "duration_s": r.duration_s,
            }
        return {"ok": False, "error": f"unknown op: {op!r}"}
    except Exception as exc:  # report, keep the worker alive
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def serve(
    socket_path: str,
    pipeline: Any,
    *,
    on_job_start: Callable[[], None] | None = None,
    on_job_end: Callable[[], None] | None = None,
) -> None:
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(socket_path)
    srv.listen(64)
    try:
        while True:
            conn, _ = srv.accept()
            with conn:
                data = conn.makefile("r").readline()
                if not data:
                    continue

                def emit(d: dict[str, Any]) -> None:
                    conn.sendall((json.dumps(d) + "\n").encode())

                if on_job_start is not None:
                    on_job_start()
                try:
                    result = handle_request(pipeline, json.loads(data), emit)
                    emit(result)
                finally:
                    if on_job_end is not None:
                        on_job_end()
    finally:
        srv.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)


def stream_request(
    socket_path: str,
    req: dict[str, Any],
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(socket_path)
    s.sendall((json.dumps(req) + "\n").encode())
    f = s.makefile("r")
    try:
        for line in f:
            d: dict[str, Any] = json.loads(line)
            if d.get("type") == "progress":
                if on_progress is not None:
                    on_progress(d)
                continue
            return d
        raise RuntimeError("worker closed the connection without a result")
    finally:
        s.close()


def send_request(socket_path: str, req: dict[str, Any]) -> dict[str, Any]:
    return stream_request(socket_path, req, None)
