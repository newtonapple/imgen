"""Warm worker: one engine held in memory, one job at a time over a Unix socket."""
from __future__ import annotations

import json
import os
import socket


def handle_request(pipeline, req: dict) -> dict:
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
                r = pipeline.run(
                    req["prompt"],
                    width=req["width"],
                    height=req["height"],
                    preset=req.get("preset", "V4_DEFAULT_20"),
                    seed=req.get("seed"),
                    target_elements=req.get("target_elements", 0),
                )
            else:
                r = pipeline.generate(
                    req["caption"],
                    width=req["width"],
                    height=req["height"],
                    preset=req.get("preset", "V4_DEFAULT_20"),
                    seed=req.get("seed"),
                )
            r.image.save(req["output_path"])
            return {
                "ok": True,
                "seed": r.seed,
                "width": r.width,
                "height": r.height,
                "preset": r.preset,
                "caption": r.caption,
                "output_path": req["output_path"],
                "duration_s": r.duration_s,
            }
        return {"ok": False, "error": f"unknown op: {op!r}"}
    except Exception as exc:  # report, keep the worker alive
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def serve(socket_path: str, pipeline) -> None:
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(socket_path)
    srv.listen(1)
    try:
        while True:
            conn, _ = srv.accept()
            with conn:
                data = conn.makefile("r").readline()
                if not data:
                    continue
                resp = handle_request(pipeline, json.loads(data))
                conn.sendall((json.dumps(resp) + "\n").encode())
    finally:
        srv.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)


def send_request(socket_path: str, req: dict) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(socket_path)
    s.sendall((json.dumps(req) + "\n").encode())
    line = s.makefile("r").readline()
    s.close()
    return json.loads(line)
