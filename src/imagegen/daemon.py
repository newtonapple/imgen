"""Single warm daemon per model: registry, liveness, detached auto-start, stop."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import config
from .worker import serve as worker_serve

_START_TIMEOUT_S = 900  # cold model load can take minutes


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def write_record(model: str, *, socket: str, pid: int, backend: str, quantize: str | None) -> None:
    _atomic_write(
        config.daemon_record_path(model),
        {
            "model": model,
            "socket": socket,
            "pid": pid,
            "backend": backend,
            "quantize": quantize,
            "started_at": time.time(),
            "state": "idle",
            "log": str(config.daemon_log_path(model)),
        },
    )


def read_record(model: str) -> dict[str, Any] | None:
    path = config.daemon_record_path(model)
    if not path.exists():
        return None
    try:
        rec: dict[str, Any] = json.loads(path.read_text())
        return rec
    except (json.JSONDecodeError, OSError):
        return None


def remove_record(model: str) -> None:
    config.daemon_record_path(model).unlink(missing_ok=True)


def set_state(model: str, state: str) -> None:
    rec = read_record(model)
    if rec is not None:
        rec["state"] = state
        _atomic_write(config.daemon_record_path(model), rec)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _socket_connectable(path: str) -> bool:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(path)
        return True
    except OSError:
        return False
    finally:
        s.close()


def is_live(record: dict[str, Any]) -> bool:
    return _pid_alive(int(record["pid"])) and _socket_connectable(str(record["socket"]))


def live_record(model: str) -> dict[str, Any] | None:
    rec = read_record(model)
    if rec is None:
        return None
    if is_live(rec):
        return rec
    remove_record(model)  # stale (crashed / dead pid)
    return None


def list_daemons() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    d = config.daemons_dir()
    if not d.exists():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            rec: dict[str, Any] = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        rec["live"] = is_live(rec)
        out.append(rec)
    return out


def stop(model: str) -> bool:
    rec = read_record(model)
    if rec is None:
        return False
    try:
        os.kill(int(rec["pid"]), signal.SIGTERM)
    except ProcessLookupError:
        pass
    remove_record(model)
    return True


def run_daemon(
    model: str,
    build_pipeline: Callable[[], Any],
    *,
    backend: str,
    quantize: str | None,
) -> None:
    """Foreground daemon body. Builds the pipeline, binds the socket (after load),
    registers, serves until SIGTERM."""
    sock = config.daemon_socket_path(model)
    config.validate_socket_path(sock)
    pipeline = build_pipeline()  # heavy: load the model warm (minutes)

    def _shutdown(signum: int, frame: Any) -> None:
        remove_record(model)
        if sock.exists():
            sock.unlink()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    write_record(model, socket=str(sock), pid=os.getpid(), backend=backend, quantize=quantize)
    try:
        worker_serve(
            str(sock),
            pipeline,
            on_job_start=lambda: set_state(model, "busy"),
            on_job_end=lambda: set_state(model, "idle"),
        )  # binds the socket; serves one job at a time
    finally:
        remove_record(model)
        if sock.exists():
            sock.unlink()


def ensure_daemon(model: str) -> str:
    """Return a live socket path for `model`, spawning a detached `ig <model> serve` if needed."""
    rec = live_record(model)
    if rec is not None:
        return str(rec["socket"])

    lock_path = config.daemons_dir() / f"{model}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        rec = live_record(model)  # re-check under lock
        if rec is not None:
            return str(rec["socket"])
        _spawn_detached(model)
        deadline = time.time() + _START_TIMEOUT_S
        while time.time() < deadline:
            ready = live_record(model)
            if ready is not None:
                return str(ready["socket"])
            time.sleep(0.25)
        raise RuntimeError(
            f"daemon for {model!r} did not become ready in {_START_TIMEOUT_S}s; "
            f"see {config.daemon_log_path(model)}"
        )


def _spawn_detached(model: str) -> None:
    logp = config.daemon_log_path(model)
    logp.parent.mkdir(parents=True, exist_ok=True)
    # Popen duplicates the fd for the child, so closing our copy here is safe.
    with open(logp, "a") as logf:
        subprocess.Popen(
            [sys.executable, "-m", "imagegen.cli", model, "serve"],
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=logf,
            start_new_session=True,  # detach from the controlling terminal / process group
        )
