"""Background-generation jobs: record store + (Task 3) detached runner."""

from __future__ import annotations

import json
import os
import secrets as _secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import config, daemon
from .metadata import build_summary, write_sidecar
from .worker import stream_request


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def new_job_id() -> str:
    return _secrets.token_hex(3)  # 6 lowercase hex chars


def create_job(job_id: str, *, model: str, out: str, request: dict[str, Any]) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "id": job_id,
        "model": model,
        "out": out,
        "status": "queued",
        "started_at": time.time(),
        "finished_at": None,
        "pid": None,
        "log": str(config.job_log_path(job_id)),
        "request": request,
        "error": None,
    }
    _atomic_write(config.job_record_path(job_id), rec)
    return rec


def read_job(job_id: str) -> dict[str, Any] | None:
    path = config.job_record_path(job_id)
    if not path.exists():
        return None
    try:
        rec: dict[str, Any] = json.loads(path.read_text())
        return rec
    except (json.JSONDecodeError, OSError):
        return None


def list_jobs() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    d = config.jobs_dir()
    if not d.exists():
        return out
    for p in d.glob("*.json"):
        try:
            out.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    out.sort(key=lambda r: r.get("started_at", 0), reverse=True)
    return out


def set_job_status(job_id: str, status: str, **fields: Any) -> None:
    rec = read_job(job_id)
    if rec is None:
        return
    rec["status"] = status
    rec.update(fields)
    _atomic_write(config.job_record_path(job_id), rec)


def spawn_runner(job_id: str) -> int:
    logp = config.job_log_path(job_id)
    logp.parent.mkdir(parents=True, exist_ok=True)
    with open(logp, "a") as logf:
        proc = subprocess.Popen(
            [sys.executable, "-m", "imagegen.jobs", job_id],
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=logf,
            start_new_session=True,
        )
    set_job_status(job_id, "queued", pid=proc.pid)
    return proc.pid


def _log_progress(d: dict[str, Any]) -> None:
    print(json.dumps(d), flush=True)  # goes to the per-job log (stdout is redirected)


def run_job(job_id: str) -> int:
    rec = read_job(job_id)
    if rec is None:
        print(f"job {job_id} not found", file=sys.stderr)
        return 1
    model = str(rec["model"])
    out = str(rec["out"])
    req: dict[str, Any] = dict(rec["request"])
    set_job_status(job_id, "running", pid=os.getpid())
    try:
        sock = daemon.ensure_daemon(model)
        result = stream_request(sock, req, _log_progress)
    except Exception as exc:  # connection reset, spawn timeout, etc.
        set_job_status(
            job_id, "failed", error=f"{type(exc).__name__}: {exc}", finished_at=time.time()
        )
        return 1
    if not result.get("ok"):
        set_job_status(
            job_id,
            "failed",
            error=str(result.get("error", "generation failed")),
            finished_at=time.time(),
        )
        return 1
    summary = build_summary(out, result, model=model, prompt=req.get("prompt"))
    write_sidecar(out, summary)
    set_job_status(job_id, "done", finished_at=time.time())
    return 0


def clean(*, older_than_days: int | None = None, truncate_running: bool = False) -> dict[str, int]:
    """Remove finished-job records/logs and dead-daemon logs.

    Returns a dict with keys "jobs", "logs", "truncated".
    """
    stats: dict[str, int] = {"jobs": 0, "logs": 0, "truncated": 0}
    cutoff = time.time() - older_than_days * 86400 if older_than_days is not None else None
    for rec in list_jobs():
        if rec.get("status") not in ("done", "failed"):
            continue
        if cutoff is not None and (rec.get("finished_at") or 0.0) > cutoff:
            continue
        config.job_record_path(rec["id"]).unlink(missing_ok=True)
        config.job_log_path(rec["id"]).unlink(missing_ok=True)
        stats["jobs"] += 1

    live = {r["model"] for r in daemon.list_daemons() if r.get("live")}
    logs = config.logs_dir()
    if logs.exists():
        for logp in logs.glob("*.log"):
            if logp.stem in live:
                if truncate_running:
                    logp.write_text("")
                    stats["truncated"] += 1
            else:
                logp.unlink(missing_ok=True)
                stats["logs"] += 1
    return stats


def _main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m imagegen.jobs <job-id>", file=sys.stderr)
        return 2
    return run_job(argv[0])


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
