"""Background-generation jobs: record store + (Task 3) detached runner."""

from __future__ import annotations

import json
import secrets as _secrets
import time
from pathlib import Path
from typing import Any

from . import config


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
