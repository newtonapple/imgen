"""`ig model` subgroup: list / show / stop-all."""

from __future__ import annotations

import datetime as _dt

import click

from .. import daemon, models
from ..config import Config


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "-"
    return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


@click.group("model")
def model_group() -> None:
    """Inspect and configure models."""


@model_group.command("list")
def list_cmd() -> None:
    by_model = {rec["model"]: rec for rec in daemon.list_daemons()}
    for m in sorted(models.all_models(), key=lambda x: x.name):
        rec = by_model.get(m.name)
        if rec is not None and rec.get("live"):
            status = f"running (pid {rec['pid']}, {rec.get('state', 'idle')})"
        elif rec is not None and rec.get("state") == "busy":
            # dead pid but the record says it was mid-job → it crashed
            status = f"crashed (log: {rec.get('log', '?')})"
        else:
            status = "-"
        click.echo(
            f"{m.name}  (aliases: {', '.join(m.aliases) or '-'})  "
            f"[daemon: {status}]  {m.description}"
        )


@model_group.command("stop-all")
def stop_all_cmd() -> None:
    """Stop all running model daemons."""
    n = 0
    for rec in daemon.list_daemons():
        if daemon.stop(rec["model"]):
            n += 1
    click.echo(f"stopped {n} daemon(s)")


model_group.add_command(list_cmd, name="ls")


@model_group.command("show")
@click.argument("name")
def show_cmd(name: str) -> None:
    try:
        m = models.get(name)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc
    cfg = Config.load()
    click.echo(f"name: {m.name}")
    click.echo(f"aliases: {', '.join(m.aliases) or '-'}")
    click.echo(f"description: {m.description}")
    click.echo(f"backends: {', '.join(b.value for b in m.supported_backends)}")
    click.echo(f"weights path: {cfg.model_path(m.name) or '(not set)'}")
    click.echo(f"model options: run `ig {m.name} gen --help`")
    click.echo("config keys:")
    for key, help_text in m.config_keys.items():
        click.echo(f"  {key}: {help_text}")


model_group.add_command(show_cmd, name="get")


@model_group.command("clean")
@click.option(
    "--all",
    "all_",
    is_flag=True,
    default=False,
    help="also truncate logs of running daemons",
)
@click.option(
    "--older-than",
    type=int,
    default=None,
    metavar="DAYS",
    help="only remove jobs finished more than DAYS days ago",
)
def clean_cmd(all_: bool, older_than: int | None) -> None:
    """Remove finished-job records/logs and dead-daemon logs."""
    from .. import jobs as jobs_mod

    stats = jobs_mod.clean(older_than_days=older_than, truncate_running=all_)
    click.echo(
        f"removed {stats['jobs']} job(s), {stats['logs']} dead-daemon log(s); "
        f"truncated {stats['truncated']} running log(s)"
    )


@model_group.command("jobs")
@click.argument("job_id", required=False)
def jobs_cmd(job_id: str | None) -> None:
    """List background (--queue) jobs, or show one with JOB_ID."""
    from .. import jobs as jobs_mod

    if job_id is None:
        rows = jobs_mod.list_jobs()
        if not rows:
            click.echo("no jobs")
            return
        for r in rows:
            started = _fmt_ts(r.get("started_at"))
            click.echo(f"{r['id']}  {r['model']:<12}  {r['status']:<8}  {started}  {r['out']}")
        return
    rec = jobs_mod.read_job(job_id)
    if rec is None:
        raise click.ClickException(f"no such job: {job_id}")
    for key in ("id", "model", "out", "status", "started_at", "finished_at", "log", "error"):
        if key in rec and rec[key] is not None:
            val = _fmt_ts(rec[key]) if key.endswith("_at") else rec[key]
            click.echo(f"{key}: {val}")
