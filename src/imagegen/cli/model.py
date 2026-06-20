"""`ig model` subgroup: list / show / stop-all."""

from __future__ import annotations

import click

from .. import daemon, models
from ..config import Config


@click.group("model")
def model_group() -> None:
    """Inspect and configure models."""


@model_group.command("list")
def list_cmd() -> None:
    live_by_model = {rec["model"]: rec for rec in daemon.list_daemons() if rec.get("live")}
    for m in sorted(models.all_models(), key=lambda x: x.name):
        rec = live_by_model.get(m.name)
        status = f"running (pid {rec['pid']})" if rec is not None else "-"
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
