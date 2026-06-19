"""`ig model` subgroup: list / show / set-path."""

from __future__ import annotations

import click

from .. import models
from ..config import Config, _config_path


@click.group("model")
def model_group() -> None:
    """Inspect and configure models."""


@model_group.command("list")
def list_cmd() -> None:
    for m in sorted(models.all_models(), key=lambda x: x.name):
        click.echo(f"{m.name}  (aliases: {', '.join(m.aliases) or '-'})  {m.description}")


model_group.add_command(list_cmd, name="ls")


@model_group.command("show")
@click.argument("name")
def show_cmd(name: str) -> None:
    m = models.get(name)
    cfg = Config.load()
    click.echo(f"name: {m.name}")
    click.echo(f"aliases: {', '.join(m.aliases) or '-'}")
    click.echo(f"description: {m.description}")
    click.echo(f"backends: {', '.join(b.value for b in m.supported_backends)}")
    click.echo(f"weights path: {cfg.model_path(m.name) or '(not set)'}")
    click.echo("model options (pass after `--`):")
    with click.Context(m.model_options) as ctx:
        click.echo(m.model_options.get_help(ctx))


model_group.add_command(show_cmd, name="get")


@model_group.command("set-path")
@click.argument("name")
@click.argument("path", type=click.Path())
def set_path_cmd(name: str, path: str) -> None:
    models.get(name)  # validate the model exists
    cfg = Config.load()
    cfg.set_model_path(name, path)
    cfg.save()
    click.echo(f"set {name} weights path -> {path}  ({_config_path()})")
