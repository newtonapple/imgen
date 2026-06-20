"""`ig serve` — warm worker over a Unix socket. The model is a nested subcommand."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import click

from ..config import Config, resolve_weights_path
from ..platform import Backend, default_backend
from ..worker import serve as worker_serve


@dataclass
class ServeCommon:
    """Common `ig serve` options, stashed on the group context for the model subcommand."""

    socket_path: str | None
    log_path: str | None
    model_path: str | None
    backend: str | None


@click.group("serve")
@click.option(
    "--socket", "socket_path", default=None, type=click.Path(), help="Unix socket to listen on"
)
@click.option(
    "--log", "log_path", default=None, type=click.Path(), help="log file (default: stderr)"
)
@click.option("--model-path", "model_path", default=None)
@click.option("--backend", type=click.Choice([b.value for b in Backend]), default=None)
@click.pass_context
def serve(
    ctx: click.Context,
    socket_path: str | None,
    log_path: str | None,
    model_path: str | None,
    backend: str | None,
) -> None:
    """Run a warm worker: ig serve [options] <model> [model options]"""
    ctx.obj = ServeCommon(socket_path, log_path, model_path, backend)


def run_serve(model: Any, model_opts: dict[str, Any]) -> None:
    """Build the model's pipeline and run the warm worker (called by the model subcommand)."""
    common: ServeCommon = click.get_current_context().obj
    be = Backend(common.backend) if common.backend else default_backend()
    if be not in model.supported_backends:
        raise click.ClickException(
            f"{model.name} does not support backend {be.value}; supported: "
            f"{', '.join(b.value for b in model.supported_backends)}"
        )
    if common.socket_path is None:
        raise click.ClickException("missing required option --socket")

    cfg = Config.load()
    weights = resolve_weights_path(model.name, common.model_path, cfg)
    pipeline = model.build_pipeline(weights_path=weights, backend=be, **model_opts)

    log = open(common.log_path, "a") if common.log_path else sys.stderr
    log.write(f"ig serve: {model.name} warm on {common.socket_path}\n")
    log.flush()
    worker_serve(common.socket_path, pipeline)
