"""`ig serve` — warm worker over a Unix socket."""

from __future__ import annotations

import sys

import click

from .. import models
from ..config import Config, resolve_weights_path
from ..platform import Backend, default_backend
from ..worker import serve as worker_serve


@click.command("serve", context_settings={"ignore_unknown_options": True})
@click.option(
    "--socket", "socket_path", required=True, type=click.Path(), help="Unix socket to listen on"
)
@click.option(
    "--log", "log_path", default=None, type=click.Path(), help="log file (default: stderr)"
)
@click.option("--model-path", "model_path", default=None)
@click.option("--backend", type=click.Choice([b.value for b in Backend]), default=None)
@click.argument("model_name")
@click.argument("model_args", nargs=-1, type=click.UNPROCESSED)
def serve(socket_path, log_path, model_path, backend, model_name, model_args):
    """ig serve [serve globals] <model> -- [model opts]"""
    try:
        model = models.get(model_name)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc
    args = list(model_args)
    if args and args[0] == "--":
        args = args[1:]
    with model.model_options.make_context(model.name, args) as ctx:
        opts = dict(ctx.params)

    be = Backend(backend) if backend else default_backend()
    if be not in model.supported_backends:
        raise click.ClickException(
            f"{model.name} does not support backend {be.value}; supported: "
            f"{', '.join(b.value for b in model.supported_backends)}"
        )
    cfg = Config.load()
    weights = resolve_weights_path(model.name, model_path, cfg)
    pipeline = model.build_pipeline(weights_path=weights, backend=be, **opts)

    log = open(log_path, "a") if log_path else sys.stderr
    log.write(f"ig serve: {model.name} warm on {socket_path}\n")
    log.flush()
    worker_serve(socket_path, pipeline)
