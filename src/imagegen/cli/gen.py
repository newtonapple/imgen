"""`ig gen` — model-agnostic image generation."""

from __future__ import annotations

import json
import sys

import click

from .. import models
from ..config import Config, resolve_weights_path
from ..platform import Backend, default_backend


@click.command("gen", context_settings={"ignore_unknown_options": True})
@click.option("-p", "--prompt", default=None, help="text prompt (the model expands/uses it)")
@click.option("-w", "--width", type=int, default=1024, show_default=True)
@click.option("-h", "--height", type=int, default=1024, show_default=True)
@click.option("--seed", type=int, default=None, help="RNG seed (omit = random)")
@click.option("-o", "--out", required=True, type=click.Path(), help="output image path")
@click.option("--model-path", "model_path", default=None, help="override the model's weights path")
@click.option(
    "--backend",
    type=click.Choice([b.value for b in Backend]),
    default=None,
    help="inference backend (default: auto)",
)
@click.argument("model_name")
@click.argument("model_args", nargs=-1, type=click.UNPROCESSED)
def gen(
    prompt: str | None,
    width: int,
    height: int,
    seed: int | None,
    out: str,
    model_path: str | None,
    backend: str | None,
    model_name: str,
    model_args: tuple[str, ...],
) -> None:
    """ig gen [globals] <model> -- [model opts]"""
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

    if seed is None:
        sys.stderr.write(
            "warning: no --seed; re-seeding will likely change the image substantially\n"
        )

    cfg = Config.load()
    weights = resolve_weights_path(model.name, model_path, cfg)
    pipeline = model.build_pipeline(weights_path=weights, backend=be, **opts)
    result = model.run_one(pipeline, prompt=prompt, width=width, height=height, seed=seed, **opts)
    result.image.save(out)
    summary = {
        field: getattr(result, field, None)
        for field in ("seed", "width", "height", "preset", "backend", "duration_s")
    }
    summary["out"] = out
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
