"""`ig gen` — generate one image. The model is a nested subcommand."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

import click

from ..config import Config, resolve_weights_path
from ..platform import Backend, default_backend


@dataclass
class GenCommon:
    """Common `ig gen` options, stashed on the group context for the model subcommand."""

    prompt: str | None
    width: int
    height: int
    seed: int | None
    out: str | None
    model_path: str | None
    backend: str | None


@click.group("gen")
@click.option("-p", "--prompt", default=None, help="text prompt (the model expands/uses it)")
@click.option("-w", "--width", type=int, default=1024, show_default=True)
@click.option("-h", "--height", type=int, default=1024, show_default=True)
@click.option("--seed", type=int, default=None, help="RNG seed (omit = random)")
@click.option("-o", "--out", default=None, type=click.Path(), help="output image path")
@click.option("--model-path", "model_path", default=None, help="override the model's weights path")
@click.option(
    "--backend",
    type=click.Choice([b.value for b in Backend]),
    default=None,
    help="inference backend (default: auto)",
)
@click.pass_context
def gen(
    ctx: click.Context,
    prompt: str | None,
    width: int,
    height: int,
    seed: int | None,
    out: str | None,
    model_path: str | None,
    backend: str | None,
) -> None:
    """Generate one image: ig gen [options] <model> [model options]"""
    ctx.obj = GenCommon(prompt, width, height, seed, out, model_path, backend)


def run_gen(model: Any, model_opts: dict[str, Any]) -> None:
    """Build the model's pipeline and generate one image (called by the model subcommand)."""
    common: GenCommon = click.get_current_context().obj
    be = Backend(common.backend) if common.backend else default_backend()
    if be not in model.supported_backends:
        raise click.ClickException(
            f"{model.name} does not support backend {be.value}; supported: "
            f"{', '.join(b.value for b in model.supported_backends)}"
        )
    if common.out is None:
        raise click.ClickException("missing required option -o/--out")
    if common.seed is None:
        sys.stderr.write(
            "warning: no --seed; re-seeding will likely change the image substantially\n"
        )

    cfg = Config.load()
    weights = resolve_weights_path(model.name, common.model_path, cfg)
    pipeline = model.build_pipeline(weights_path=weights, backend=be, **model_opts)
    result = model.run_one(
        pipeline,
        prompt=common.prompt,
        width=common.width,
        height=common.height,
        seed=common.seed,
        **model_opts,
    )
    result.image.save(common.out)
    summary: dict[str, Any] = {
        field: getattr(result, field, None)
        for field in ("seed", "width", "height", "preset", "backend", "duration_s")
    }
    summary["out"] = common.out
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
