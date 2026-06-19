"""ideogram4 model plugin — wraps the existing factory, engines, and magic-prompt."""

from __future__ import annotations

import json
from pathlib import Path

import click

from .. import models
from ..config import ModelSpec
from ..engine.factory import create_pipeline
from ..magic_prompt.cli_provider import CliMagicPromptProvider
from ..pipeline import Pipeline
from ..platform import Backend

PRESETS = ["V4_TURBO_12", "V4_DEFAULT_20", "V4_QUALITY_48"]


@click.command("ideogram4")
@click.option("--preset", type=click.Choice(PRESETS), default="V4_DEFAULT_20", show_default=True)
@click.option(
    "--quantize",
    type=click.Choice(["4", "8"]),
    default=None,
    help="MLX: quantize fp8 to N bits on load (8 = int8); default keeps fp8",
)
@click.option(
    "--magic-model",
    "magic_model",
    default="codex - gpt-5.5",
    show_default=True,
    help="magic-prompt provider/model string",
)
@click.option("--target-elements", "target_elements", type=int, default=0, show_default=True)
@click.option(
    "--caption",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="prebuilt caption JSON; skips magic-prompt",
)
def _options(**_kwargs):  # callback unused; params are read via make_context
    pass


class Ideogram4Model:
    name = "ideogram4"
    aliases = ["ig4"]
    description = (
        "Ideogram 4 (fp8) — structured-caption text-to-image; MLX on Mac, PyTorch on CUDA."
    )
    supported_backends = [Backend.MLX, Backend.TORCH]
    model_options = _options

    def default_weights_path(self, cfg) -> Path | None:
        return cfg.model_path(self.name)

    def build_pipeline(self, *, weights_path: Path, backend: Backend, **opts) -> Pipeline:
        quantize = opts.get("quantize")
        engine = create_pipeline(
            ModelSpec(name=self.name, path=Path(weights_path), backend=backend),
            backend=backend,
            quantize=int(quantize) if quantize else None,
        )
        provider = CliMagicPromptProvider(opts.get("magic_model", "codex - gpt-5.5"))
        return Pipeline(engine=engine, magic_prompt=provider)

    def run_one(self, pipeline, *, prompt, width, height, seed, **opts):
        preset = opts.get("preset", "V4_DEFAULT_20")
        caption = opts.get("caption")
        if caption:
            cap = json.loads(Path(caption).read_text())
            return pipeline.generate(cap, width=width, height=height, preset=preset, seed=seed)
        return pipeline.run(
            prompt,
            width=width,
            height=height,
            preset=preset,
            seed=seed,
            target_elements=opts.get("target_elements", 0),
        )


models.register(Ideogram4Model())
