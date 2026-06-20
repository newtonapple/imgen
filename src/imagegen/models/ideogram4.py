"""ideogram4 model plugin — wraps the existing factory, engines, and magic-prompt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from .. import models
from ..config import Config, ModelSpec, Secrets
from ..engine.base import GenerationResult
from ..engine.factory import create_pipeline
from ..magic_prompt.providers import make_magic_provider, resolve_magic_settings
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
    "--magic-prompt-provider",
    "--mp",
    "magic_prompt_provider",
    default=None,
    help="magic-prompt provider: codex|claude|pi|anthropic|openai|openrouter",
)
@click.option("--magic-model", "--mm", "magic_model", default=None, help="magic-prompt model id")
@click.option(
    "--set-magic-prompt-provider",
    "--set-mp",
    "set_magic_prompt_provider",
    default=None,
    help="persist the default magic-prompt provider",
)
@click.option(
    "--set-magic-model",
    "--set-mm",
    "set_magic_model",
    default=None,
    help="persist the default magic-prompt model",
)
@click.option(
    "--set-magic-prompt-api-key",
    "--set-mk",
    "set_magic_prompt_api_key",
    default=None,
    help="store an API key for the active provider in ~/.config/ig/secrets.toml",
)
@click.option("--target-elements", "target_elements", type=int, default=0, show_default=True)
@click.option(
    "--caption",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="prebuilt caption JSON; skips magic-prompt",
)
def _options(**_kwargs: Any) -> None:  # callback unused; params are read via make_context
    pass


class Ideogram4Model:
    name = "ideogram4"
    aliases = ["ig4"]
    description = (
        "Ideogram 4 (fp8) — structured-caption text-to-image; MLX on Mac, PyTorch on CUDA."
    )
    supported_backends = [Backend.MLX, Backend.TORCH]
    model_options = _options

    def default_weights_path(self, cfg: Config) -> Path | None:
        return cfg.model_path(self.name)

    def build_pipeline(self, *, weights_path: Path, backend: Backend, **opts: Any) -> Pipeline:
        quantize = opts.get("quantize")
        engine = create_pipeline(
            ModelSpec(name=self.name, path=Path(weights_path), backend=backend),
            backend=backend,
            quantize=int(quantize) if quantize else None,
        )
        config = Config.load()
        secrets = Secrets.load()
        provider, model = resolve_magic_settings(opts, config=config, secrets=secrets)
        magic = make_magic_provider(provider, model, secrets=secrets)
        return Pipeline(engine=engine, magic_prompt=magic)

    def run_one(
        self,
        pipeline: Any,
        *,
        prompt: str | None,
        width: int,
        height: int,
        seed: int | None,
        **opts: Any,
    ) -> GenerationResult:
        preset = opts.get("preset", "V4_DEFAULT_20")
        caption = opts.get("caption")
        result: GenerationResult
        if caption:
            cap = json.loads(Path(caption).read_text())
            result = pipeline.generate(cap, width=width, height=height, preset=preset, seed=seed)
        else:
            result = pipeline.run(
                prompt,
                width=width,
                height=height,
                preset=preset,
                seed=seed,
                target_elements=opts.get("target_elements", 0),
            )
        return result


models.register(Ideogram4Model())
