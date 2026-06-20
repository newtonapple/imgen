"""ideogram4 model plugin — wraps the existing factory, engines, and magic-prompt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from .. import models
from ..config import Config, ModelSpec, Secrets, resolve_quantize
from ..engine.base import GenerationResult
from ..engine.factory import create_pipeline
from ..magic_prompt.providers import (
    ALL_PROVIDERS,
    effective_magic,
    make_magic_provider,
    resolve_magic_provider,
)
from ..pipeline import Pipeline
from ..platform import Backend

PRESETS = ["V4_TURBO_12", "V4_DEFAULT_20", "V4_QUALITY_48"]

GEN_OPTIONS: list[click.Parameter] = [
    click.Option(["-p", "--prompt"], default=None, help="text prompt (the model expands/uses it)"),
    click.Option(["-w", "--width"], type=int, default=1024, show_default=True),
    click.Option(["-h", "--height"], type=int, default=1024, show_default=True),
    click.Option(["--seed"], type=int, default=None, help="RNG seed (omit = random)"),
    click.Option(
        ["--preset"], type=click.Choice(PRESETS), default="V4_DEFAULT_20", show_default=True
    ),
    click.Option(["--target-elements", "target_elements"], type=int, default=0, show_default=True),
    click.Option(
        ["--caption"],
        type=click.Path(exists=True, dir_okay=False),
        default=None,
        help="prebuilt caption JSON; skips magic-prompt",
    ),
    click.Option(
        ["--magic-provider", "--mp", "magic_provider"],
        default=None,
        type=click.Choice(sorted(ALL_PROVIDERS)),
        help="per-request magic-prompt provider",
    ),
    click.Option(
        ["--magic-model", "--mm", "magic_model"],
        default=None,
        help="per-request magic-prompt model",
    ),
]

CONFIG_KEYS = {
    "weights-path": "path to the model weights directory",
    "quantize": "4|8 MLX quantization on load (empty to clear)",
    "backend": "mlx|torch default backend (empty to clear)",
    "magic-provider": "magic-prompt provider: codex|claude|pi|anthropic|openai|openrouter",
    "magic-model": "magic-prompt model id",
}


class Ideogram4Model:
    name = "ideogram4"
    aliases = ["ig4"]
    description = (
        "Ideogram 4 (fp8) — structured-caption text-to-image; MLX on Mac, PyTorch on CUDA."
    )
    supported_backends = [Backend.MLX, Backend.TORCH]
    gen_options = GEN_OPTIONS
    config_keys = CONFIG_KEYS

    def build_pipeline(
        self, *, weights_path: Path, backend: Backend, config: Config, secrets: Secrets
    ) -> Pipeline:
        quantize = resolve_quantize(config, self.name)
        engine = create_pipeline(
            ModelSpec(name=self.name, path=Path(weights_path), backend=backend),
            backend=backend,
            quantize=int(quantize) if quantize else None,
        )
        provider, mp_model = resolve_magic_provider(config)
        default_magic = make_magic_provider(provider, mp_model, secrets=secrets)

        def magic_factory(p_override: str | None, m_override: str | None) -> Any:
            p, m = effective_magic(provider, mp_model, provider=p_override, model=m_override)
            if p == provider and m == mp_model:
                return default_magic
            return make_magic_provider(p, m, secrets=secrets)

        return Pipeline(engine=engine, magic_prompt=default_magic, magic_factory=magic_factory)

    def run_one(self, pipeline: Any, **g: Any) -> GenerationResult:
        preset = g.get("preset", "V4_DEFAULT_20")
        caption = g.get("caption")
        result: GenerationResult
        if caption:
            cap = json.loads(Path(caption).read_text())
            result = pipeline.generate(
                cap, width=g["width"], height=g["height"], preset=preset, seed=g["seed"]
            )
        else:
            result = pipeline.run(
                g["prompt"],
                width=g["width"],
                height=g["height"],
                preset=preset,
                seed=g["seed"],
                target_elements=g.get("target_elements", 0),
            )
        return result


models.register(Ideogram4Model())
