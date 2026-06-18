"""MLX backend (Apple Silicon) via mflux.

Loads the Ideogram-4 weights once, keeps the model warm in unified memory, and
generates one image per `generate` call.

Implementation pending the MLX spike: wraps mflux's in-process Ideogram-4
generator (mflux @ ideogram-mlx-forge-loader). The weights directory auto-detects
precision (bf16 / int8) from `split_model.json`. Preset names match the reference
package (V4_QUALITY_48 / V4_DEFAULT_20 / V4_TURBO_12).
"""

from __future__ import annotations

from ..config import ModelSpec
from .base import GenerationResult


class MlxEngine:
    backend = "mlx"

    def __init__(self, model: ModelSpec, **options):
        self.model = model
        self.options = options
        self._generator = None  # lazily-loaded mflux generator, kept warm
        # TODO(mlx-spike): import the mflux Ideogram-4 generator and load
        # `model.path` into memory once here.

    def generate(
        self,
        caption: dict | str,
        *,
        width: int,
        height: int,
        preset: str = "V4_DEFAULT_20",
        seed: int | None = None,
    ) -> GenerationResult:
        raise NotImplementedError(
            "MlxEngine.generate is pending the MLX spike: drive mflux's in-process "
            "Ideogram-4 generator (the mflux-generate-ideogram4 equivalent), feeding "
            "the JSON caption as the prompt."
        )
