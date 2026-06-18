"""MLX backend (Apple Silicon) via mflux.

Loads the Ideogram-4 weights once (warm in unified memory) and generates one
image per `generate` call. Uses the official HF fp8 checkpoint layout, which
mflux loads directly. See docs/notes/mflux-ideogram4-api.md for the API.
"""

from __future__ import annotations

import json
import random
import time

from ..caption import model_caption, validate_caption
from ..config import ModelSpec
from .base import GenerationResult

_SEED_MAX = 2**31 - 1


class MlxEngine:
    backend = "mlx"

    def __init__(self, model: ModelSpec, **options):
        self.model = model
        self.options = options
        # Heavy: load the model once and keep it warm. Imported lazily here so
        # the factory never imports mflux unless the MLX backend is selected.
        from mflux.models.common.config import ModelConfig
        from mflux.models.ideogram4 import Ideogram4

        self._generator = Ideogram4(
            model_path=str(model.path),
            model_config=ModelConfig.ideogram4_fp8(),
        )

    def generate(
        self,
        caption: dict | str,
        *,
        width: int,
        height: int,
        preset: str = "V4_DEFAULT_20",
        seed: int | None = None,
    ) -> GenerationResult:
        caption_dict = caption if isinstance(caption, dict) else json.loads(caption)
        validate_caption(caption_dict)  # warn-not-fail
        if seed is None:
            seed = random.randint(0, _SEED_MAX)

        t0 = time.time()
        result = self._generator.generate_image(
            prompt=model_caption(caption_dict),  # drop non-schema keys (e.g. aspect_ratio)
            seed=seed,
            width=width,
            height=height,
            preset=preset,
        )
        image = getattr(result, "image", result)  # GeneratedImage wraps a PIL image
        return GenerationResult(
            image=image,
            seed=seed,
            width=width,
            height=height,
            preset=preset,
            caption=caption_dict,
            backend="mlx",
            duration_s=time.time() - t0,
        )
