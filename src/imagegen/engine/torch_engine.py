"""PyTorch backend (CUDA, the DGX Spark) via the official `ideogram4` package.

Loads the reference `Ideogram4Pipeline` once and generates one image per call.

Implementation pending Spark bring-up: build the pipeline from the official
fp8 weights (ideogram-ai/ideogram-4-fp8 layout) and map our (caption, size,
preset, seed) call onto `pipe(...)` using `ideogram4.PRESETS`.
"""

from __future__ import annotations

from ..config import ModelSpec
from .base import GenerationResult


class TorchEngine:
    backend = "torch"

    def __init__(self, model: ModelSpec, device: str | None = None, **options):
        self.model = model
        self.device = device
        self.options = options
        self._pipe = None  # lazily-loaded ideogram4 Ideogram4Pipeline, kept warm
        # TODO(spark): build Ideogram4Pipeline from `model.path` once here.

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
            "TorchEngine.generate is pending Spark bring-up: call the reference "
            "Ideogram4Pipeline with ideogram4.PRESETS[preset], feeding the JSON "
            "caption as the prompt."
        )
