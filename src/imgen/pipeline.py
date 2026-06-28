"""Orchestrates magic-prompt + engine. Pure glue; the caller saves the image."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .engine.base import GenerationResult, ImageEngine
from .magic_prompt.base import MagicPromptProvider


class Pipeline:
    def __init__(
        self,
        engine: ImageEngine,
        magic_prompt: MagicPromptProvider | None = None,
        *,
        magic_factory: "Callable[[str | None, str | None], MagicPromptProvider] | None" = None,
    ):
        self.engine = engine
        self.magic_prompt = magic_prompt
        self._magic_factory = magic_factory

    def _provider_for(
        self, magic_provider: str | None, magic_model: str | None
    ) -> MagicPromptProvider | None:
        if (magic_provider is None and magic_model is None) or self._magic_factory is None:
            return self.magic_prompt
        return self._magic_factory(magic_provider, magic_model)

    def magic(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        target_elements: int = 0,
        magic_provider: str | None = None,
        magic_model: str | None = None,
    ) -> dict[str, Any]:
        provider = self._provider_for(magic_provider, magic_model)
        if provider is None:
            raise RuntimeError("no MagicPromptProvider configured")
        return provider.expand(prompt, width=width, height=height, target_elements=target_elements)

    def generate(
        self,
        caption: dict[str, Any] | str,
        *,
        width: int,
        height: int,
        preset: str = "V4_DEFAULT_20",
        seed: int | None = None,
        progress: "Callable[[int, int], None] | None" = None,
    ) -> GenerationResult:
        return self.engine.generate(
            caption, width=width, height=height, preset=preset, seed=seed, progress=progress
        )

    def run(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        preset: str = "V4_DEFAULT_20",
        seed: int | None = None,
        target_elements: int = 0,
        magic_provider: str | None = None,
        magic_model: str | None = None,
    ) -> GenerationResult:
        caption = self.magic(
            prompt,
            width=width,
            height=height,
            target_elements=target_elements,
            magic_provider=magic_provider,
            magic_model=magic_model,
        )
        return self.generate(caption, width=width, height=height, preset=preset, seed=seed)
