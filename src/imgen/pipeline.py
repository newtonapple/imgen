"""Orchestrates magic-prompt + engine. Pure glue; the caller saves the image."""

from __future__ import annotations

from typing import Any

from .engine.base import GenerationResult, ImageEngine
from .magic_prompt.base import MagicPromptProvider


class Pipeline:
    def __init__(
        self,
        engine: ImageEngine,
        magic_prompt: MagicPromptProvider | None = None,
    ):
        self.engine = engine
        self.magic_prompt = magic_prompt

    def magic(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        target_elements: int = 0,
    ) -> dict[str, Any]:
        if self.magic_prompt is None:
            raise RuntimeError("no MagicPromptProvider configured")
        return self.magic_prompt.expand(
            prompt, width=width, height=height, target_elements=target_elements
        )

    def generate(
        self,
        caption: dict[str, Any] | str,
        *,
        width: int,
        height: int,
        preset: str = "V4_DEFAULT_20",
        seed: int | None = None,
    ) -> GenerationResult:
        return self.engine.generate(caption, width=width, height=height, preset=preset, seed=seed)

    def run(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        preset: str = "V4_DEFAULT_20",
        seed: int | None = None,
        target_elements: int = 0,
    ) -> GenerationResult:
        caption = self.magic(prompt, width=width, height=height, target_elements=target_elements)
        return self.generate(caption, width=width, height=height, preset=preset, seed=seed)
