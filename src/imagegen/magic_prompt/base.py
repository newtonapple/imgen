# src/imagegen/magic_prompt/base.py
from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class MagicPromptProvider(Protocol):
    def expand(self, prompt: str, *, width: int, height: int, target_elements: int = 0) -> dict: ...
