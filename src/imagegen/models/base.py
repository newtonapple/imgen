"""Model plugin protocol. Each model registers one of these."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..platform import Backend  # re-export

if TYPE_CHECKING:
    import click

    from ..config import Config
    from ..engine.base import GenerationResult
    from ..pipeline import Pipeline

__all__ = ["Backend", "Model"]


@runtime_checkable
class Model(Protocol):
    name: str
    aliases: list[str]
    description: str
    supported_backends: list[Backend]
    model_options: "click.Command"  # parses post-`--` args into params

    def default_weights_path(self, cfg: "Config") -> Path | None: ...

    def build_pipeline(
        self, *, weights_path: Path, backend: Backend, **opts: Any
    ) -> "Pipeline": ...

    def run_one(
        self,
        pipeline: Any,
        *,
        prompt: str | None,
        width: int,
        height: int,
        seed: int | None,
        **opts: Any,
    ) -> "GenerationResult": ...
