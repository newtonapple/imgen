"""Model plugin protocol. Each model registers one of these."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..platform import Backend  # re-export

if TYPE_CHECKING:
    import click

__all__ = ["Backend", "Model"]


@runtime_checkable
class Model(Protocol):
    name: str
    aliases: list[str]
    description: str
    supported_backends: list[Backend]
    model_options: "click.Command"  # parses post-`--` args into params

    def default_weights_path(self, cfg) -> Path | None: ...

    def build_pipeline(self, *, weights_path: Path, backend: Backend, **opts): ...

    def run_one(self, pipeline, *, prompt: str, width: int, height: int, seed: int | None, **opts): ...
