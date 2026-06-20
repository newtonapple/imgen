"""Model plugin protocol. Each model registers one of these."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..platform import Backend  # re-export

if TYPE_CHECKING:
    import click

    from ..config import Config, Secrets
    from ..engine.base import GenerationResult
    from ..pipeline import Pipeline

__all__ = ["Backend", "Model"]


@runtime_checkable
class Model(Protocol):
    name: str
    aliases: list[str]
    description: str
    supported_backends: list[Backend]
    gen_options: "list[click.Parameter]"  # per-request options for `gen`
    config_keys: dict[str, str]  # `config set` key -> help text

    def build_pipeline(
        self, *, weights_path: Path, backend: Backend, config: "Config", secrets: "Secrets"
    ) -> "Pipeline": ...

    def run_one(self, pipeline: Any, **gen_opts: Any) -> "GenerationResult": ...
