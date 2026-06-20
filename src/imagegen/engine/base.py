"""ImageEngine interface + result type.

An engine is a *warm, load-once inference pipeline*: one model held in memory,
`generate` called many times (one image per call). Backends implement this; the
factory builds the right one for the platform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass
class GenerationResult:
    image: "Image"
    seed: int  # the actual seed used (resolved if the caller passed None)
    width: int  # resolved/rounded dimensions actually sampled
    height: int
    preset: str
    caption: dict[str, Any]  # the JSON caption fed to the model
    backend: str  # "mlx" | "torch"
    duration_s: float


@runtime_checkable
class ImageEngine(Protocol):
    """Load-once, reuse-many inference pipeline. The unit a worker wraps."""

    backend: str

    def generate(
        self,
        caption: dict[str, Any] | str,
        *,
        width: int,
        height: int,
        preset: str = "V4_DEFAULT_20",
        seed: int | None = None,
    ) -> GenerationResult: ...
