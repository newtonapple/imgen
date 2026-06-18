"""Configuration: model specs + engine config.

Weights live OUTSIDE the repo (an external volume on the Mac, ~/ai/image-gen on
the Spark) and are referenced by path only — never committed, never in iCloud.
The weights root is supplied via the IMAGEGEN_WEIGHTS_ROOT env var or an explicit
path, so nothing is hardcoded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .platform import Backend, default_backend

DEFAULT_PRESET = "V4_DEFAULT_20"
WEIGHTS_ROOT_ENV = "IMAGEGEN_WEIGHTS_ROOT"


@dataclass(frozen=True)
class ModelSpec:
    """Identifies a set of weights on disk for a given backend."""

    name: str
    path: Path
    backend: Backend
    default_preset: str = DEFAULT_PRESET

    @classmethod
    def from_path(
        cls,
        path: str | os.PathLike,
        *,
        backend: Backend | str | None = None,
        name: str | None = None,
        default_preset: str = DEFAULT_PRESET,
    ) -> "ModelSpec":
        p = Path(path).expanduser()
        be = Backend(backend) if backend is not None else default_backend()
        return cls(name=name or p.name, path=p, backend=be, default_preset=default_preset)


@dataclass
class EngineConfig:
    """Everything an engine needs to load + run, beyond the model itself."""

    model: ModelSpec
    device: str | None = None  # backend-specific; None = auto
    options: dict = field(default_factory=dict)


def weights_root() -> Path | None:
    """Root directory holding per-model weight subdirs, if configured."""
    v = os.environ.get(WEIGHTS_ROOT_ENV)
    return Path(v).expanduser() if v else None
