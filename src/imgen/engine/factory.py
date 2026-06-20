"""Factory: take a model + inference backend, build the inference pipeline.

Backends are imported lazily so importing the factory never pulls in torch or
mflux — only the selected backend's dependencies load. This is what keeps the
Mac (MLX) install free of torch/bitsandbytes and vice-versa.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..config import ModelSpec
from ..platform import Backend, default_backend, is_apple_silicon

if TYPE_CHECKING:  # typing only, avoids importing backend modules at runtime
    from .base import ImageEngine


def resolve_backend(
    backend: Backend | str | None = None,
    model: ModelSpec | None = None,
) -> Backend:
    """Explicit backend wins; else the model's backend; else the platform default."""
    if backend is not None:
        return Backend(backend)
    if model is not None:
        return model.backend
    return default_backend()


def ensure_supported(backend: Backend) -> None:
    """Raise a clear error if this host can't run the requested backend."""
    if backend == Backend.MLX and not is_apple_silicon():
        raise RuntimeError("MLX backend requires Apple Silicon (mflux).")
    if backend == Backend.TORCH:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch backend requires the 'cuda' extra: pip install '.[cuda]'."
            ) from exc


def create_pipeline(
    model: ModelSpec,
    backend: Backend | str | None = None,
    **options: Any,
) -> "ImageEngine":
    """Build and return a ready inference pipeline (an ImageEngine) for `model`.

    `backend` selects the inference engine; when omitted it is taken from the
    model, then from the platform default.
    """
    backend = resolve_backend(backend, model)
    ensure_supported(backend)
    if backend == Backend.MLX:
        from .mlx_engine import MlxEngine

        return MlxEngine(model, **options)
    if backend == Backend.TORCH:
        from .torch_engine import TorchEngine

        return TorchEngine(model, **options)
    raise ValueError(f"Unknown backend: {backend!r}")
