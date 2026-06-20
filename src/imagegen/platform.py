"""Platform detection: pick the inference backend + dependency set for this host.

Apple Silicon -> MLX (mflux). Linux/CUDA (the DGX Spark) -> PyTorch reference
pipeline. Kept dependency-free so it can run before any heavy deps are installed.
"""

from __future__ import annotations

import platform as _platform
from enum import Enum
from typing import Any


class Backend(str, Enum):
    MLX = "mlx"
    TORCH = "torch"


def is_apple_silicon() -> bool:
    return _platform.system() == "Darwin" and _platform.machine() in {"arm64", "aarch64"}


def is_linux() -> bool:
    return _platform.system() == "Linux"


def cuda_available() -> bool:
    """True only if a CUDA-capable torch is importable and a device is present."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def default_backend() -> Backend:
    """MLX on Apple Silicon, otherwise the PyTorch/CUDA backend."""
    return Backend.MLX if is_apple_silicon() else Backend.TORCH


def platform_summary() -> dict[str, Any]:
    return {
        "system": _platform.system(),
        "machine": _platform.machine(),
        "apple_silicon": is_apple_silicon(),
        "cuda": cuda_available(),
        "default_backend": default_backend().value,
    }
