"""imagegen — Ideogram-4 image-generation pipeline with pluggable backends."""

from __future__ import annotations

from .config import EngineConfig, ModelSpec, weights_root
from .engine.base import GenerationResult, ImageEngine
from .engine.factory import create_pipeline, ensure_supported, resolve_backend
from .platform import Backend, default_backend, platform_summary

__all__ = [
    "Backend",
    "EngineConfig",
    "GenerationResult",
    "ImageEngine",
    "ModelSpec",
    "create_pipeline",
    "default_backend",
    "ensure_supported",
    "platform_summary",
    "resolve_backend",
    "weights_root",
]
