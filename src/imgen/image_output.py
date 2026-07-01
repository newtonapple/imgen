# src/imgen/image_output.py
"""Resolve an output format and save a PIL image with format-appropriate options.

Kept separate from the worker/CLI so the (image, path, format) → file behaviour
is unit-testable without a daemon or a real engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_FORMATS = ("png", "webp")


def resolve_format(out_path: str, explicit: str | None) -> str:
    """Return 'png' or 'webp'. Explicit choice wins; otherwise infer from the
    output extension; otherwise default to 'png' (backward compatible)."""
    if explicit in _FORMATS:
        return explicit  # type: ignore[return-value]
    ext = Path(out_path).suffix.lower().lstrip(".")
    if ext in _FORMATS:
        return ext
    return "png"


def save_image(image: Any, out_path: str, fmt: str) -> None:
    """Save a PIL image. WebP is always lossless; PNG uses Pillow's default
    encoder (byte-identical to `image.save(path)` for a .png path)."""
    if fmt == "webp":
        image.save(out_path, format="WEBP", lossless=True, quality=100, method=6)
    else:
        image.save(out_path, format="PNG")
