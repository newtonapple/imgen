"""Size rounding + aspect ratio, matching the ComfyUI workflow."""
from __future__ import annotations
from math import gcd, floor

_MULTIPLE = 16
_FLOOR = 256


def _round(v: int) -> int:
    # Round to nearest multiple of 16, rounding .5 up
    rounded = floor(v / _MULTIPLE + 0.5) * _MULTIPLE
    return max(int(rounded), _FLOOR)


def resolve_size(width: int, height: int) -> tuple[int, int]:
    return _round(int(width)), _round(int(height))


def aspect_ratio(width: int, height: int) -> str:
    w, h = int(width), int(height)
    if w <= 0 or h <= 0:
        raise ValueError("width and height must be positive")
    d = gcd(w, h)
    return f"{w // d}:{h // d}"
