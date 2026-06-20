"""Per-generation metadata: build the summary dict + write the `out.png.json` sidecar."""

from __future__ import annotations

import json
from typing import Any


def build_summary(
    out: str, result: dict[str, Any], *, model: str, prompt: str | None
) -> dict[str, Any]:
    return {
        "seed": result.get("seed"),
        "width": result.get("width"),
        "height": result.get("height"),
        "preset": result.get("preset"),
        "backend": result.get("backend"),
        "duration_s": result.get("duration_s"),
        "out": out,
        "model": model,
        "prompt": prompt,
        "caption": result.get("caption"),
    }


def write_sidecar(out: str, summary: dict[str, Any]) -> None:
    with open(out + ".json", "w") as f:
        json.dump(summary, f, indent=2)
