"""Shared magic-prompt instruction building + JSON extraction (provider-agnostic).

Both provider families (CLI subprocess and HTTP) build the same Ideogram-4
instruction and parse model output identically through these helpers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..engine.resolution import aspect_ratio

_TEMPLATE = (Path(__file__).parent / "templates" / "ideogram4_caption.txt").read_text()


def build_instruction(prompt: str, *, width: int, height: int, target_elements: int = 0) -> str:
    ar = aspect_ratio(width, height)
    instruction = (
        f"{_TEMPLATE}\n\nUSER PROMPT:\n{prompt}\n\n[ASPECT RATIO]\n"
        f'Return aspect_ratio exactly "{ar}" and compose bbox for a {width}x{height} canvas.'
    )
    if target_elements > 0:
        instruction += (
            f"\n\n[ELEMENT COUNT]\nReturn a target of exactly {target_elements} "
            "entries in compositional_deconstruction.elements (add real subjects/"
            "props/text, not body parts); include every explicit text element even if it exceeds the target."
        )
    return instruction


def extract_json(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end != -1 and end > start else text


def finalize_caption(text: str, *, width: int, height: int) -> dict[str, Any]:
    parsed = json.loads(extract_json(text))
    if not isinstance(parsed, dict):
        raise ValueError("magic-prompt output was not a JSON object")
    result: dict[str, Any] = parsed
    result["aspect_ratio"] = aspect_ratio(width, height)
    return result


def run_magic_prompt(
    call_model: Callable[[str], str],
    *,
    prompt: str,
    width: int,
    height: int,
    target_elements: int = 0,
) -> dict[str, Any]:
    """Build the instruction, call the model, parse — retrying once on bad JSON.

    `call_model(instruction)` returns the model's raw text. Hard failures inside
    `call_model` (process error, HTTP error) propagate immediately; only invalid
    JSON triggers the single retry.
    """
    instruction = build_instruction(
        prompt, width=width, height=height, target_elements=target_elements
    )
    last_err: Exception | None = None
    for _ in range(2):
        text = call_model(instruction)
        try:
            return finalize_caption(text, width=width, height=height)
        except (json.JSONDecodeError, ValueError) as exc:
            last_err = exc
    raise RuntimeError(f"magic-prompt returned invalid JSON after a retry: {last_err}")
