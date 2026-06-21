"""Ideogram hosted magic-prompt provider (https://api.ideogram.ai).

Unlike the chat providers (`openai`/`anthropic`/`openrouter`), Ideogram's
magic-prompt endpoint expands a plain `text_prompt` into the structured JSON
caption directly — server-side, no local model, no system prompt, and *not* using
our own caption instruction template (`_caption_prompt`). It is the reference
implementation of the magic-prompt step (its system prompt is open source), so we
just send `(text_prompt, aspect_ratio)` and return its `json_prompt` object.

Get an API key at https://developer.ideogram.ai; the expansion itself is free.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from ..engine.resolution import aspect_ratio

IDEOGRAM_API_KEY_ENV = "IDEOGRAM_API_KEY"
IDEOGRAM_MAGIC_PROMPT_URL = "https://api.ideogram.ai/v1/ideogram-v4/magic-prompt"

# Aspect-ratio buckets accepted by the /v1/ideogram-v4/magic-prompt endpoint
# (the AspectRatioV4 enum, excluding AUTO). The API rejects arbitrary ratios, so
# the requested width/height is mapped to the closest supported bucket to guide
# bbox composition. The engine generates at the exact requested dimensions
# regardless (the caption's aspect_ratio is dropped before generation).
_ASPECT_RATIO_BUCKETS: tuple[tuple[int, int], ...] = (
    (1, 4),
    (1, 3),
    (1, 2),
    (9, 16),
    (10, 16),
    (2, 3),
    (3, 4),
    (4, 5),
    (1, 1),
    (5, 4),
    (4, 3),
    (3, 2),
    (16, 10),
    (16, 9),
    (2, 1),
    (3, 1),
    (4, 1),
)


def _match_aspect_ratio(width: int, height: int) -> str:
    """Return the closest supported `WxH` bucket for the given canvas."""
    target = math.log(width / height)
    bw, bh = min(_ASPECT_RATIO_BUCKETS, key=lambda wh: abs(math.log(wh[0] / wh[1]) - target))
    return f"{bw}x{bh}"


class IdeogramMagicPromptProvider:
    """Calls Ideogram's hosted magic-prompt API and returns its JSON caption.

    `model` is accepted for interface uniformity with the other providers but is
    ignored — the magic-prompt endpoint has no model selector (it is always v4).
    `target_elements` is likewise a no-op: the hosted API has no element-count
    knob. Use a chat provider (codex/openai/…) when you need element-count control.
    """

    def __init__(
        self,
        model: str = "v4",
        api_key: str = "",
        timeout_s: int = 120,
        transport: Callable[..., Any] = urllib.request.urlopen,
    ):
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._transport = transport

    def expand(
        self, prompt: str, *, width: int, height: int, target_elements: int = 0
    ) -> dict[str, Any]:
        api_ratio = _match_aspect_ratio(width, height)
        data = json.dumps({"text_prompt": prompt, "aspect_ratio": api_ratio}).encode()
        req = urllib.request.Request(
            IDEOGRAM_MAGIC_PROMPT_URL,
            data=data,
            headers={"Api-Key": self.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        payload = self._post(req)
        if not isinstance(payload, dict) or "json_prompt" not in payload:
            raise ValueError("ideogram magic-prompt returned no json_prompt")
        caption = payload["json_prompt"]
        if not isinstance(caption, dict):
            raise ValueError("ideogram magic-prompt json_prompt was not a JSON object")
        # Stamp our computed W:H ratio (matches the chat providers' finalize_caption);
        # the API's WxH bucket only guides bbox composition.
        caption["aspect_ratio"] = aspect_ratio(width, height)
        return caption

    def _post(self, req: urllib.request.Request) -> Any:
        try:
            with self._transport(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise RuntimeError("invalid API key for ideogram") from exc
            if exc.code == 429:
                raise RuntimeError("rate-limited by ideogram; retry shortly") from exc
            detail = exc.read().decode(errors="replace")[:300] if hasattr(exc, "read") else ""
            raise RuntimeError(f"ideogram HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"network error calling ideogram: {exc.reason}") from exc
