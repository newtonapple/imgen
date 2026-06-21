import json
import urllib.error
import urllib.request
from http.client import HTTPMessage
from typing import Any

import pytest

from imgen.magic_prompt.ideogram_provider import (
    IDEOGRAM_API_KEY_ENV,
    IDEOGRAM_MAGIC_PROMPT_URL,
    IdeogramMagicPromptProvider,
)

CAPTION = {
    "high_level_description": "a cat",
    "style_description": {"aesthetics": "photo"},
    "compositional_deconstruction": {"background": "plain", "elements": []},
}


class _FakeResp:
    def __init__(self, body: str):
        self._body = body.encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _transport_for(captured, body: str):
    def transport(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = req
        captured["body"] = json.loads(req.data.decode())
        return _FakeResp(body)

    return transport


def test_request_shape_and_extract():
    captured: dict[str, Any] = {}
    body = json.dumps({"json_prompt": CAPTION})
    p = IdeogramMagicPromptProvider("v4", "ig-key", transport=_transport_for(captured, body))
    cap = p.expand("a cat", width=768, height=768)
    assert cap["high_level_description"] == "a cat"
    # the caption's aspect_ratio is stamped in W:H form (matches chat providers)
    assert cap["aspect_ratio"] == "1:1"
    assert captured["url"] == IDEOGRAM_MAGIC_PROMPT_URL
    assert captured["headers"].get_header("Api-key") == "ig-key"
    assert captured["headers"].get_header("Content-type") == "application/json"
    # the API request uses the WxH enum form
    assert captured["body"] == {"text_prompt": "a cat", "aspect_ratio": "1x1"}


def test_aspect_ratio_maps_to_closest_bucket():
    captured: dict[str, Any] = {}
    body = json.dumps({"json_prompt": CAPTION})
    p = IdeogramMagicPromptProvider("v4", "ig-key", transport=_transport_for(captured, body))
    # exact bucket ratios map directly
    p.expand("a cat", width=1536, height=768)
    assert captured["body"]["aspect_ratio"] == "2x1"
    p.expand("a cat", width=768, height=1536)
    assert captured["body"]["aspect_ratio"] == "1x2"
    # non-bucket dimensions snap to the closest supported bucket
    p.expand("a cat", width=1000, height=700)  # ~1.43 -> 3x2 (1.5)
    assert captured["body"]["aspect_ratio"] == "3x2"
    p.expand("a cat", width=700, height=1000)  # ~0.70 -> 2x3 (0.67)
    assert captured["body"]["aspect_ratio"] == "2x3"


def test_missing_json_prompt_errors():
    p = IdeogramMagicPromptProvider(
        "v4", "ig-key", transport=_transport_for({}, json.dumps({"unrelated": 1}))
    )
    with pytest.raises(ValueError, match="json_prompt"):
        p.expand("a cat", width=512, height=512)


def test_http_401_raises_invalid_key():
    def transport(req: urllib.request.Request, timeout: int) -> _FakeResp:
        hdrs = HTTPMessage()
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", hdrs, None)

    p = IdeogramMagicPromptProvider("v4", "bad", transport=transport)
    with pytest.raises(RuntimeError, match="invalid API key"):
        p.expand("a cat", width=512, height=512)


def test_http_429_raises_rate_limited():
    def transport(req: urllib.request.Request, timeout: int) -> _FakeResp:
        hdrs = HTTPMessage()
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many", hdrs, None)

    p = IdeogramMagicPromptProvider("v4", "ig-key", transport=transport)
    with pytest.raises(RuntimeError, match="rate-limited"):
        p.expand("a cat", width=512, height=512)


def test_env_var_constant_matches_repo_convention():
    assert IDEOGRAM_API_KEY_ENV == "IDEOGRAM_API_KEY"


def _resolve_ideogram_key() -> str | None:
    import os

    from imgen.config import Secrets

    return os.environ.get(IDEOGRAM_API_KEY_ENV) or Secrets.load().api_key("ideogram")


@pytest.mark.integration
def test_ideogram_magic_prompt_live():
    """Calls Ideogram's hosted magic-prompt API (real network).

    Run with: pytest tests/test_ideogram_provider.py::test_ideogram_magic_prompt_live -m integration
    (or `make test-integration`). Skips if no key is configured (set IDEOGRAM_API_KEY
    or run `ig ideogram4 config set-key ideogram <key>`). The expansion itself is free.
    """
    key = _resolve_ideogram_key()
    if not key:
        pytest.skip(
            f"no ideogram API key found; set {IDEOGRAM_API_KEY_ENV} or run "
            f"`ig ideogram4 config set-key ideogram <key>` to run this integration test"
        )
    provider = IdeogramMagicPromptProvider("v4", api_key=key)
    caption = provider.expand("a ginger cat in a wizard hat", width=768, height=768)
    assert isinstance(caption, dict)
    assert "compositional_deconstruction" in caption
    # our W:H stamp is applied regardless of the API's WxH bucket
    assert caption["aspect_ratio"] == "1:1"
