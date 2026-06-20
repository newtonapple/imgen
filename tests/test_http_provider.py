import json
import urllib.error
import urllib.request
from http.client import HTTPMessage
from typing import Any

import pytest

from imagegen.magic_prompt.http_provider import PROVIDERS, HttpMagicPromptProvider

CAPTION = {
    "high_level_description": "a cat",
    "style_description": {},
    "compositional_deconstruction": {},
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


def test_openai_request_shape_and_extract():
    captured: dict[str, Any] = {}
    body = json.dumps({"choices": [{"message": {"content": json.dumps(CAPTION)}}]})
    p = HttpMagicPromptProvider(
        PROVIDERS["openai"], "gpt-4o-mini", "sk-x", transport=_transport_for(captured, body)
    )
    cap = p.expand("a cat", width=512, height=512)
    assert cap["high_level_description"] == "a cat"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"].get_header("Authorization") == "Bearer sk-x"
    assert captured["body"]["model"] == "gpt-4o-mini"
    assert captured["body"]["response_format"] == {"type": "json_object"}


def test_anthropic_request_shape_and_extract():
    captured: dict[str, Any] = {}
    body = json.dumps({"content": [{"type": "text", "text": json.dumps(CAPTION)}]})
    p = HttpMagicPromptProvider(
        PROVIDERS["anthropic"], "claude-haiku-4-5", "sk-a", transport=_transport_for(captured, body)
    )
    cap = p.expand("a cat", width=512, height=512)
    assert cap["high_level_description"] == "a cat"
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"].get_header("X-api-key") == "sk-a"
    assert captured["headers"].get_header("Anthropic-version") == "2023-06-01"
    assert captured["body"]["max_tokens"] == 4096
    assert "response_format" not in captured["body"]


def test_openrouter_comma_separated_models_become_array():
    captured: dict[str, Any] = {}
    body = json.dumps({"choices": [{"message": {"content": json.dumps(CAPTION)}}]})
    p = HttpMagicPromptProvider(
        PROVIDERS["openrouter"], "a/x,b/y", "sk-o", transport=_transport_for(captured, body)
    )
    p.expand("a cat", width=512, height=512)
    assert captured["body"]["models"] == ["a/x", "b/y"]
    assert "model" not in captured["body"]


def test_openrouter_single_model_uses_model_field():
    captured: dict[str, Any] = {}
    body = json.dumps({"choices": [{"message": {"content": json.dumps(CAPTION)}}]})
    p = HttpMagicPromptProvider(
        PROVIDERS["openrouter"], "openrouter/free", "sk-o", transport=_transport_for(captured, body)
    )
    p.expand("a cat", width=512, height=512)
    assert captured["body"]["model"] == "openrouter/free"
    assert "models" not in captured["body"]


def test_http_401_raises_invalid_key():
    def transport(req: urllib.request.Request, timeout: int) -> _FakeResp:
        hdrs = HTTPMessage()
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", hdrs, None)

    p = HttpMagicPromptProvider(PROVIDERS["openai"], "gpt-4o-mini", "bad", transport=transport)
    with pytest.raises(RuntimeError, match="invalid API key"):
        p.expand("a cat", width=512, height=512)
