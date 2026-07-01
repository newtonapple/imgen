# tests/test_actions.py
"""Tests for _build_request format threading."""

from imgen.cli.actions import _build_request


def test_build_request_threads_format_from_flag():
    req = _build_request("out.png", {"width": 512, "height": 512, "format": "webp"})
    assert req["format"] == "webp"
    assert req["output_path"] == "out.png"


def test_build_request_infers_format_from_extension():
    req = _build_request("out.webp", {"width": 512, "height": 512})
    assert req["format"] == "webp"


def test_build_request_defaults_png():
    req = _build_request("out.png", {"width": 512, "height": 512})
    assert req["format"] == "png"


def test_build_request_format_on_caption_path():
    # caption path (op=generate) must also carry format
    import json, tempfile, os
    fd, capf = tempfile.mkstemp(suffix=".json")
    os.write(fd, json.dumps({"elements": []}).encode())
    os.close(fd)
    req = _build_request("out.webp", {"width": 512, "height": 512, "caption": capf})
    assert req["op"] == "generate"
    assert req["format"] == "webp"
