# tests/test_cli_provider.py
import json
from types import SimpleNamespace
from imagegen.magic_prompt.cli_provider import CliMagicPromptProvider

CAPTION = {"high_level_description": "a cat",
           "style_description": {"aesthetics": "warm"},
           "compositional_deconstruction": {"background": "room", "elements": []}}

def fake_runner_factory(stdout, returncode=0):
    def run(cmd, capture_output, text, timeout, check):
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")
    return run

def test_expand_parses_fenced_json_and_injects_aspect_ratio():
    out = "```json\n" + json.dumps(CAPTION) + "\n```"
    p = CliMagicPromptProvider(runner=fake_runner_factory(out))
    # 576x1024 → gcd=64 → 9:16 (brief had 832x1472 which is 13:23, not 9:16)
    result = p.expand("a cat", width=576, height=1024)
    assert result["high_level_description"] == "a cat"
    assert result["aspect_ratio"] == "9:16"

def test_expand_includes_target_elements_directive():
    captured = {}
    def run(cmd, capture_output, text, timeout, check):
        captured["prompt"] = cmd[-1]
        return SimpleNamespace(returncode=0, stdout=json.dumps(CAPTION), stderr="")
    p = CliMagicPromptProvider(runner=run)
    p.expand("a cat", width=1024, height=1024, target_elements=5)
    assert "5" in captured["prompt"] and "elements" in captured["prompt"].lower()
