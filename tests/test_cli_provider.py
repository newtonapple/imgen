# tests/test_cli_provider.py
import json
from types import SimpleNamespace
from imagegen.magic_prompt.cli_provider import CliMagicPromptProvider

CAPTION = {
    "high_level_description": "a cat",
    "style_description": {"aesthetics": "warm"},
    "compositional_deconstruction": {"background": "room", "elements": []},
}


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


# ---------------------------------------------------------------------------
# available_models() tests (TDD: RED first, then GREEN after implementation)
# ---------------------------------------------------------------------------


def test_available_models_codex_only_when_no_pi_file(tmp_path, monkeypatch):
    """With PI_MODELS_JSON pointing at a nonexistent path, only codex entries returned."""
    monkeypatch.setenv("PI_MODELS_JSON", str(tmp_path / "nonexistent.json"))
    models = CliMagicPromptProvider.available_models()
    assert models == ["codex - gpt-5.5", "codex - gpt-5.4", "codex - gpt-5.4-mini"]


def test_available_models_includes_pi_text_models(tmp_path, monkeypatch):
    """PI_MODELS_JSON with text + image models: only text ones appear, image-only excluded."""
    pi_data = {
        "providers": {
            "openrouter": {
                "models": [
                    {"id": "x/y", "input": ["text"]},
                    {"id": "vision-only", "input": ["image"]},
                ]
            }
        }
    }
    pi_file = tmp_path / "models.json"
    pi_file.write_text(json.dumps(pi_data))
    monkeypatch.setenv("PI_MODELS_JSON", str(pi_file))

    models = CliMagicPromptProvider.available_models()
    assert "codex - gpt-5.5" in models
    assert "codex - gpt-5.4" in models
    assert "codex - gpt-5.4-mini" in models
    assert "pi - openrouter - x/y" in models
    assert "pi - openrouter - vision-only" not in models


def test_available_models_skips_model_without_id(tmp_path, monkeypatch):
    """Models without an 'id' field are silently skipped."""
    pi_data = {
        "providers": {
            "local": {
                "models": [
                    {"name": "no-id-here", "input": ["text"]},
                    {"id": "good-model"},  # no input key → should include (not filtered by input)
                ]
            }
        }
    }
    pi_file = tmp_path / "models.json"
    pi_file.write_text(json.dumps(pi_data))
    monkeypatch.setenv("PI_MODELS_JSON", str(pi_file))

    models = CliMagicPromptProvider.available_models()
    # model without id is skipped
    assert not any("no-id-here" in m for m in models)
    # model with id but no input key is included (input is None → condition skipped)
    assert "pi - local - good-model" in models


def test_available_models_handles_corrupt_json(tmp_path, monkeypatch):
    """Corrupt JSON file falls back to codex-only list."""
    pi_file = tmp_path / "models.json"
    pi_file.write_text("this is not json{{{{")
    monkeypatch.setenv("PI_MODELS_JSON", str(pi_file))

    models = CliMagicPromptProvider.available_models()
    assert models == ["codex - gpt-5.5", "codex - gpt-5.4", "codex - gpt-5.4-mini"]


def _capture_cmd(stdout):
    captured = {}

    def run(cmd, capture_output, text, timeout, check):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    return captured, run


def test_codex_argv_unchanged():
    captured, run = _capture_cmd(json.dumps(CAPTION))
    CliMagicPromptProvider(provider="codex", model="gpt-5.5", runner=run).expand(
        "a cat", width=512, height=512
    )
    assert captured["cmd"][:5] == [captured["cmd"][0], "exec", "--model", "gpt-5.5", "--sandbox"]


def test_claude_argv():
    captured, run = _capture_cmd(json.dumps(CAPTION))
    CliMagicPromptProvider(provider="claude", model="sonnet", runner=run).expand(
        "a cat", width=512, height=512
    )
    cmd = captured["cmd"]
    assert cmd[1] == "-p" and cmd[-2:] == ["--model", "sonnet"]


def test_pi_argv_splits_provider_and_model_on_first_slash():
    captured, run = _capture_cmd(json.dumps(CAPTION))
    CliMagicPromptProvider(provider="pi", model="openrouter/some/model", runner=run).expand(
        "a cat", width=512, height=512
    )
    cmd = captured["cmd"]
    assert cmd[1:5] == ["--provider", "openrouter", "--model", "some/model"]
