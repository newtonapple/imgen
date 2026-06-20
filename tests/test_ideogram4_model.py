# tests/test_ideogram4_model.py
from pathlib import Path

from imagegen import models
from imagegen.models import ideogram4 as ig4
from imagegen.platform import Backend


def _parse(args):
    with ig4.Ideogram4Model().model_options.make_context("ideogram4", list(args)) as ctx:
        return dict(ctx.params)


def test_registered_with_alias():
    assert models.get("ideogram4").name == "ideogram4"
    assert models.get("ig4").name == "ideogram4"


def test_model_options_defaults():
    p = _parse([])
    assert p["preset"] == "V4_DEFAULT_20"
    assert p["quantize"] is None
    assert p["magic_prompt_provider"] is None
    assert p["magic_model"] is None
    assert p["target_elements"] == 0
    assert p["caption"] is None


def test_model_options_parsing():
    p = _parse(["--mp", "openrouter", "--mm", "openrouter/free", "--preset", "V4_TURBO_12"])
    assert p["magic_prompt_provider"] == "openrouter"
    assert p["magic_model"] == "openrouter/free"
    assert p["preset"] == "V4_TURBO_12"


def test_set_flags_parse():
    p = _parse(["--set-mp", "openrouter", "--set-mm", "openrouter/free", "--set-mk", "sk-x"])
    assert p["set_magic_prompt_provider"] == "openrouter"
    assert p["set_magic_model"] == "openrouter/free"
    assert p["set_magic_prompt_api_key"] == "sk-x"


def test_build_pipeline_resolves_and_builds_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("IG_CONFIG_DIR", str(tmp_path))
    calls = {}

    def fake_create_pipeline(spec, backend=None, quantize=None):
        calls["spec_path"] = spec.path
        calls["backend"] = backend
        calls["quantize"] = quantize
        return "ENGINE"

    monkeypatch.setattr(ig4, "create_pipeline", fake_create_pipeline)
    monkeypatch.setattr(
        ig4,
        "resolve_magic_settings",
        lambda opts, *, config, secrets: ("openrouter", "openrouter/free"),
    )
    monkeypatch.setattr(
        ig4,
        "make_magic_provider",
        lambda provider, model, *, secrets: ("PROVIDER", provider, model),
    )
    monkeypatch.setattr(
        ig4, "Pipeline", lambda engine, magic_prompt: ("PIPE", engine, magic_prompt)
    )

    m = ig4.Ideogram4Model()
    pipe = m.build_pipeline(
        weights_path=Path("/w"),
        backend=Backend.MLX,
        quantize="8",
        magic_prompt_provider="openrouter",
        magic_model="openrouter/free",
    )
    assert pipe[0] == "PIPE" and pipe[1] == "ENGINE"  # type: ignore[index]
    assert pipe[2] == ("PROVIDER", "openrouter", "openrouter/free")  # type: ignore[index]
    assert calls["quantize"] == 8  # converted to int
    assert calls["backend"] == Backend.MLX


def test_run_one_prompt_path_calls_pipeline_run():
    class FakePipe:
        def run(self, prompt, *, width, height, preset, seed, target_elements):
            return ("run", prompt, preset, seed, target_elements)

        def generate(self, caption, *, width, height, preset, seed):
            raise AssertionError("should not be called")

    m = ig4.Ideogram4Model()
    r = m.run_one(
        FakePipe(),
        prompt="a cat",
        width=512,
        height=512,
        seed=7,
        preset="V4_TURBO_12",
        target_elements=3,
        caption=None,
        quantize=None,
        magic_model="x",
    )
    assert r == ("run", "a cat", "V4_TURBO_12", 7, 3)  # type: ignore[comparison-overlap]


def test_run_one_caption_path_calls_pipeline_generate(tmp_path):
    cap = tmp_path / "c.json"
    cap.write_text(
        '{"high_level_description":"x","style_description":{},"compositional_deconstruction":{}}'
    )

    class FakePipe:
        def run(self, *a, **k):
            raise AssertionError("should not be called")

        def generate(self, caption, *, width, height, preset, seed):
            return ("generate", caption["high_level_description"], preset, seed)

    m = ig4.Ideogram4Model()
    r = m.run_one(
        FakePipe(),
        prompt=None,
        width=512,
        height=512,
        seed=1,
        preset="V4_DEFAULT_20",
        target_elements=0,
        caption=str(cap),
        quantize=None,
        magic_model="x",
    )
    assert r == ("generate", "x", "V4_DEFAULT_20", 1)  # type: ignore[comparison-overlap]
