# tests/test_ideogram4_model.py
from pathlib import Path

import click

from imagegen import models
from imagegen.config import Config, Secrets
from imagegen.models import ideogram4 as ig4
from imagegen.platform import Backend


def test_registered_with_alias():
    assert models.get("ideogram4").name == "ideogram4"
    assert models.get("ig4").name == "ideogram4"


def test_gen_options_include_dimensions_and_preset():
    names = {p.name for p in ig4.Ideogram4Model().gen_options if isinstance(p, click.Option)}
    assert {"prompt", "width", "height", "seed", "preset", "target_elements", "caption"} <= names
    # build flags are NOT gen options
    assert "quantize" not in names and "magic_prompt_provider" not in names


def test_config_keys_present():
    keys = set(ig4.Ideogram4Model().config_keys)
    assert {"weights-path", "quantize", "backend", "magic-provider", "magic-model"} == keys


def test_build_pipeline_reads_quantize_from_config(monkeypatch):
    calls = {}

    def fake_create_pipeline(spec, backend=None, quantize=None):
        calls["quantize"] = quantize
        calls["backend"] = backend
        return "ENGINE"

    monkeypatch.setattr(ig4, "create_pipeline", fake_create_pipeline)
    monkeypatch.setattr(ig4, "resolve_magic_provider", lambda config: ("codex", "gpt-5.5"))
    monkeypatch.setattr(ig4, "make_magic_provider", lambda provider, model, *, secrets: "PROVIDER")
    monkeypatch.setattr(
        ig4, "Pipeline", lambda engine, magic_prompt: ("PIPE", engine, magic_prompt)
    )

    cfg = Config({"models": {"ideogram4": {"quantize": "8"}}})
    pipe = ig4.Ideogram4Model().build_pipeline(
        weights_path=Path("/w"), backend=Backend.MLX, config=cfg, secrets=Secrets({})
    )
    assert pipe == ("PIPE", "ENGINE", "PROVIDER")  # type: ignore[comparison-overlap]
    assert calls["quantize"] == 8
    assert calls["backend"] == Backend.MLX


def test_run_one_prompt_path():
    class FakePipe:
        def run(self, prompt, *, width, height, preset, seed, target_elements):
            return ("run", prompt, preset, seed, target_elements)

        def generate(self, *a, **k):
            raise AssertionError("should not be called")

    r = ig4.Ideogram4Model().run_one(
        FakePipe(),
        prompt="a cat",
        width=512,
        height=512,
        seed=7,
        preset="V4_TURBO_12",
        target_elements=3,
        caption=None,
    )
    assert r == ("run", "a cat", "V4_TURBO_12", 7, 3)  # type: ignore[comparison-overlap]


def test_run_one_caption_path(tmp_path):
    cap = tmp_path / "c.json"
    cap.write_text('{"high_level_description":"x"}')

    class FakePipe:
        def run(self, *a, **k):
            raise AssertionError("should not be called")

        def generate(self, caption, *, width, height, preset, seed):
            return ("generate", caption["high_level_description"], preset, seed)

    r = ig4.Ideogram4Model().run_one(
        FakePipe(),
        prompt=None,
        width=512,
        height=512,
        seed=1,
        preset="V4_DEFAULT_20",
        target_elements=0,
        caption=str(cap),
    )
    assert r == ("generate", "x", "V4_DEFAULT_20", 1)  # type: ignore[comparison-overlap]
