"""Unit + integration tests for the torch (ideogram4 package) backend.

Unit tests are torch-free: they target the pure `build_caption_prompt` helper and
`PRESET_NAMES`, importable on the Mac (the module's torch/ideogram4 imports are
lazy, inside __init__/generate). The integration tests import torch/ideogram4
lazily and skip without CUDA + weights.
"""

import json
from pathlib import Path

import pytest

from imgen.caption import model_caption
from imgen.config import Config, ModelSpec
from imgen.engine.torch_engine import PRESET_NAMES, build_caption_prompt
from imgen.platform import Backend

CAPTION = {
    "high_level_description": "a red cube on a white studio table",
    "style_description": {"aesthetics": "clean studio product shot"},
    "compositional_deconstruction": {"background": "white", "elements": []},
}


def test_preset_names_match_model_choices():
    # Must match models/ideogram4.py:PRESETS (the CLI --preset choices) and the
    # keys the ideogram4 package exposes.
    from imgen.models.ideogram4 import PRESETS as MODEL_PRESETS

    assert list(PRESET_NAMES) == MODEL_PRESETS


def test_build_caption_prompt_is_model_caption_json():
    caption_dict, prompt = build_caption_prompt(CAPTION)
    assert caption_dict == CAPTION
    assert prompt == json.dumps(model_caption(CAPTION), ensure_ascii=False)


def test_build_caption_prompt_accepts_json_string():
    caption_dict, prompt = build_caption_prompt(json.dumps(CAPTION))
    assert caption_dict == CAPTION
    assert prompt == json.dumps(model_caption(CAPTION), ensure_ascii=False)


def test_build_caption_prompt_keeps_non_ascii_literal():
    # Non-Latin text (e.g. Japanese) must stay literal UTF-8, not \uXXXX escapes,
    # so the package's caption verifier doesn't warn and the model tokenizes it well.
    cap = {**CAPTION, "high_level_description": "ネオン東京の夜"}
    _, prompt = build_caption_prompt(cap)
    assert "ネオン東京の夜" in prompt
    assert "\\u" not in prompt


def test_build_caption_prompt_drops_non_schema_keys():
    # model_caption keeps only schema root keys (e.g. drops aspect_ratio).
    cap = {**CAPTION, "aspect_ratio": "1x1"}
    _, prompt = build_caption_prompt(cap)
    assert "aspect_ratio" not in json.loads(prompt)


def _torch_model_dir() -> Path | None:
    """Resolve the torch weights dir host-agnostically (config/env); None if unset/missing."""
    from imgen.config import resolve_weights_path

    try:
        path = resolve_weights_path("ideogram4", None, Config.load())
    except RuntimeError:
        return None
    return path if path.exists() else None


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


MODEL_DIR = _torch_model_dir()


@pytest.mark.integration
@pytest.mark.skipif(
    not _cuda_available() or MODEL_DIR is None,
    reason="needs CUDA + a configured torch weights dir (nf4)",
)
def test_torch_engine_generates_image():
    from imgen.engine.factory import create_pipeline

    assert MODEL_DIR is not None
    engine = create_pipeline(ModelSpec(name="ideogram4", path=MODEL_DIR, backend=Backend.TORCH))
    result = engine.generate(CAPTION, width=512, height=512, preset="V4_TURBO_12", seed=0)
    assert result.backend == "torch"
    assert result.seed == 0
    assert result.image.size == (512, 512)


@pytest.mark.integration
@pytest.mark.skipif(not _cuda_available(), reason="torch backend needs CUDA")
def test_torch_engine_clear_error_on_wrong_layout(tmp_path):
    from imgen.engine.torch_engine import TorchEngine

    with pytest.raises(RuntimeError, match="Ideogram-4 snapshot dir"):
        TorchEngine(ModelSpec(name="bad", path=tmp_path, backend=Backend.TORCH))


def test_torch_generate_forwards_per_step_progress(monkeypatch):
    import sys, types
    from imgen.engine.torch_engine import TorchEngine

    # Fake the `ideogram4` package: generate() does `from ideogram4 import PRESETS`.
    fake_pkg = types.ModuleType("ideogram4")
    class _P:  # preset params object with the attrs generate() reads
        num_steps = 4
        guidance_schedule = None
        mu = 0.0
        std = 1.0
    fake_pkg.PRESETS = {"V4_TURBO_12": _P()}
    monkeypatch.setitem(sys.modules, "ideogram4", fake_pkg)

    eng = object.__new__(TorchEngine)  # skip __init__ (no model load)
    eng.device = "cpu"

    class _FakeImg:
        size = (64, 64)
    def fake_pipe(prompt, *, height, width, num_steps, guidance_schedule, mu, std, seed,
                  raise_on_caption_issues, callback=None):
        if callback is not None:
            for k in range(1, num_steps + 1):
                callback(k, num_steps)
        return [_FakeImg()]
    eng._pipe = fake_pipe

    seen = []
    eng.generate(
        {"high_level_description": "x", "style_description": {},
         "compositional_deconstruction": {"background": "", "elements": []}},
        width=64, height=64, preset="V4_TURBO_12", seed=0,
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]
