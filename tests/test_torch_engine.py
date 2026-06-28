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


def test_torch_generate_owns_loop_and_reports_progress(monkeypatch):
    import sys, types
    from types import SimpleNamespace
    import torch
    from imgen.engine.torch_engine import TorchEngine

    N = 3                                   # num_steps
    BATCH, NIMG, LATENT, MAXTXT = 1, 4, 2, 3

    # Fake `ideogram4` package (generate() does `from ideogram4 import PRESETS`).
    ideg = types.ModuleType("ideogram4")
    ideg.PRESETS = {"V4_TURBO_12": SimpleNamespace(
        num_steps=N, guidance_schedule=[1.0] * N, mu=0.5, std=1.0)}
    monkeypatch.setitem(sys.modules, "ideogram4", ideg)

    # Fake `ideogram4.scheduler` helpers (identity schedule; intervals length N+1).
    sched = types.ModuleType("ideogram4.scheduler")
    sched.get_schedule_for_resolution = lambda hw, *, known_mean, std: (lambda x: x)
    sched.make_step_intervals = lambda n: torch.arange(n + 1, dtype=torch.float32)
    monkeypatch.setitem(sys.modules, "ideogram4.scheduler", sched)

    calls = {"cond": 0, "uncond": 0, "decoded": False}

    class _Transformer:                     # callable + exposes .config.in_channels
        config = SimpleNamespace(in_channels=LATENT)
        def __call__(self, **kw):
            calls["cond"] += 1
            return torch.zeros_like(kw["x"])            # [B, MAXTXT+NIMG, LATENT]

    def _uncond(**kw):
        calls["uncond"] += 1
        return torch.zeros_like(kw["x"])               # [B, NIMG, LATENT]

    def _build_inputs(prompts, *, height, width):
        return {
            "token_ids": torch.zeros(BATCH, MAXTXT + NIMG, dtype=torch.long),
            "text_position_ids": torch.zeros(BATCH, MAXTXT + NIMG, 3, dtype=torch.long),
            "position_ids": torch.zeros(BATCH, MAXTXT + NIMG, 3, dtype=torch.long),
            "segment_ids": torch.zeros(BATCH, MAXTXT + NIMG, dtype=torch.long),
            "indicator": torch.zeros(BATCH, MAXTXT + NIMG, dtype=torch.long),
            "num_image_tokens": NIMG, "grid_h": 2, "grid_w": 2, "max_text_tokens": MAXTXT,
        }

    class _FakeImg:
        size = (64, 64)

    def _decode(z, *, grid_h, grid_w):
        calls["decoded"] = True
        return [_FakeImg()]

    pipe = SimpleNamespace(
        device="cpu",
        conditional_transformer=_Transformer(),
        unconditional_transformer=_uncond,
        _verify_prompts=lambda prompts, *, raise_on_issues: None,
        _build_inputs=_build_inputs,
        _encode_text=lambda tok, pos, ind: torch.zeros(BATCH, MAXTXT + NIMG, LATENT),
        _decode=_decode,
    )

    eng = object.__new__(TorchEngine)       # skip __init__/model load
    eng._pipe = pipe

    cap = {"high_level_description": "x", "style_description": {},
           "compositional_deconstruction": {"background": "", "elements": []}}

    seen = []
    res = eng.generate(cap, width=64, height=64, preset="V4_TURBO_12", seed=0,
                       progress=lambda d, t: seen.append((d, t)))
    assert seen == [(1, 3), (2, 3), (3, 3)]            # 1-based, num_steps times
    assert calls["cond"] == N and calls["uncond"] == N
    assert calls["decoded"] and res.image is not None and res.backend == "torch" and res.seed == 0

    seen2 = []                                          # progress=None fires nothing
    eng.generate(cap, width=64, height=64, preset="V4_TURBO_12", seed=0, progress=None)
    assert seen2 == []


@pytest.mark.integration
@pytest.mark.skipif(
    not _cuda_available() or MODEL_DIR is None,
    reason="needs CUDA + a configured torch weights dir",
)
def test_torch_owned_loop_matches_pipeline_call():
    import numpy as np
    from imgen.engine.factory import create_pipeline
    from imgen.engine.torch_engine import build_caption_prompt
    from imgen.models.ideogram4 import PRESETS as MODEL_PRESETS

    assert MODEL_DIR is not None
    engine = create_pipeline(ModelSpec(name="ideogram4", path=MODEL_DIR, backend=Backend.TORCH))
    preset, seed = "V4_TURBO_12", 1234

    # ours (the owned loop)
    ours = engine.generate(CAPTION, width=512, height=512, preset=preset, seed=seed)

    # theirs (the sealed __call__) — same args our generate uses
    params = __import__("ideogram4").PRESETS[preset]
    _, prompt = build_caption_prompt(CAPTION)
    theirs_img = engine._pipe(
        prompt, height=512, width=512, num_steps=params.num_steps,
        guidance_schedule=params.guidance_schedule, mu=params.mu, std=params.std,
        seed=seed, raise_on_caption_issues=False)[0]

    a = np.asarray(ours.image)
    b = np.asarray(theirs_img)
    assert a.shape == b.shape
    # Fixed seed + identical ops ⇒ bit-identical. If a GPU nondeterminism delta appears,
    # relax to PSNR >= 50 dB and record it in the task report.
    assert np.array_equal(a, b), "owned loop diverged from Ideogram4Pipeline.__call__"


