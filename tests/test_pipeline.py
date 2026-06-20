# tests/test_pipeline.py
from imgen.pipeline import Pipeline
from imgen.engine.base import GenerationResult


class FakeEngine:
    backend = "fake"

    def __init__(self):
        self.calls = []

    def generate(self, caption, *, width, height, preset="V4_DEFAULT_20", seed=None):
        self.calls.append((caption, seed))
        return GenerationResult(
            image=object(),  # type: ignore[arg-type]
            seed=seed or 7,
            width=width,
            height=height,
            preset=preset,
            caption=caption if isinstance(caption, dict) else {},
            backend=self.backend,
            duration_s=0.0,
        )


class FakeProvider:
    def expand(self, prompt, *, width, height, target_elements=0):
        return {
            "high_level_description": prompt,
            "style_description": {},
            "compositional_deconstruction": {},
            "aspect_ratio": "1:1",
        }


def test_run_calls_magic_then_generate():
    eng = FakeEngine()
    p = Pipeline(engine=eng, magic_prompt=FakeProvider())
    r = p.run("a cat", width=1024, height=1024, preset="V4_DEFAULT_20", seed=42)
    assert r.caption["high_level_description"] == "a cat"
    assert eng.calls[0][1] == 42


def test_generate_passes_caption_through():
    eng = FakeEngine()
    p = Pipeline(engine=eng)
    cap = {
        "high_level_description": "x",
        "style_description": {},
        "compositional_deconstruction": {},
    }
    p.generate(cap, width=512, height=512, preset="V4_TURBO_12", seed=None)
    assert eng.calls[0][0] == cap


def test_magic_raises_without_provider():
    eng = FakeEngine()
    p = Pipeline(engine=eng)
    try:
        p.magic("foo", width=512, height=512)
        assert False, "should have raised"
    except RuntimeError as e:
        assert "MagicPromptProvider" in str(e)


def test_run_without_seed_still_runs():
    eng = FakeEngine()
    p = Pipeline(engine=eng, magic_prompt=FakeProvider())
    r = p.run("a dog", width=512, height=512, preset="V4_DEFAULT_20", seed=None)
    assert r.seed == 7  # FakeEngine returns 7 when seed is None


def test_pipeline_per_request_override_uses_factory():
    from imgen.pipeline import Pipeline

    class FakeProvider:
        def __init__(self, tag):
            self.tag = tag

        def expand(self, prompt, *, width, height, target_elements=0):
            return {"high_level_description": f"{self.tag}:{prompt}"}

    default = FakeProvider("default")
    built: list[tuple[str | None, str | None]] = []

    def factory(provider: str | None, model: str | None) -> FakeProvider:
        built.append((provider, model))
        return FakeProvider(f"{provider}/{model}")

    p = Pipeline(engine=FakeEngine(), magic_prompt=default, magic_factory=factory)
    assert (
        p.magic("x", width=64, height=64)["high_level_description"] == "default:x"
    )  # default path
    out = p.magic("x", width=64, height=64, magic_provider="openrouter", magic_model="m")
    assert out["high_level_description"] == "openrouter/m:x"
    assert built == [("openrouter", "m")]
