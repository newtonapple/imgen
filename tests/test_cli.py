# tests/test_cli.py
import json
from imagegen.cli import main


def test_platform_cmd(capsys):
    assert main(["platform"]) == 0
    assert "default_backend" in capsys.readouterr().out


def test_magic_prompt_cmd_writes_json(tmp_path, monkeypatch):
    out = tmp_path / "cap.json"
    import imagegen.cli as cli

    monkeypatch.setattr(
        cli,
        "_build_provider",
        lambda model: type(
            "P",
            (),
            {
                "expand": lambda self, prompt, *, width, height, target_elements=0: {
                    "high_level_description": prompt,
                    "style_description": {},
                    "compositional_deconstruction": {},
                    "aspect_ratio": "1:1",
                }
            },
        )(),
    )
    rc = main(
        ["magic-prompt", "a cat", "--width", "1024", "--height", "1024", "--out", str(out)]
    )
    assert rc == 0 and json.loads(out.read_text())["high_level_description"] == "a cat"


def test_magic_prompt_cmd_default_out(tmp_path, monkeypatch, capsys):
    """magic-prompt without --out prints JSON to stdout."""
    import imagegen.cli as cli

    monkeypatch.setattr(
        cli,
        "_build_provider",
        lambda model: type(
            "P",
            (),
            {
                "expand": lambda self, prompt, *, width, height, target_elements=0: {
                    "high_level_description": prompt,
                    "style_description": {},
                    "compositional_deconstruction": {},
                    "aspect_ratio": "1:1",
                }
            },
        )(),
    )
    rc = main(["magic-prompt", "a dog", "--width", "512", "--height", "512"])
    assert rc == 0
    out = capsys.readouterr().out
    assert json.loads(out)["high_level_description"] == "a dog"


def test_generate_cmd(tmp_path, monkeypatch):
    """generate: reads caption JSON, calls engine, saves image."""
    import imagegen.cli as cli
    from imagegen.engine.base import GenerationResult

    cap = {
        "high_level_description": "test",
        "style_description": {},
        "compositional_deconstruction": {},
    }
    cap_file = tmp_path / "cap.json"
    cap_file.write_text(json.dumps(cap))

    class FakeImage:
        def save(self, path):
            import pathlib
            pathlib.Path(path).write_bytes(b"fake-image")

    class FakeEngine:
        backend = "fake"
        calls = []

        def generate(self, caption, *, width, height, preset="V4_DEFAULT_20", seed=None):
            self.calls.append((caption, seed))
            return GenerationResult(
                image=FakeImage(),
                seed=seed or 99,
                width=width,
                height=height,
                preset=preset,
                caption=caption if isinstance(caption, dict) else {},
                backend=self.backend,
                duration_s=0.1,
            )

    fake_engine = FakeEngine()

    monkeypatch.setattr(cli, "_build_engine", lambda model_path, backend=None: fake_engine)

    out_img = tmp_path / "out.png"
    rc = main(
        [
            "generate",
            "--caption", str(cap_file),
            "--width", "1024",
            "--height", "1024",
            "--seed", "42",
            "--out", str(out_img),
        ]
    )
    assert rc == 0
    assert out_img.exists()
    assert fake_engine.calls[0][1] == 42


def test_generate_warns_without_seed(tmp_path, monkeypatch, capsys):
    """generate: warns on stderr when no seed is provided."""
    import imagegen.cli as cli
    from imagegen.engine.base import GenerationResult

    cap = {"high_level_description": "test", "style_description": {}, "compositional_deconstruction": {}}
    cap_file = tmp_path / "cap.json"
    cap_file.write_text(json.dumps(cap))

    class FakeImage:
        def save(self, path):
            import pathlib
            pathlib.Path(path).write_bytes(b"fake-image")

    class FakeEngine:
        backend = "fake"
        calls = []

        def generate(self, caption, *, width, height, preset="V4_DEFAULT_20", seed=None):
            self.calls.append((caption, seed))
            return GenerationResult(
                image=FakeImage(),
                seed=12345,
                width=width,
                height=height,
                preset=preset,
                caption=caption if isinstance(caption, dict) else {},
                backend=self.backend,
                duration_s=0.1,
            )

    monkeypatch.setattr(cli, "_build_engine", lambda model_path, backend=None: FakeEngine())

    out_img = tmp_path / "out.png"
    rc = main(
        [
            "generate",
            "--caption", str(cap_file),
            "--width", "512",
            "--height", "512",
            "--out", str(out_img),
        ]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "re-seeding will likely change the image substantially" in err


def test_run_cmd(tmp_path, monkeypatch):
    """run: calls magic-prompt then generate, saves image."""
    import imagegen.cli as cli
    from imagegen.engine.base import GenerationResult

    class FakeImage:
        def save(self, path):
            import pathlib
            pathlib.Path(path).write_bytes(b"fake-image")

    class FakeEngine:
        backend = "fake"
        calls = []

        def generate(self, caption, *, width, height, preset="V4_DEFAULT_20", seed=None):
            self.calls.append((caption, seed))
            return GenerationResult(
                image=FakeImage(),
                seed=seed or 55,
                width=width,
                height=height,
                preset=preset,
                caption=caption if isinstance(caption, dict) else {},
                backend=self.backend,
                duration_s=0.1,
            )

    monkeypatch.setattr(
        cli,
        "_build_provider",
        lambda model: type(
            "P",
            (),
            {
                "expand": lambda self, prompt, *, width, height, target_elements=0: {
                    "high_level_description": prompt,
                    "style_description": {},
                    "compositional_deconstruction": {},
                    "aspect_ratio": "1:1",
                }
            },
        )(),
    )
    monkeypatch.setattr(cli, "_build_engine", lambda model_path, backend=None: FakeEngine())

    out_img = tmp_path / "run_out.png"
    rc = main(
        [
            "run",
            "a cat on a red couch",
            "--width", "512",
            "--height", "512",
            "--seed", "77",
            "--out", str(out_img),
        ]
    )
    assert rc == 0
    assert out_img.exists()


# ---------------------------------------------------------------------------
# New tests for Task 9 (TDD: RED first, GREEN after implementation)
# ---------------------------------------------------------------------------

def test_run_cmd_magic_model_flag(tmp_path, monkeypatch):
    """run: --magic-model passes the model string to _build_provider."""
    import imagegen.cli as cli
    from imagegen.engine.base import GenerationResult

    recorded = {}

    class FakeImage:
        def save(self, path):
            import pathlib
            pathlib.Path(path).write_bytes(b"fake-image")

    class FakeEngine:
        backend = "fake"

        def generate(self, caption, *, width, height, preset="V4_DEFAULT_20", seed=None):
            return GenerationResult(
                image=FakeImage(), seed=seed or 1, width=width, height=height,
                preset=preset, caption={}, backend=self.backend, duration_s=0.0,
            )

    def recording_build_provider(model):
        recorded["model"] = model
        return type("P", (), {
            "expand": lambda self, prompt, *, width, height, target_elements=0: {
                "high_level_description": prompt, "style_description": {},
                "compositional_deconstruction": {}, "aspect_ratio": "1:1",
            }
        })()

    monkeypatch.setattr(cli, "_build_provider", recording_build_provider)
    monkeypatch.setattr(cli, "_build_engine", lambda model_path, backend=None: FakeEngine())

    out_img = tmp_path / "out.png"
    rc = main([
        "run", "a cat", "--magic-model", "pi - openrouter - x/y",
        "--seed", "1", "--out", str(out_img),
    ])
    assert rc == 0
    assert recorded["model"] == "pi - openrouter - x/y"


def test_magic_models_subcommand(tmp_path, monkeypatch, capsys):
    """magic-models: prints available models one per line; codex entries always present."""
    import json as _json
    pi_data = {
        "providers": {
            "openrouter": {
                "models": [{"id": "x/y", "input": ["text"]}]
            }
        }
    }
    pi_file = tmp_path / "models.json"
    pi_file.write_text(_json.dumps(pi_data))
    monkeypatch.setenv("PI_MODELS_JSON", str(pi_file))

    rc = main(["magic-models"])
    assert rc == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert "codex - gpt-5.5" in lines
    assert "pi - openrouter - x/y" in lines  # noqa: E741


def test_run_cmd_backend_flag(tmp_path, monkeypatch):
    """run: --backend mlx is accepted and forwarded to _build_engine."""
    import imagegen.cli as cli
    from imagegen.engine.base import GenerationResult

    recorded = {}

    class FakeImage:
        def save(self, path):
            import pathlib
            pathlib.Path(path).write_bytes(b"fake-image")

    class FakeEngine:
        backend = "mlx"

        def generate(self, caption, *, width, height, preset="V4_DEFAULT_20", seed=None):
            return GenerationResult(
                image=FakeImage(), seed=1, width=width, height=height,
                preset=preset, caption={}, backend=self.backend, duration_s=0.0,
            )

    def recording_build_engine(model_path, backend=None):
        recorded["backend"] = backend
        return FakeEngine()

    monkeypatch.setattr(
        cli, "_build_provider",
        lambda model: type("P", (), {
            "expand": lambda self, prompt, *, width, height, target_elements=0: {
                "high_level_description": prompt, "style_description": {},
                "compositional_deconstruction": {}, "aspect_ratio": "1:1",
            }
        })(),
    )
    monkeypatch.setattr(cli, "_build_engine", recording_build_engine)

    out_img = tmp_path / "out.png"
    rc = main([
        "run", "a cat", "--backend", "mlx", "--seed", "1", "--out", str(out_img),
    ])
    assert rc == 0
    assert recorded["backend"] == "mlx"


def test_generate_cmd_backend_flag(tmp_path, monkeypatch):
    """generate: --backend torch is accepted and forwarded to _build_engine."""
    import imagegen.cli as cli
    from imagegen.engine.base import GenerationResult

    recorded = {}

    class FakeImage:
        def save(self, path):
            import pathlib
            pathlib.Path(path).write_bytes(b"fake-image")

    class FakeEngine:
        backend = "torch"

        def generate(self, caption, *, width, height, preset="V4_DEFAULT_20", seed=None):
            return GenerationResult(
                image=FakeImage(), seed=1, width=width, height=height,
                preset=preset, caption={}, backend=self.backend, duration_s=0.0,
            )

    def recording_build_engine(model_path, backend=None):
        recorded["backend"] = backend
        return FakeEngine()

    monkeypatch.setattr(cli, "_build_engine", recording_build_engine)

    cap = {"high_level_description": "test", "style_description": {}, "compositional_deconstruction": {}}
    cap_file = tmp_path / "cap.json"
    cap_file.write_text(__import__("json").dumps(cap))
    out_img = tmp_path / "out.png"

    rc = main([
        "generate", "--caption", str(cap_file), "--backend", "torch",
        "--seed", "1", "--out", str(out_img),
    ])
    assert rc == 0
    assert recorded["backend"] == "torch"
