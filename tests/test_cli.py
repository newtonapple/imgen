# tests/test_cli.py
"""CLI tests for the model-first `ig <model> <action>` grammar."""

from __future__ import annotations

import json

from click.testing import CliRunner, Result

from imagegen.cli import ig


def run(args: list[str]) -> Result:
    return CliRunner().invoke(ig, args)


# ---------------------------------------------------------------------------
# platform + model group
# ---------------------------------------------------------------------------


def test_platform_cmd() -> None:
    r = run(["platform"])
    assert r.exit_code == 0
    assert "default_backend" in r.output


def test_model_list_includes_ideogram4() -> None:
    r = run(["model", "list"])
    assert r.exit_code == 0
    assert "ideogram4" in r.output


def test_model_ls_alias() -> None:
    assert run(["model", "ls"]).exit_code == 0


def test_model_show_outputs_info() -> None:
    r = run(["model", "show", "ideogram4"])
    assert r.exit_code == 0
    assert "ideogram4" in r.output
    assert "mlx" in r.output.lower() or "torch" in r.output.lower()


# ---------------------------------------------------------------------------
# ig ideogram4 gen — per-request opts routed through run_one
# ---------------------------------------------------------------------------


def test_gen_routes_request_opts(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("IG_CONFIG_DIR", str(tmp_path))
    import imagegen.cli.actions as actions
    from imagegen import models
    from imagegen.platform import Backend

    m = models.get("ideogram4")
    seen: dict[str, object] = {}

    class FakeImg:
        def save(self, p: str) -> None:
            open(p, "wb").close()

    class FakeResult:
        image = FakeImg()
        seed = 42
        width = 768
        height = 768
        preset = "V4_TURBO_12"
        backend = "mlx"
        duration_s = 1.0

    monkeypatch.setattr(
        m, "build_pipeline", lambda *, weights_path, backend, config, secrets: "PIPE"
    )
    monkeypatch.setattr(m, "run_one", lambda pipeline, **g: seen.update(g) or FakeResult())
    monkeypatch.setattr(m, "supported_backends", [Backend.MLX])
    monkeypatch.setattr(actions, "resolve_weights_path", lambda name, override, cfg: tmp_path / "w")

    out = tmp_path / "o.png"
    r = run(
        [
            "ideogram4",
            "gen",
            "-p",
            "a cat",
            "-w",
            "768",
            "--seed",
            "42",
            "-o",
            str(out),
            "--preset",
            "V4_TURBO_12",
        ]
    )
    assert r.exit_code == 0, r.output
    assert out.exists()
    assert seen["prompt"] == "a cat"
    assert seen["preset"] == "V4_TURBO_12"
    assert seen["width"] == 768


def test_gen_result_missing_preset_attribute(monkeypatch, tmp_path) -> None:
    """gen prints model-agnostic JSON even when the result lacks a .preset attribute."""
    monkeypatch.setenv("IG_CONFIG_DIR", str(tmp_path))
    import imagegen.cli.actions as actions
    from imagegen import models
    from imagegen.platform import Backend

    m = models.get("ideogram4")

    class FakeImg:
        def save(self, p: str) -> None:
            open(p, "wb").close()

    class FakeResultNoPreset:
        image = FakeImg()
        seed = 123
        width = 512
        height = 512
        backend = "mlx"
        duration_s = 2.5

    monkeypatch.setattr(
        m, "build_pipeline", lambda *, weights_path, backend, config, secrets: "PIPE"
    )
    monkeypatch.setattr(m, "run_one", lambda pipeline, **g: FakeResultNoPreset())
    monkeypatch.setattr(m, "supported_backends", [Backend.MLX])
    monkeypatch.setattr(actions, "resolve_weights_path", lambda name, override, cfg: tmp_path / "w")

    out = tmp_path / "t.png"
    r = run(
        [
            "ideogram4",
            "gen",
            "-p",
            "test",
            "-w",
            "512",
            "--seed",
            "123",
            "-o",
            str(out),
        ]
    )
    assert r.exit_code == 0, r.output
    assert out.exists()
    output_json = json.loads(r.output)
    assert output_json["preset"] is None
    assert output_json["seed"] == 123
    assert output_json["backend"] == "mlx"
    assert output_json["duration_s"] == 2.5
    assert output_json["out"] == str(out)


# ---------------------------------------------------------------------------
# ig ideogram4 config — set / set-key / set bogus
# ---------------------------------------------------------------------------


def test_config_set_no_out_required(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("IG_CONFIG_DIR", str(tmp_path))
    r = run(["ideogram4", "config", "set", "magic-provider", "openrouter"])
    assert r.exit_code == 0, r.output
    from imagegen.config import Config

    assert Config.load(tmp_path / "config.toml").magic_prompt_provider() == "openrouter"


def test_config_set_key_writes_secrets(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("IG_CONFIG_DIR", str(tmp_path))
    r = run(["ideogram4", "config", "set-key", "openrouter", "sk-x"])
    assert r.exit_code == 0, r.output
    from imagegen.config import Secrets

    s = Secrets.load(tmp_path / "secrets.toml")
    assert s.api_key("openrouter") == "sk-x"


def test_config_set_bogus_key_errors(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("IG_CONFIG_DIR", str(tmp_path))
    r = run(["ideogram4", "config", "set", "bogus", "x"])
    assert r.exit_code != 0
    assert "bogus" in r.output or "unknown" in r.output.lower()


def test_config_set_weights_path_persists(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("IG_CONFIG_DIR", str(tmp_path))
    r = run(["ideogram4", "config", "set", "weights-path", "/weights/ig4"])
    assert r.exit_code == 0, r.output
    from imagegen.config import Config

    assert Config.load(tmp_path / "config.toml").model_path("ideogram4") is not None


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


def test_gen_missing_out_clean_error() -> None:
    """Omitting -o/--out produces a clean 'Missing option' error (no traceback)."""
    r = run(["ideogram4", "gen", "-p", "x"])
    assert r.exit_code != 0
    assert "out" in r.output.lower() or "missing" in r.output.lower()
    assert r.exception is None or isinstance(r.exception, SystemExit)


def test_gen_unknown_model_clean_error(tmp_path) -> None:
    """ig nosuch gen … → Click 'No such command', not a traceback."""
    r = run(["nosuch", "gen", "-p", "x", "-o", str(tmp_path / "o.png")])
    assert r.exit_code != 0
    assert "No such command" in r.output
    assert r.exception is None or isinstance(r.exception, SystemExit)


def test_gen_backend_unsupported_error(monkeypatch, tmp_path) -> None:
    """Setting backend=torch via config then gen errors cleanly when only MLX is supported."""
    monkeypatch.setenv("IG_CONFIG_DIR", str(tmp_path))
    from imagegen import models
    from imagegen.platform import Backend

    m = models.get("ideogram4")
    monkeypatch.setattr(m, "supported_backends", [Backend.MLX])

    # Set backend to torch via config
    r_cfg = run(["ideogram4", "config", "set", "backend", "torch"])
    assert r_cfg.exit_code == 0, r_cfg.output

    import imagegen.cli.actions as actions

    monkeypatch.setattr(actions, "resolve_weights_path", lambda name, override, cfg: tmp_path / "w")

    out = tmp_path / "o.png"
    r = run(["ideogram4", "gen", "-p", "x", "--seed", "1", "-o", str(out)])
    assert r.exit_code != 0
    assert "does not support" in r.output
    assert r.exception is None or isinstance(r.exception, SystemExit)


def test_model_show_unknown_model_clean_error() -> None:
    r = run(["model", "show", "nosuchmodel"])
    assert r.exit_code != 0
    assert "Unknown model" in r.output
    assert r.exception is None or isinstance(r.exception, SystemExit)
