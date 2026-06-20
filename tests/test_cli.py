# tests/test_cli.py
import json

from click.testing import CliRunner

from imagegen.cli import ig


def run(args, **kw):
    return CliRunner().invoke(ig, args, **kw)


# ---------------------------------------------------------------------------
# platform + model group
# ---------------------------------------------------------------------------


def test_platform_cmd():
    r = run(["platform"])
    assert r.exit_code == 0
    assert "default_backend" in r.output


def test_model_list_includes_ideogram4():
    r = run(["model", "list"])
    assert r.exit_code == 0
    assert "ideogram4" in r.output


def test_model_ls_alias():
    assert run(["model", "ls"]).exit_code == 0


def test_model_show_outputs_info():
    r = run(["model", "show", "ideogram4"])
    assert r.exit_code == 0
    assert "ideogram4" in r.output
    assert "mlx" in r.output.lower() or "torch" in r.output.lower()


def test_model_set_path_writes_config(tmp_path, monkeypatch):
    monkeypatch.setenv("IG_CONFIG_DIR", str(tmp_path))
    import importlib

    import imagegen.config as c

    importlib.reload(c)
    r = run(["model", "set-path", "ideogram4", "/weights/ig4"])
    assert r.exit_code == 0
    assert (tmp_path / "config.toml").exists()
    assert (
        c.Config.load(tmp_path / "config.toml").model_path("ideogram4").as_posix()  # type: ignore[union-attr]
        == "/weights/ig4"
    )
    importlib.reload(c)


# ---------------------------------------------------------------------------
# gen / serve route through the model as a nested subcommand (no `--`)
# ---------------------------------------------------------------------------


def test_gen_routes_common_and_model_opts(monkeypatch, tmp_path):
    import imagegen.cli.gen as genmod
    from imagegen import models
    from imagegen.platform import Backend

    m = models.get("ideogram4")
    seen = {}

    class FakeImg:
        def save(self, p):
            open(p, "wb").close()

    class FakeResult:
        image = FakeImg()
        seed = 42
        width = 768
        height = 768
        preset = "V4_TURBO_12"
        backend = "mlx"
        duration_s = 1.0

    def fake_build(*, weights_path, backend, **opts):
        seen["build"] = {"weights_path": str(weights_path), "backend": backend, **opts}
        return "PIPE"

    def fake_run(pipeline, *, prompt, width, height, seed, **opts):
        seen["run"] = {"prompt": prompt, "width": width, "seed": seed, **opts}
        return FakeResult()

    monkeypatch.setattr(m, "build_pipeline", fake_build)
    monkeypatch.setattr(m, "run_one", fake_run)
    monkeypatch.setattr(m, "supported_backends", [Backend.MLX])
    monkeypatch.setattr(genmod, "resolve_weights_path", lambda name, override, cfg: tmp_path / "w")

    out = tmp_path / "o.png"
    r = run(
        [
            "gen",
            "-p",
            "a cat",
            "-w",
            "768",
            "-h",
            "768",
            "--seed",
            "42",
            "-o",
            str(out),
            "ideogram4",
            "--preset",
            "V4_TURBO_12",
        ]
    )
    assert r.exit_code == 0, r.output
    assert out.exists()
    assert seen["run"]["prompt"] == "a cat"
    assert seen["run"]["preset"] == "V4_TURBO_12"
    assert seen["build"]["backend"] == Backend.MLX


def test_serve_builds_pipeline_and_calls_worker(monkeypatch, tmp_path):
    import imagegen.cli.serve as servemod
    from imagegen import models
    from imagegen.platform import Backend

    m = models.get("ideogram4")
    seen = {}

    def fake_build(*, weights_path, backend, **opts):
        seen["opts"] = opts
        return "PIPE"

    monkeypatch.setattr(m, "build_pipeline", fake_build)
    monkeypatch.setattr(m, "supported_backends", [Backend.MLX])
    monkeypatch.setattr(
        servemod, "resolve_weights_path", lambda name, override, cfg: tmp_path / "w"
    )
    monkeypatch.setattr(
        servemod,
        "worker_serve",
        lambda socket, pipeline: seen.update(socket=socket, pipeline=pipeline),
    )

    r = run(["serve", "--socket", str(tmp_path / "s.sock"), "ideogram4", "--preset", "V4_TURBO_12"])
    assert r.exit_code == 0, r.output
    assert seen["pipeline"] == "PIPE"  # type: ignore[comparison-overlap]
    assert seen["socket"] == str(tmp_path / "s.sock")  # type: ignore[comparison-overlap]
    assert seen["opts"]["preset"] == "V4_TURBO_12"


def test_gen_result_missing_preset_attribute(monkeypatch, tmp_path):
    """gen prints model-agnostic JSON even when the result lacks a .preset attribute."""
    import imagegen.cli.gen as genmod
    from imagegen import models
    from imagegen.platform import Backend

    m = models.get("ideogram4")

    class FakeImg:
        def save(self, p):
            open(p, "wb").close()

    class FakeResultNoPreset:
        image = FakeImg()
        seed = 123
        width = 512
        height = 512
        backend = "torch"
        duration_s = 2.5

    monkeypatch.setattr(m, "build_pipeline", lambda *, weights_path, backend, **o: "PIPE")
    monkeypatch.setattr(
        m, "run_one", lambda pipeline, *, prompt, width, height, seed, **o: FakeResultNoPreset()
    )
    monkeypatch.setattr(m, "supported_backends", [Backend.TORCH])
    monkeypatch.setattr(genmod, "resolve_weights_path", lambda name, override, cfg: tmp_path / "w")

    out = tmp_path / "t.png"
    r = run(
        [
            "gen",
            "-p",
            "test",
            "-w",
            "512",
            "-h",
            "512",
            "--seed",
            "123",
            "-o",
            str(out),
            "--backend",
            "torch",
            "ideogram4",
        ]
    )
    assert r.exit_code == 0, r.output
    assert out.exists()
    output_json = json.loads(r.output)
    assert output_json["preset"] is None
    assert output_json["seed"] == 123
    assert output_json["backend"] == "torch"
    assert output_json["duration_s"] == 2.5
    assert output_json["out"] == str(out)


def test_gen_short_flags_w_h_o(monkeypatch, tmp_path):
    """-w / -h / -o are aliases for --width / --height / --out on the gen group."""
    import imagegen.cli.gen as genmod
    from imagegen import models
    from imagegen.platform import Backend

    m = models.get("ideogram4")
    seen = {}

    class FakeImg:
        def save(self, p):
            open(p, "wb").close()

    class FakeResult:
        image = FakeImg()
        seed = 1
        width = 256
        height = 256
        preset = None
        backend = "mlx"
        duration_s = 0.0

    def fake_run(pipeline, *, prompt, width, height, seed, **o):
        seen["wh"] = (width, height)
        return FakeResult()

    monkeypatch.setattr(m, "build_pipeline", lambda *, weights_path, backend, **o: "P")
    monkeypatch.setattr(m, "run_one", fake_run)
    monkeypatch.setattr(m, "supported_backends", [Backend.MLX])
    monkeypatch.setattr(genmod, "resolve_weights_path", lambda name, override, cfg: tmp_path / "w")

    out = tmp_path / "o.png"
    r = run(["gen", "-p", "x", "-w", "256", "-h", "256", "-o", str(out), "ideogram4"])
    assert r.exit_code == 0, r.output
    assert seen["wh"] == (256, 256)
    assert out.exists()


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


def test_gen_unknown_model_clean_error(tmp_path):
    """ig gen with an unknown model is a Click 'No such command' error, not a traceback."""
    r = run(["gen", "-p", "x", "-o", str(tmp_path / "o.png"), "nosuchmodel"])
    assert r.exit_code != 0
    assert "No such command" in r.output
    assert r.exception is None or isinstance(r.exception, SystemExit)


def test_serve_unknown_model_clean_error(tmp_path):
    r = run(["serve", "--socket", str(tmp_path / "s.sock"), "nosuchmodel"])
    assert r.exit_code != 0
    assert "No such command" in r.output
    assert r.exception is None or isinstance(r.exception, SystemExit)


def test_gen_missing_out_clean_error():
    """Omitting -o/--out produces a clean error (no traceback)."""
    r = run(["gen", "-p", "x", "ideogram4", "--preset", "V4_TURBO_12"])
    assert r.exit_code != 0
    assert "out" in r.output.lower()
    assert r.exception is None or isinstance(r.exception, SystemExit)


def test_serve_unsupported_backend_clean_error(monkeypatch, tmp_path):
    """ig serve with an unsupported --backend errors cleanly before building anything."""
    from imagegen import models
    from imagegen.platform import Backend

    m = models.get("ideogram4")
    monkeypatch.setattr(m, "supported_backends", [Backend.MLX])

    r = run(["serve", "--socket", str(tmp_path / "s.sock"), "--backend", "torch", "ideogram4"])
    assert r.exit_code != 0
    assert "does not support" in r.output
    assert r.exception is None or isinstance(r.exception, SystemExit)


def test_model_show_unknown_model_clean_error():
    r = run(["model", "show", "nosuchmodel"])
    assert r.exit_code != 0
    assert "Unknown model" in r.output
    assert r.exception is None or isinstance(r.exception, SystemExit)


def test_model_set_path_unknown_model_clean_error(tmp_path):
    r = run(["model", "set-path", "nosuchmodel", "/some/path"])
    assert r.exit_code != 0
    assert "Unknown model" in r.output
    assert r.exception is None or isinstance(r.exception, SystemExit)
