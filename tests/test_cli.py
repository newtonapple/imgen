# tests/test_cli.py   (REPLACES the old argparse tests)
from click.testing import CliRunner

from imagegen.cli import ig


def run(args, **kw):
    return CliRunner().invoke(ig, args, **kw)


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
        c.Config.load(tmp_path / "config.toml").model_path("ideogram4").as_posix() == "/weights/ig4"
    )
    importlib.reload(c)


def genmod_options_for_test():
    import click

    @click.command("ideogram4")
    @click.option("--preset", default="V4_DEFAULT_20")
    def _o(**k):
        pass

    return _o


def test_gen_routes_globals_and_model_opts(monkeypatch, tmp_path):
    import imagegen.cli.gen as genmod
    from imagegen.platform import Backend

    seen = {}

    class FakeImg:
        size = (768, 768)

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

    class FakeModel:
        name = "ideogram4"
        aliases = []
        description = ""
        supported_backends = [Backend.MLX]
        model_options = genmod_options_for_test()

        def default_weights_path(self, cfg):
            return None

        def build_pipeline(self, *, weights_path, backend, **opts):
            seen["build"] = {"weights_path": str(weights_path), "backend": backend, **opts}
            return "PIPE"

        def run_one(self, pipeline, *, prompt, width, height, seed, **opts):
            seen["run"] = {"prompt": prompt, "width": width, "seed": seed, **opts}
            return FakeResult()

    monkeypatch.setattr(genmod.models, "get", lambda name: FakeModel())
    monkeypatch.setattr(genmod, "resolve_weights_path", lambda name, override, cfg: tmp_path / "w")

    out = tmp_path / "o.png"
    r = run(
        [
            "gen",
            "-p",
            "a cat",
            "--width",
            "768",
            "--height",
            "768",
            "--seed",
            "42",
            "--out",
            str(out),
            "ideogram4",
            "--",
            "--preset",
            "V4_TURBO_12",
        ]
    )
    assert r.exit_code == 0, r.output
    assert out.exists()
    assert seen["run"]["prompt"] == "a cat"
    assert seen["run"]["preset"] == "V4_TURBO_12"
    assert seen["build"]["backend"] == Backend.MLX
