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
