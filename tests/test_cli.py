# tests/test_cli.py
"""CLI tests for the model-first `ig <model> <action>` grammar."""

from __future__ import annotations

import json
from typing import Any, Callable

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
# ig ideogram4 gen — daemon client (monkeypatched)
# ---------------------------------------------------------------------------


def test_gen_streams_and_writes_sidecar(monkeypatch: Any, tmp_path: Any) -> None:
    import imagegen.cli.actions as actions
    from imagegen import daemon

    monkeypatch.setattr(daemon, "ensure_daemon", lambda name: "/tmp/fake.sock")

    def fake_stream(
        sock: str,
        req: dict[str, Any],
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if on_progress:
            on_progress({"type": "progress", "phase": "sampling"})
        open(req["output_path"], "wb").close()  # daemon would write the PNG
        return {
            "ok": True,
            "seed": 42,
            "width": 768,
            "height": 768,
            "preset": "V4_TURBO_12",
            "backend": "mlx",
            "duration_s": 1.0,
            "caption": {"high_level_description": req.get("prompt")},
        }

    monkeypatch.setattr(actions, "stream_request", fake_stream)
    out = tmp_path / "o.png"
    r = run(["ideogram4", "gen", "-p", "a cat", "-w", "768", "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert out.exists() and (tmp_path / "o.png.json").exists()
    meta = json.loads((tmp_path / "o.png.json").read_text())
    assert meta["caption"]["high_level_description"] == "a cat"
    assert meta["model"] == "ideogram4"
    assert meta["prompt"] == "a cat"
    assert json.loads(r.stdout)["out"] == str(out)


def test_gen_with_seed_and_preset(monkeypatch: Any, tmp_path: Any) -> None:
    import imagegen.cli.actions as actions
    from imagegen import daemon

    monkeypatch.setattr(daemon, "ensure_daemon", lambda name: "/tmp/fake.sock")
    seen: dict[str, Any] = {}

    def fake_stream(
        sock: str,
        req: dict[str, Any],
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        seen.update(req)
        open(req["output_path"], "wb").close()
        return {
            "ok": True,
            "seed": req.get("seed"),
            "width": req["width"],
            "height": req["height"],
            "preset": req.get("preset"),
            "backend": "mlx",
            "duration_s": 0.5,
            "caption": None,
        }

    monkeypatch.setattr(actions, "stream_request", fake_stream)
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
            "--preset",
            "V4_TURBO_12",
            "-o",
            str(out),
        ]
    )
    assert r.exit_code == 0, r.output
    assert seen["seed"] == 123
    assert seen["preset"] == "V4_TURBO_12"
    assert seen["width"] == 512
    output_json = json.loads(r.stdout)
    assert output_json["seed"] == 123
    assert output_json["out"] == str(out)


def test_gen_error_from_daemon(monkeypatch: Any, tmp_path: Any) -> None:
    """When the daemon returns ok=False, gen exits non-zero with a clean error."""
    import imagegen.cli.actions as actions
    from imagegen import daemon

    monkeypatch.setattr(daemon, "ensure_daemon", lambda name: "/tmp/fake.sock")
    monkeypatch.setattr(
        actions,
        "stream_request",
        lambda sock, req, on_progress=None: {"ok": False, "error": "OOM"},
    )
    out = tmp_path / "o.png"
    r = run(["ideogram4", "gen", "-p", "x", "-o", str(out)])
    assert r.exit_code != 0
    assert "OOM" in r.output


# ---------------------------------------------------------------------------
# ig ideogram4 serve + stop
# ---------------------------------------------------------------------------


def test_serve_already_running_errors(monkeypatch: Any) -> None:
    from imagegen import daemon

    monkeypatch.setattr(daemon, "live_record", lambda name: {"pid": 999, "socket": "/tmp/x.sock"})
    monkeypatch.setattr(
        daemon,
        "read_record",
        lambda name: {"pid": 999, "socket": "/tmp/x.sock"},
    )
    r = run(["ideogram4", "serve"])
    assert r.exit_code != 0
    assert "already running" in r.output
    assert "999" in r.output


def test_serve_detach(monkeypatch: Any, tmp_path: Any) -> None:
    from imagegen import daemon
    from imagegen import config as cfg_mod

    monkeypatch.setattr(daemon, "live_record", lambda name: None)
    started: list[str] = []

    def fake_ensure(name: str) -> str:
        started.append(name)
        return "/tmp/fake.sock"

    monkeypatch.setattr(daemon, "ensure_daemon", fake_ensure)
    monkeypatch.setattr(cfg_mod, "daemon_log_path", lambda name: tmp_path / f"{name}.log")
    r = run(["ideogram4", "serve", "--detach"])
    assert r.exit_code == 0, r.output
    assert "ideogram4" in started
    assert "ideogram4 daemon started" in r.output


def test_stop_running_daemon(monkeypatch: Any) -> None:
    from imagegen import daemon

    stopped: list[str] = []

    def fake_stop(name: str) -> bool:
        stopped.append(name)
        return True

    monkeypatch.setattr(daemon, "stop", fake_stop)
    r = run(["ideogram4", "stop"])
    assert r.exit_code == 0, r.output
    assert "ideogram4" in stopped
    assert "stopped" in r.output


def test_stop_no_daemon(monkeypatch: Any) -> None:
    from imagegen import daemon

    monkeypatch.setattr(daemon, "stop", lambda name: False)
    r = run(["ideogram4", "stop"])
    assert r.exit_code == 0, r.output
    assert "no daemon" in r.output


# ---------------------------------------------------------------------------
# ig model stop-all
# ---------------------------------------------------------------------------


def test_model_stop_all(monkeypatch: Any) -> None:
    from imagegen import daemon

    monkeypatch.setattr(
        daemon,
        "list_daemons",
        lambda: [{"model": "ideogram4", "pid": 1, "live": True}],
    )
    stopped: list[str] = []

    def fake_stop(name: str) -> bool:
        stopped.append(name)
        return True

    monkeypatch.setattr(daemon, "stop", fake_stop)
    r = run(["model", "stop-all"])
    assert r.exit_code == 0, r.output
    assert "ideogram4" in stopped
    assert "stopped 1" in r.output


# ---------------------------------------------------------------------------
# ig ideogram4 config — set / set-key / set bogus
# ---------------------------------------------------------------------------


def test_config_set_no_out_required(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.setenv("IG_CONFIG_DIR", str(tmp_path))
    r = run(["ideogram4", "config", "set", "magic-provider", "openrouter"])
    assert r.exit_code == 0, r.output
    from imagegen.config import Config

    assert Config.load(tmp_path / "config.toml").magic_prompt_provider() == "openrouter"


def test_config_set_key_writes_secrets(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.setenv("IG_CONFIG_DIR", str(tmp_path))
    r = run(["ideogram4", "config", "set-key", "openrouter", "sk-x"])
    assert r.exit_code == 0, r.output
    from imagegen.config import Secrets

    s = Secrets.load(tmp_path / "secrets.toml")
    assert s.api_key("openrouter") == "sk-x"


def test_config_set_bogus_key_errors(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.setenv("IG_CONFIG_DIR", str(tmp_path))
    r = run(["ideogram4", "config", "set", "bogus", "x"])
    assert r.exit_code != 0
    assert "bogus" in r.output or "unknown" in r.output.lower()


def test_config_set_weights_path_persists(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.setenv("IG_CONFIG_DIR", str(tmp_path))
    r = run(["ideogram4", "config", "set", "weights-path", "/weights/ig4"])
    assert r.exit_code == 0, r.output
    from imagegen.config import Config

    assert Config.load(tmp_path / "config.toml").model_path("ideogram4") is not None


def test_config_set_invalid_value_rejected(monkeypatch: Any, tmp_path: Any) -> None:
    """A fixed-choice config key rejects an out-of-set value with the allowed values."""
    monkeypatch.setenv("IG_CONFIG_DIR", str(tmp_path))
    r = run(["ideogram4", "config", "set", "quantize", "9"])
    assert r.exit_code != 0
    assert "choose from" in r.output and "4" in r.output and "8" in r.output
    assert not (tmp_path / "config.toml").exists()  # nothing persisted on rejection


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


def test_gen_missing_out_clean_error() -> None:
    """Omitting -o/--out produces a clean 'Missing option' error (no traceback)."""
    r = run(["ideogram4", "gen", "-p", "x"])
    assert r.exit_code != 0
    assert "out" in r.output.lower() or "missing" in r.output.lower()
    assert r.exception is None or isinstance(r.exception, SystemExit)


def test_gen_unknown_model_clean_error(tmp_path: Any) -> None:
    """ig nosuch gen … → Click 'No such command', not a traceback."""
    r = run(["nosuch", "gen", "-p", "x", "-o", str(tmp_path / "o.png")])
    assert r.exit_code != 0
    assert "No such command" in r.output
    assert r.exception is None or isinstance(r.exception, SystemExit)


def test_serve_backend_unsupported_error(monkeypatch: Any, tmp_path: Any) -> None:
    """serve (foreground) errors cleanly when the configured backend is unsupported."""
    monkeypatch.setenv("IG_CONFIG_DIR", str(tmp_path))
    from imagegen import daemon, models
    from imagegen.platform import Backend

    m = models.get("ideogram4")
    monkeypatch.setattr(m, "supported_backends", [Backend.MLX])

    # Set backend to torch via config
    r_cfg = run(["ideogram4", "config", "set", "backend", "torch"])
    assert r_cfg.exit_code == 0, r_cfg.output

    # No live daemon so serve tries foreground (hits _resolve_backend before building)
    monkeypatch.setattr(daemon, "live_record", lambda name: None)

    r = run(["ideogram4", "serve"])
    assert r.exit_code != 0
    assert "does not support" in r.output
    assert r.exception is None or isinstance(r.exception, SystemExit)


def test_model_show_unknown_model_clean_error() -> None:
    r = run(["model", "show", "nosuchmodel"])
    assert r.exit_code != 0
    assert "Unknown model" in r.output
    assert r.exception is None or isinstance(r.exception, SystemExit)


def test_model_jobs_list(monkeypatch: Any) -> None:
    from imagegen import jobs

    monkeypatch.setattr(
        jobs,
        "list_jobs",
        lambda: [
            {
                "id": "bbbbbb",
                "model": "ideogram4",
                "out": "/b.png",
                "status": "done",
                "started_at": 2.0,
            },
            {
                "id": "aaaaaa",
                "model": "ideogram4",
                "out": "/a.png",
                "status": "queued",
                "started_at": 1.0,
            },
        ],
    )
    r = run(["model", "jobs"])
    assert r.exit_code == 0, r.output
    assert (
        "bbbbbb" in r.output
        and "aaaaaa" in r.output
        and "done" in r.output
        and "queued" in r.output
    )


def test_model_jobs_show_one(monkeypatch: Any) -> None:
    from imagegen import jobs

    monkeypatch.setattr(
        jobs,
        "read_job",
        lambda jid: {
            "id": jid,
            "model": "ideogram4",
            "out": "/o.png",
            "status": "running",
            "started_at": 1.0,
            "finished_at": None,
            "log": "/tmp/x.log",
        },
    )
    r = run(["model", "jobs", "ab12cd"])
    assert r.exit_code == 0, r.output
    assert "ab12cd" in r.output and "running" in r.output and "/tmp/x.log" in r.output


def test_model_jobs_show_unknown(monkeypatch: Any) -> None:
    from imagegen import jobs

    monkeypatch.setattr(jobs, "read_job", lambda jid: None)
    r = run(["model", "jobs", "nope12"])
    assert r.exit_code != 0
    assert "no such job" in r.output.lower()


def test_model_clean_cmd(monkeypatch: Any) -> None:
    from imagegen import jobs

    monkeypatch.setattr(jobs, "clean", lambda **kw: {"jobs": 2, "logs": 1, "truncated": 0})
    r = run(["model", "clean"])
    assert r.exit_code == 0, r.output
    assert "2" in r.output and "removed" in r.output.lower()


def test_model_list_shows_state(monkeypatch: Any) -> None:
    from imagegen import daemon

    monkeypatch.setattr(
        daemon,
        "list_daemons",
        lambda: [{"model": "ideogram4", "pid": 7, "state": "busy", "live": True}],
    )
    r = run(["model", "list"])
    assert r.exit_code == 0, r.output
    assert "busy" in r.output and "pid 7" in r.output


def test_gen_queue_spawns_and_prints_job(monkeypatch: Any, tmp_path: Any) -> None:
    from imagegen import jobs

    monkeypatch.setattr(jobs, "spawn_runner", lambda job_id: 4242)
    monkeypatch.setenv("IG_RUNTIME_DIR", str(tmp_path))
    import imagegen.config as cfg
    import importlib

    importlib.reload(cfg)
    importlib.reload(jobs)
    monkeypatch.setattr(jobs, "spawn_runner", lambda job_id: 4242)
    out = tmp_path / "o.png"
    r = run(["ideogram4", "gen", "-p", "a cat", "-o", str(out), "--queue"])
    assert r.exit_code == 0, r.output
    assert "poll: ig model jobs" in r.output
    listed = jobs.list_jobs()
    assert len(listed) == 1 and listed[0]["status"] == "queued"
    importlib.reload(cfg)
