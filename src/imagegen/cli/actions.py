"""Action bodies for the per-model CLI groups: gen (daemon client), serve, stop, config."""

from __future__ import annotations

import json as _json
import sys
from pathlib import Path
from typing import Any

import click

from .. import daemon
from ..config import Config, Secrets, resolve_weights_path, daemon_log_path
from ..magic_prompt.providers import ALL_PROVIDERS
from ..platform import Backend, default_backend
from ..worker import stream_request


def _resolve_backend(model: Any, config: Config) -> Backend:
    name = config.model_backend(model.name)
    be = Backend(name) if name else default_backend()
    if be not in model.supported_backends:
        raise click.ClickException(
            f"{model.name} does not support backend {be.value}; supported: "
            f"{', '.join(b.value for b in model.supported_backends)}"
        )
    return be


def _build_pipeline_for(model: Any) -> Any:
    config = Config.load()
    secrets = Secrets.load()
    be = _resolve_backend(model, config)
    weights = resolve_weights_path(model.name, None, config)
    return model.build_pipeline(weights_path=weights, backend=be, config=config, secrets=secrets)


def _render_progress(d: dict[str, Any]) -> None:
    sys.stderr.write(f"\r[{d.get('phase', '')}] ".ljust(40))
    sys.stderr.flush()


def run_gen(model: Any, out: str, gen_opts: dict[str, Any]) -> None:
    sock = daemon.ensure_daemon(model.name)  # auto-start + wait-ready
    caption = gen_opts.get("caption")
    if caption:
        req: dict[str, Any] = {
            "op": "generate",
            "caption": _json.loads(Path(caption).read_text()),
            "width": gen_opts["width"],
            "height": gen_opts["height"],
            "preset": gen_opts.get("preset", "V4_DEFAULT_20"),
            "seed": gen_opts.get("seed"),
            "output_path": out,
        }
    else:
        req = {
            "op": "run",
            "prompt": gen_opts.get("prompt"),
            "width": gen_opts["width"],
            "height": gen_opts["height"],
            "preset": gen_opts.get("preset", "V4_DEFAULT_20"),
            "seed": gen_opts.get("seed"),
            "target_elements": gen_opts.get("target_elements", 0),
            "output_path": out,
        }
    result = stream_request(sock, req, _render_progress)
    sys.stderr.write("\r".ljust(40) + "\r")
    sys.stderr.flush()
    if not result.get("ok"):
        raise click.ClickException(str(result.get("error", "generation failed")))

    summary: dict[str, Any] = {
        "seed": result.get("seed"),
        "width": result.get("width"),
        "height": result.get("height"),
        "preset": result.get("preset"),
        "backend": result.get("backend"),
        "duration_s": result.get("duration_s"),
        "out": out,
        "model": model.name,
        "prompt": gen_opts.get("prompt"),
        "caption": result.get("caption"),
    }
    Path(out + ".json").write_text(_json.dumps(summary, indent=2))  # sidecar
    _json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")


def run_serve(model: Any, detach: bool) -> None:
    if daemon.live_record(model.name) is not None:
        rec = daemon.read_record(model.name)
        pid = rec["pid"] if rec is not None else "?"
        raise click.ClickException(
            f"{model.name} is already running (pid {pid}). Stop it first:  ig {model.name} stop"
        )
    if detach:
        daemon.ensure_daemon(model.name)
        click.echo(f"{model.name} daemon started (log: {daemon_log_path(model.name)})")
        return
    config = Config.load()
    be = _resolve_backend(model, config)
    quantize = config.model_quantize(model.name)
    daemon.run_daemon(
        model.name,
        lambda: _build_pipeline_for(model),
        backend=be.value,
        quantize=quantize,
    )


def run_stop(model: Any) -> None:
    if daemon.stop(model.name):
        click.echo(f"{model.name}: stopped")
    else:
        click.echo(f"{model.name}: no daemon running")


_CONFIG_SETTERS = {
    "weights-path": lambda cfg, model, value: cfg.set_model_path(model, value),
    "quantize": lambda cfg, model, value: cfg.set_model_quantize(model, value or None),
    "backend": lambda cfg, model, value: cfg.set_model_backend(model, value or None),
    "magic-provider": lambda cfg, model, value: cfg.set_magic_prompt_provider(value),
    "magic-model": lambda cfg, model, value: cfg.set_magic_prompt_model(value),
}

# Config keys whose value is constrained to a fixed set ("" clears where allowed).
_CONFIG_VALUE_CHOICES: dict[str, list[str]] = {
    "quantize": ["4", "8"],
    "backend": [b.value for b in Backend],
    "magic-provider": sorted(ALL_PROVIDERS),
}
_CLEARABLE = {"quantize", "backend"}


def run_config_set(model: Any, key: str, value: str) -> None:
    if key not in model.config_keys:
        raise click.ClickException(
            f"unknown config key {key!r}; valid: {', '.join(sorted(model.config_keys))}"
        )
    choices = _CONFIG_VALUE_CHOICES.get(key)
    if choices is not None and value and value not in choices:
        clear = " (or empty to clear)" if key in _CLEARABLE else ""
        raise click.ClickException(
            f"invalid value {value!r} for {key!r}; choose from: {', '.join(choices)}{clear}"
        )
    config = Config.load()
    _CONFIG_SETTERS[key](config, model.name, value)
    config.save()
    click.echo(f"{model.name}: set {key} = {value!r}")


def run_config_set_key(model: Any, provider: str, api_key: str) -> None:
    secrets = Secrets.load()
    secrets.set_api_key(provider, api_key)
    secrets.save()
    click.echo(f"{model.name}: stored API key for {provider}")


def run_config_show(model: Any) -> None:
    config = Config.load()
    click.echo(f"weights-path: {config.model_path(model.name) or '(not set)'}")
    click.echo(f"quantize: {config.model_quantize(model.name) or '(default fp8)'}")
    click.echo(f"backend: {config.model_backend(model.name) or '(auto)'}")
    click.echo(f"magic-provider: {config.magic_prompt_provider() or 'codex'}")
    click.echo(f"magic-model: {config.magic_prompt_model() or '(provider default)'}")
