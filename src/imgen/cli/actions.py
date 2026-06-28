"""Action bodies for the per-model CLI groups: gen (daemon client), serve, stop, config."""

from __future__ import annotations

import json as _json
import sys
from pathlib import Path
from typing import Any

import click

from .. import daemon, metadata
from ..config import Config, Secrets, resolve_backend, resolve_weights_path, daemon_log_path
from ..engine.resolution import resolve_size
from ..magic_prompt.providers import ALL_PROVIDERS
from ..platform import Backend, default_backend
from ..worker import stream_request


def _resolve_backend(model: Any, config: Config) -> Backend:
    name = resolve_backend(config, model.name)
    be = Backend(name) if name else default_backend()
    if be not in model.supported_backends:
        raise click.ClickException(
            f"{model.name} does not support backend {be.value}; supported: "
            f"{', '.join(b.value for b in model.supported_backends)}"
        )
    return be


def _warn_if_live(model_name: str, build_overrides: dict[str, str] | None) -> None:
    if build_overrides and daemon.live_record(model_name) is not None:
        click.echo(
            f"warning: {model_name} daemon already running; build overrides ignored "
            f"(ig {model_name} stop to apply)",
            err=True,
        )


def _build_pipeline_for(model: Any) -> Any:
    config = Config.load()
    secrets = Secrets.load()
    be = _resolve_backend(model, config)
    weights = resolve_weights_path(model.name, None, config)
    return model.build_pipeline(weights_path=weights, backend=be, config=config, secrets=secrets)


def _render_progress(d: dict[str, Any]) -> None:
    sys.stderr.write(f"\r[{d.get('phase', '')}] ".ljust(40))
    sys.stderr.flush()


def _resolve_dims(width: Any, height: Any) -> tuple[int, int]:
    """Round to the model's pixel grid (multiples of 16, min 256); warn on change."""
    w, h = int(width), int(height)
    rw, rh = resolve_size(w, h)
    if rw != w:
        click.echo(f"warning: width {w} -> {rw} (Ideogram 4 requires multiples of 16)", err=True)
    if rh != h:
        click.echo(f"warning: height {h} -> {rh} (Ideogram 4 requires multiples of 16)", err=True)
    return rw, rh


def _build_request(out: str, gen_opts: dict[str, Any]) -> dict[str, Any]:
    width, height = _resolve_dims(gen_opts["width"], gen_opts["height"])
    caption = gen_opts.get("caption")
    if caption:
        return {
            "op": "generate",
            "caption": _json.loads(Path(caption).read_text()),
            "width": width,
            "height": height,
            "preset": gen_opts.get("preset", "V4_DEFAULT_20"),
            "seed": gen_opts.get("seed"),
            "output_path": out,
        }  # caption path skips magic-prompt, so --mp/--mm don't apply here
    return {
        "op": "run",
        "prompt": gen_opts.get("prompt"),
        "width": width,
        "height": height,
        "preset": gen_opts.get("preset", "V4_DEFAULT_20"),
        "seed": gen_opts.get("seed"),
        "target_elements": gen_opts.get("target_elements", 0),
        "output_path": out,
        "magic_provider": gen_opts.get("magic_provider"),
        "magic_model": gen_opts.get("magic_model"),
    }


def run_gen(
    model: Any, out: str, gen_opts: dict[str, Any], build_overrides: dict[str, str] | None = None
) -> None:
    req = _build_request(out, gen_opts)
    if gen_opts.get("queue"):
        from .. import jobs

        job_id = jobs.new_job_id()
        jobs.create_job(job_id, model=model.name, out=out, request=req)
        jobs.spawn_runner(job_id)
        click.echo(f"job {job_id} → {out}   (poll: ig model jobs {job_id})")
        return
    if gen_opts.get("json"):
        _warn_if_live(model.name, build_overrides)
        sock = daemon.ensure_daemon(model.name)

        def _emit(d: dict[str, Any]) -> None:
            sys.stdout.write(_json.dumps(d) + "\n")
            sys.stdout.flush()

        result = stream_request(sock, req, _emit)
        _emit(result)
        if result.get("ok"):
            summary = metadata.build_summary(out, result, model=model.name,
                                             prompt=gen_opts.get("prompt"))
            metadata.write_sidecar(out, summary)
        else:
            raise click.ClickException(str(result.get("error", "generation failed")))
        return
    _warn_if_live(model.name, build_overrides)
    sock = daemon.ensure_daemon(model.name)  # auto-start + wait-ready
    result = stream_request(sock, req, _render_progress)
    sys.stderr.write("\r".ljust(40) + "\r")
    sys.stderr.flush()
    if not result.get("ok"):
        raise click.ClickException(str(result.get("error", "generation failed")))

    summary = metadata.build_summary(out, result, model=model.name, prompt=gen_opts.get("prompt"))
    metadata.write_sidecar(out, summary)  # sidecar
    _json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")


def run_serve(model: Any, detach: bool, build_overrides: dict[str, str] | None = None) -> None:
    _warn_if_live(model.name, build_overrides)
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
