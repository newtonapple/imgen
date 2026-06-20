"""Action bodies for the per-model CLI groups: gen (in-process), serve, config."""

from __future__ import annotations

import json
import sys
from typing import Any

import click

from ..config import Config, Secrets, resolve_weights_path
from ..platform import Backend, default_backend
from ..worker import serve as worker_serve


def _resolve_backend(model: Any, config: Config) -> Backend:
    name = config.model_backend(model.name)
    be = Backend(name) if name else default_backend()
    if be not in model.supported_backends:
        raise click.ClickException(
            f"{model.name} does not support backend {be.value}; supported: "
            f"{', '.join(b.value for b in model.supported_backends)}"
        )
    return be


def run_gen(model: Any, out: str, gen_opts: dict[str, Any]) -> None:
    config = Config.load()
    secrets = Secrets.load()
    be = _resolve_backend(model, config)
    if gen_opts.get("seed") is None:
        sys.stderr.write(
            "warning: no --seed; re-seeding will likely change the image substantially\n"
        )
    weights = resolve_weights_path(model.name, None, config)
    pipeline = model.build_pipeline(
        weights_path=weights, backend=be, config=config, secrets=secrets
    )
    result = model.run_one(pipeline, **gen_opts)
    result.image.save(out)
    summary: dict[str, Any] = {
        field: getattr(result, field, None)
        for field in ("seed", "width", "height", "preset", "backend", "duration_s")
    }
    summary["out"] = out
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")


def run_serve(model: Any, socket_path: str, log_path: str | None) -> None:
    config = Config.load()
    secrets = Secrets.load()
    be = _resolve_backend(model, config)
    weights = resolve_weights_path(model.name, None, config)
    pipeline = model.build_pipeline(
        weights_path=weights, backend=be, config=config, secrets=secrets
    )
    log = open(log_path, "a") if log_path else sys.stderr  # noqa: SIM115
    log.write(f"ig serve: {model.name} warm on {socket_path}\n")
    log.flush()
    worker_serve(socket_path, pipeline)


_CONFIG_SETTERS = {
    "weights-path": lambda cfg, model, value: cfg.set_model_path(model, value),
    "quantize": lambda cfg, model, value: cfg.set_model_quantize(model, value or None),
    "backend": lambda cfg, model, value: cfg.set_model_backend(model, value or None),
    "magic-provider": lambda cfg, model, value: cfg.set_magic_prompt_provider(value),
    "magic-model": lambda cfg, model, value: cfg.set_magic_prompt_model(value),
}


def run_config_set(model: Any, key: str, value: str) -> None:
    if key not in model.config_keys:
        raise click.ClickException(
            f"unknown config key {key!r}; valid: {', '.join(sorted(model.config_keys))}"
        )
    config = Config.load()
    _CONFIG_SETTERS[key](config, model.name, value)
    config.save()
    click.echo(f"{model.name}: set {key} = {value}")


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
