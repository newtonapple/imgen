"""Build a Click group per model: `ig <model> gen|serve|config`."""

from __future__ import annotations

from typing import Any

import click

from . import actions
from ..magic_prompt.providers import HTTP_PROVIDERS


def build_model_group(model: Any) -> click.Group:
    group = click.Group(
        model.name,
        help=model.description,
        short_help=model.description.split("—")[0].strip(),
    )

    # --- gen -----------------------------------------------------------------
    def gen_callback(out: str, **gen_opts: Any) -> None:
        actions.run_gen(model, out, gen_opts)

    gen_params: list[click.Parameter] = [
        *model.gen_options,
        click.Option(["-o", "--out"], required=True, type=click.Path(), help="output image path"),
    ]
    group.add_command(
        click.Command(
            "gen",
            params=gen_params,
            callback=gen_callback,
            help="Generate one image (loads the model in-process).",
            short_help="generate one image",
            epilog="Build options (quantize, backend, magic-prompt provider) come from "
            "`ig "
            + model.name
            + " config` — not from gen flags. Run `ig "
            + model.name
            + " config show` to see them.",
        )
    )

    # --- serve ---------------------------------------------------------------
    def serve_callback(socket_path: str, log_path: str | None) -> None:
        actions.run_serve(model, socket_path, log_path)

    group.add_command(
        click.Command(
            "serve",
            params=[
                click.Option(
                    ["--socket", "socket_path"],
                    required=True,
                    type=click.Path(),
                    help="Unix socket to listen on",
                ),
                click.Option(
                    ["--log", "log_path"], default=None, type=click.Path(), help="log file"
                ),
            ],
            callback=serve_callback,
            help="Run a warm worker in the foreground (reused build config from `config`).",
            short_help="run a warm worker (foreground)",
        )
    )

    # --- config --------------------------------------------------------------
    config_group = click.Group(
        "config",
        help="Get/set persisted build config for this model (weights, quantize, backend, "
        "magic-prompt provider/model, API keys).",
        short_help="get/set persisted config",
    )

    def set_callback(key: str, value: str) -> None:
        actions.run_config_set(model, key, value)

    keys = list(model.config_keys)
    set_epilog = "\b\nKeys and accepted values:\n" + "\n".join(
        f"  {k:<14} {model.config_keys[k]}" for k in keys
    )
    config_group.add_command(
        click.Command(
            "set",
            params=[
                click.Argument(["key"], type=click.Choice(keys)),
                click.Argument(["value"]),
            ],
            callback=set_callback,
            help="Set a persisted config value (use an empty value to clear quantize/backend).",
            short_help="set a config value",
            epilog=set_epilog,
        )
    )

    def set_key_callback(provider: str, api_key: str) -> None:
        actions.run_config_set_key(model, provider, api_key)

    key_providers = sorted(HTTP_PROVIDERS)
    config_group.add_command(
        click.Command(
            "set-key",
            params=[
                click.Argument(["provider"], type=click.Choice(key_providers)),
                click.Argument(["api_key"]),
            ],
            callback=set_key_callback,
            help="Store an API key for an HTTP magic-prompt provider in secrets.toml (mode 600).",
            short_help="store a provider API key",
            epilog="\b\nProviders needing a key: "
            + ", ".join(key_providers)
            + "\n(codex, claude, pi use a local CLI and need no key)",
        )
    )

    config_group.add_command(
        click.Command(
            "show",
            callback=lambda: actions.run_config_show(model),
            help="Show this model's current persisted config.",
            short_help="show current config",
        )
    )
    group.add_command(config_group)
    return group
