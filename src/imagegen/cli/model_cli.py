"""Build a Click group per model: `ig <model> gen|serve|config`."""

from __future__ import annotations

from typing import Any

import click

from . import actions


def build_model_group(model: Any) -> click.Group:
    group = click.Group(model.name, help=model.description)

    def gen_callback(out: str, **gen_opts: Any) -> None:
        actions.run_gen(model, out, gen_opts)

    gen_params: list[click.Parameter] = [
        *model.gen_options,
        click.Option(["-o", "--out"], required=True, type=click.Path(), help="output image path"),
    ]
    group.add_command(
        click.Command("gen", params=gen_params, callback=gen_callback, help="generate one image")
    )

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
            help="run a warm worker (foreground)",
        )
    )

    config_group = click.Group("config", help="get/set persisted config for this model")

    def set_callback(key: str, value: str) -> None:
        actions.run_config_set(model, key, value)

    config_group.add_command(
        click.Command(
            "set",
            params=[click.Argument(["key"]), click.Argument(["value"])],
            callback=set_callback,
            help="set a config value (keys: " + ", ".join(sorted(model.config_keys)) + ")",
        )
    )

    def set_key_callback(provider: str, api_key: str) -> None:
        actions.run_config_set_key(model, provider, api_key)

    config_group.add_command(
        click.Command(
            "set-key",
            params=[click.Argument(["provider"]), click.Argument(["api_key"])],
            callback=set_key_callback,
            help="store an API key for a provider in secrets.toml",
        )
    )

    config_group.add_command(
        click.Command("show", callback=lambda: actions.run_config_show(model), help="show config")
    )
    group.add_command(config_group)
    return group
