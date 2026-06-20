"""Build a Click group per model: `ig <model> gen|serve|config`."""

from __future__ import annotations

import os
from typing import Any

import click

from . import actions
from .. import config as cfg_mod
from ..magic_prompt.providers import ALL_PROVIDERS, HTTP_PROVIDERS


def build_model_group(model: Any) -> click.Group:
    backends = [b.value for b in model.supported_backends]

    def _group_cb(
        ctx: click.Context,
        weights_path: str | None,
        backend: str | None,
        quantize: str | None,
        magic_provider: str | None,
        magic_model: str | None,
    ) -> None:
        overrides: dict[str, str] = {}
        pairs = [
            (weights_path, cfg_mod.WEIGHTS_PATH_ENV, "weights-path"),
            (backend, cfg_mod.BACKEND_ENV, "backend"),
            (quantize, cfg_mod.QUANTIZE_ENV, "quantize"),
            (magic_provider, cfg_mod.MAGIC_PROVIDER_ENV, "magic-provider"),
            (magic_model, cfg_mod.MAGIC_MODEL_ENV, "magic-model"),
        ]
        for value, env_name, key in pairs:
            if value is not None:
                os.environ[env_name] = value
                overrides[key] = value
        ctx.ensure_object(dict)
        ctx.obj["build_overrides"] = overrides

    group = click.Group(
        model.name,
        help=model.description,
        short_help=model.description.split("—")[0].strip(),
        params=[
            click.Option(
                ["--weights-path"],
                default=None,
                type=click.Path(),
                help="override weights dir (build)",
            ),
            click.Option(
                ["--backend"],
                default=None,
                type=click.Choice(backends),
                help="override backend (build)",
            ),
            click.Option(
                ["--quantize"],
                default=None,
                type=click.Choice(["4", "8"]),
                help="override quantize (build)",
            ),
            click.Option(
                ["--magic-provider", "--mp", "magic_provider"],
                default=None,
                type=click.Choice(sorted(ALL_PROVIDERS)),
                help="daemon-default magic provider",
            ),
            click.Option(
                ["--magic-model", "--mm", "magic_model"],
                default=None,
                help="daemon-default magic model",
            ),
        ],
        callback=click.pass_context(_group_cb),
    )

    # --- gen -----------------------------------------------------------------
    @click.pass_context
    def gen_callback(ctx: click.Context, /, out: str, **gen_opts: Any) -> None:
        actions.run_gen(
            model, out, gen_opts, build_overrides=(ctx.obj or {}).get("build_overrides")
        )

    gen_params: list[click.Parameter] = [
        *model.gen_options,
        click.Option(["-o", "--out"], required=True, type=click.Path(), help="output image path"),
        click.Option(
            ["--queue", "-q"],
            is_flag=True,
            default=False,
            help="run in the background; returns a job id to poll with `ig model jobs`",
        ),
    ]
    group.add_command(
        click.Command(
            "gen",
            params=gen_params,
            callback=gen_callback,
            help="Generate one image via the warm daemon (auto-started if needed).",
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
    @click.pass_context
    def serve_callback(ctx: click.Context, /, detach: bool) -> None:
        actions.run_serve(model, detach, build_overrides=(ctx.obj or {}).get("build_overrides"))

    group.add_command(
        click.Command(
            "serve",
            params=[
                click.Option(
                    ["--detach", "-d"],
                    is_flag=True,
                    default=False,
                    help="run the daemon in the background",
                ),
            ],
            callback=serve_callback,
            help="Start the warm daemon (foreground; --detach to background).",
            short_help="start the warm daemon",
        )
    )

    # --- stop ----------------------------------------------------------------
    group.add_command(
        click.Command(
            "stop",
            callback=lambda: actions.run_stop(model),
            help="Stop this model's daemon.",
            short_help="stop the daemon",
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
