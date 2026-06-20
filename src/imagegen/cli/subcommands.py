"""Register each model as a real nested subcommand under both `gen` and `serve`.

The model's existing ``model_options`` (a Click command) supplies the parameter
list; the callback routes the parsed options into the model's ``build_pipeline``
+ ``run_one`` (gen) or the warm worker (serve). This is what lets

    ig gen [common options] <model> [model options]

work without a ``--`` separator.
"""

from __future__ import annotations

from typing import Any

import click

from .. import models
from .gen import gen, run_gen
from .serve import run_serve, serve


def make_subcommand(model: Any, action: str) -> click.Command:
    def callback(**model_opts: Any) -> None:
        if action == "gen":
            run_gen(model, model_opts)
        else:
            run_serve(model, model_opts)

    return click.Command(
        model.name,
        params=list(model.model_options.params),
        callback=callback,
        help=model.description,
    )


def register_models() -> None:
    """Attach every registered model as a subcommand of both `gen` and `serve`."""
    for model in models.all_models():
        for action, group in (("gen", gen), ("serve", serve)):
            cmd = make_subcommand(model, action)
            group.add_command(cmd)
            for alias in model.aliases:
                group.add_command(cmd, name=alias)
