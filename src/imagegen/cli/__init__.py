"""`ig` — model-agnostic image-generation CLI (model-first)."""

from __future__ import annotations

import json
import sys

import click

from .. import models as _models  # noqa: F401  (registry)
from ..models import ideogram4 as _ideogram4  # noqa: F401  (registers the model)
from ..platform import platform_summary


@click.group("ig")
@click.version_option(package_name="imagegen", prog_name="ig")
def ig() -> None:
    """Generate images from text via pluggable models."""


@ig.command("platform")
def platform_cmd() -> None:
    """Show detected platform + default backend."""
    json.dump(platform_summary(), sys.stdout, indent=2)
    sys.stdout.write("\n")


from .model import model_group  # noqa: E402
from .model_cli import build_model_group  # noqa: E402

ig.add_command(model_group)
for _model in _models.all_models():
    _grp = build_model_group(_model)
    ig.add_command(_grp)
    for _alias in _model.aliases:
        ig.add_command(_grp, name=_alias)

main = ig
