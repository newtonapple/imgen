"""Configuration: model specs + engine config.

Weights live OUTSIDE the repo (an external volume on the Mac, ~/ai/image-gen on
the Spark) and are referenced by path only — never committed, never in iCloud.
The weights root is supplied via the IMAGEGEN_WEIGHTS_ROOT env var or an explicit
path, so nothing is hardcoded.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .platform import Backend, default_backend

DEFAULT_PRESET = "V4_DEFAULT_20"
WEIGHTS_ROOT_ENV = "IMAGEGEN_WEIGHTS_ROOT"


@dataclass(frozen=True)
class ModelSpec:
    """Identifies a set of weights on disk for a given backend."""

    name: str
    path: Path
    backend: Backend
    default_preset: str = DEFAULT_PRESET

    @classmethod
    def from_path(
        cls,
        path: str | os.PathLike[str],
        *,
        backend: Backend | str | None = None,
        name: str | None = None,
        default_preset: str = DEFAULT_PRESET,
    ) -> "ModelSpec":
        p = Path(path).expanduser()
        be = Backend(backend) if backend is not None else default_backend()
        return cls(name=name or p.name, path=p, backend=be, default_preset=default_preset)


@dataclass
class EngineConfig:
    """Everything an engine needs to load + run, beyond the model itself."""

    model: ModelSpec
    device: str | None = None  # backend-specific; None = auto
    options: dict[str, Any] = field(default_factory=dict)


def weights_root() -> Path | None:
    """Root directory holding per-model weight subdirs, if configured."""
    v = os.environ.get(WEIGHTS_ROOT_ENV)
    return Path(v).expanduser() if v else None


def _config_dir() -> Path:
    return Path(os.environ.get("IG_CONFIG_DIR", Path.home() / ".config" / "ig"))


def _config_path() -> Path:
    return _config_dir() / "config.toml"


def _secrets_path() -> Path:
    return _config_dir() / "secrets.toml"


# Module-level constants computed at import time (kept for backward compat).
# Prefer calling _config_path() directly when a live env-var value is needed.
CONFIG_DIR = _config_dir()
CONFIG_PATH = _config_path()


class Config:
    """Reads/writes ~/.config/ig/config.toml (per-model default weights paths)."""

    def __init__(self, data: dict[str, Any] | None = None, path: Path | None = None):
        self.data = data or {}
        self.path = Path(path) if path is not None else _config_path()

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = Path(path) if path is not None else _config_path()
        if path.exists():
            with open(path, "rb") as f:
                return cls(tomllib.load(f), path)
        return cls({}, path)

    def model_path(self, model: str) -> Path | None:
        p = self.data.get("models", {}).get(model, {}).get("path")
        return Path(p).expanduser() if p else None

    def set_model_path(self, model: str, path: str | os.PathLike[str]) -> None:
        models = self.data.setdefault("models", {})
        models.setdefault(model, {})["path"] = str(Path(path).expanduser())

    def magic_prompt_provider(self) -> str | None:
        v = self.data.get("magic_prompt", {}).get("provider")
        return str(v) if v else None

    def magic_prompt_model(self) -> str | None:
        v = self.data.get("magic_prompt", {}).get("model")
        return str(v) if v else None

    def set_magic_prompt_provider(self, provider: str) -> None:
        self.data.setdefault("magic_prompt", {})["provider"] = provider

    def set_magic_prompt_model(self, model: str) -> None:
        self.data.setdefault("magic_prompt", {})["model"] = model

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(_dump_toml(self.data))


def _dump_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    for model, conf in data.get("models", {}).items():
        lines.append(f"[models.{model}]")
        for key, value in conf.items():
            lines.append(f'{key} = "{value}"')
        lines.append("")
    mp = data.get("magic_prompt")
    if mp:
        lines.append("[magic_prompt]")
        for key, value in mp.items():
            lines.append(f'{key} = "{value}"')
        lines.append("")
    return "\n".join(lines)


class Secrets:
    """Reads/writes ~/.config/ig/secrets.toml (per-provider API keys, mode 600)."""

    def __init__(self, data: dict[str, Any] | None = None, path: Path | None = None):
        self.data = data or {}
        self.path = Path(path) if path is not None else _secrets_path()

    @classmethod
    def load(cls, path: Path | None = None) -> "Secrets":
        path = Path(path) if path is not None else _secrets_path()
        if path.exists():
            with open(path, "rb") as f:
                return cls(tomllib.load(f), path)
        return cls({}, path)

    def api_key(self, provider: str) -> str | None:
        v = self.data.get(provider, {}).get("api_key")
        return str(v) if v else None

    def set_api_key(self, provider: str, key: str) -> None:
        self.data.setdefault(provider, {})["api_key"] = key

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(_dump_secrets(self.data))
        self.path.chmod(0o600)


def _dump_secrets(data: dict[str, Any]) -> str:
    lines: list[str] = []
    for section, conf in data.items():
        lines.append(f"[{section}]")
        for key, value in conf.items():
            lines.append(f'{key} = "{value}"')
        lines.append("")
    return "\n".join(lines)


def resolve_weights_path(
    model_name: str,
    override: str | os.PathLike[str] | None,
    cfg: Config,
    model_default: Path | None = None,
) -> Path:
    """--model-path override → config → IMAGEGEN_WEIGHTS_ROOT/<name> → model_default → error."""
    if override:
        return Path(override).expanduser()
    configured = cfg.model_path(model_name)
    if configured:
        return configured
    root = os.environ.get(WEIGHTS_ROOT_ENV)
    if root:
        return Path(root).expanduser() / model_name
    if model_default:
        return model_default
    raise RuntimeError(
        f"No weights path for '{model_name}'. Set one with: "
        f"ig model set-path {model_name} <path>  (or pass --model-path, or set {WEIGHTS_ROOT_ENV})."
    )
