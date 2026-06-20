"""Configuration: model specs + engine config.

Weights live OUTSIDE the repo (e.g. an external volume, or ~/ai/image-gen) and
are referenced by path only — never committed.
The weights root is supplied via the IMGEN_WEIGHTS_ROOT env var or an explicit
path, so nothing is hardcoded.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .platform import Backend, default_backend

DEFAULT_PRESET = "V4_DEFAULT_20"
WEIGHTS_ROOT_ENV = "IMGEN_WEIGHTS_ROOT"
BACKEND_ENV = "IG_BACKEND"
QUANTIZE_ENV = "IG_QUANTIZE"
WEIGHTS_PATH_ENV = "IG_WEIGHTS_PATH"
MAGIC_PROVIDER_ENV = "IG_MAGIC_PROVIDER"
MAGIC_MODEL_ENV = "IG_MAGIC_MODEL"

_VALID_QUANTIZE = {"4", "8"}
_VALID_BACKEND = {"mlx", "torch"}


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


RUNTIME_DIR_ENV = "IG_RUNTIME_DIR"
# AF_UNIX sun_path limit: 104 on Darwin (BSD), 108 on Linux. Leave 1 byte for NUL.
_SUN_PATH_MAX = 104 if sys.platform == "darwin" else 108


def runtime_dir() -> Path:
    v = os.environ.get(RUNTIME_DIR_ENV)
    return Path(v).expanduser() if v else (Path.home() / ".cache" / "ig")


def daemons_dir() -> Path:
    return runtime_dir() / "daemons"


def logs_dir() -> Path:
    return runtime_dir() / "logs"


def daemon_socket_path(model: str) -> Path:
    return daemons_dir() / f"{model}.sock"


def daemon_record_path(model: str) -> Path:
    return daemons_dir() / f"{model}.json"


def daemon_log_path(model: str) -> Path:
    return logs_dir() / f"{model}.log"


def jobs_dir() -> Path:
    return runtime_dir() / "jobs"


def job_record_path(job_id: str) -> Path:
    return jobs_dir() / f"{job_id}.json"


def job_log_path(job_id: str) -> Path:
    return jobs_dir() / f"{job_id}.log"


def validate_socket_path(path: Path) -> None:
    if len(str(path).encode()) >= _SUN_PATH_MAX:
        raise RuntimeError(
            f"socket path too long for this platform ({len(str(path).encode())} >= "
            f"{_SUN_PATH_MAX} bytes): {path}. Set {RUNTIME_DIR_ENV} to a shorter directory."
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"runtime socket dir not writable: {path.parent} ({exc})") from exc


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

    def model_quantize(self, model: str) -> str | None:
        v = self.data.get("models", {}).get(model, {}).get("quantize")
        return str(v) if v else None

    def set_model_quantize(self, model: str, value: str | None) -> None:
        entry = self.data.setdefault("models", {}).setdefault(model, {})
        if value:
            entry["quantize"] = value
        else:
            entry.pop("quantize", None)

    def model_backend(self, model: str) -> str | None:
        v = self.data.get("models", {}).get(model, {}).get("backend")
        return str(v) if v else None

    def set_model_backend(self, model: str, value: str | None) -> None:
        entry = self.data.setdefault("models", {}).setdefault(model, {})
        if value:
            entry["backend"] = value
        else:
            entry.pop("backend", None)

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


def resolve_quantize(cfg: Config, model: str) -> str | None:
    v = os.environ.get(QUANTIZE_ENV) or cfg.model_quantize(model)
    if not v:
        return None
    if v not in _VALID_QUANTIZE:
        raise ValueError(f"invalid quantize {v!r}; choose from: 4, 8")
    return v


def resolve_backend(cfg: Config, model: str) -> str | None:
    v = os.environ.get(BACKEND_ENV) or cfg.model_backend(model)
    if not v:
        return None
    if v not in _VALID_BACKEND:
        raise ValueError(f"invalid backend {v!r}; choose from: mlx, torch")
    return v


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
    """--model-path override → IG_WEIGHTS_PATH → config → IMGEN_WEIGHTS_ROOT/<name> → model_default → error."""
    if override:
        return Path(override).expanduser()
    env_path = os.environ.get(WEIGHTS_PATH_ENV)
    if env_path:
        return Path(env_path).expanduser()
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
        f"ig {model_name} config set weights-path <path>  (or set {WEIGHTS_ROOT_ENV})."
    )
