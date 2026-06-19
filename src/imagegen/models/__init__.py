"""Model registry. Importing a model module registers it via register()."""
from __future__ import annotations

from .base import Backend, Model

_REGISTRY: dict[str, Model] = {}


def register(model: Model) -> None:
    for key in (model.name, *model.aliases):
        _REGISTRY[key] = model


def get(name_or_alias: str) -> Model:
    try:
        return _REGISTRY[name_or_alias]
    except KeyError:
        names = sorted({m.name for m in _REGISTRY.values()})
        raise KeyError(f"Unknown model {name_or_alias!r}. Available: {', '.join(names) or '(none)'}")


def all_models() -> list[Model]:
    seen: dict[int, Model] = {}
    for m in _REGISTRY.values():
        seen[id(m)] = m
    return list(seen.values())


__all__ = ["Backend", "Model", "register", "get", "all_models"]
