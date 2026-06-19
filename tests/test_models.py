import click
import pytest

from imagegen import models
from imagegen.platform import Backend


class FakeModel:
    name = "fake"
    aliases = ["fk"]
    description = "a fake model"
    supported_backends = [Backend.MLX]
    model_options = click.Command("fake", params=[])

    def default_weights_path(self, cfg):
        return None

    def build_pipeline(self, *, weights_path, backend, **opts):
        return ("pipeline", weights_path, backend, opts)

    def run_one(self, pipeline, *, prompt, width, height, seed, **opts):
        return ("result", prompt, opts)


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    monkeypatch.setattr(models, "_REGISTRY", {})


def test_register_and_get_by_name_and_alias():
    m = FakeModel()
    models.register(m)
    assert models.get("fake") is m
    assert models.get("fk") is m


def test_all_models_lists_registered():
    m = FakeModel()
    models.register(m)
    assert m in models.all_models()


def test_get_unknown_raises_with_available():
    with pytest.raises(KeyError, match="fake|available|Unknown"):
        models.get("nope")
