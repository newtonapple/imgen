from pathlib import Path

import pytest

from imgen.config import ModelSpec
from imgen.engine.factory import create_pipeline, ensure_supported, resolve_backend
from imgen.platform import Backend, is_apple_silicon


def _torch_installed() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def test_resolve_backend_explicit():
    assert resolve_backend("mlx") == Backend.MLX
    assert resolve_backend("torch") == Backend.TORCH


def test_resolve_backend_from_model():
    m = ModelSpec(name="x", path=Path("/tmp/x"), backend=Backend.MLX)
    assert resolve_backend(None, m) == Backend.MLX


def test_modelspec_from_path_defaults_backend_to_platform():
    m = ModelSpec.from_path("/tmp/some-weights")
    assert m.name == "some-weights"
    assert m.backend == (Backend.MLX if is_apple_silicon() else Backend.TORCH)


@pytest.mark.skipif(not is_apple_silicon(), reason="MLX requires Apple Silicon")
def test_mlx_supported_here():
    ensure_supported(Backend.MLX)  # must not raise


@pytest.mark.skipif(
    _torch_installed(), reason="torch is installed; the unsupported path won't trigger"
)
def test_torch_backend_errors_without_torch():
    with pytest.raises(RuntimeError, match="cuda"):
        ensure_supported(Backend.TORCH)


@pytest.mark.skipif(not is_apple_silicon(), reason="MLX requires Apple Silicon")
def test_create_pipeline_dispatches_to_mlx(monkeypatch):
    # Verify the factory builds an MlxEngine for an MLX spec, without loading a
    # real model (MlxEngine.__init__ otherwise loads mflux from model.path).
    import imgen.engine.mlx_engine as mlx_engine

    monkeypatch.setattr(
        mlx_engine.MlxEngine,
        "__init__",
        lambda self, model, **opts: setattr(self, "model", model),
    )
    m = ModelSpec(name="x", path=Path("/tmp/x"), backend=Backend.MLX)
    engine = create_pipeline(m)
    assert engine.backend == "mlx"
    assert type(engine).__name__ == "MlxEngine"


@pytest.mark.skipif(not is_apple_silicon(), reason="MLX requires Apple Silicon")
def test_create_pipeline_forwards_quantize_to_mlx(monkeypatch):
    import imgen.engine.mlx_engine as mlx_engine

    seen = {}

    def fake_init(self, model, quantize=None, **opts):
        self.model = model
        seen["quantize"] = quantize

    monkeypatch.setattr(mlx_engine.MlxEngine, "__init__", fake_init)
    create_pipeline(ModelSpec(name="x", path=Path("/tmp/x"), backend=Backend.MLX), quantize=8)
    assert seen["quantize"] == 8
