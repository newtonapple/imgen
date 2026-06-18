from pathlib import Path

import pytest

from imagegen.config import ModelSpec
from imagegen.engine.factory import create_pipeline, ensure_supported, resolve_backend
from imagegen.platform import Backend, is_apple_silicon


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


@pytest.mark.skipif(_torch_installed(), reason="torch is installed; the unsupported path won't trigger")
def test_torch_backend_errors_without_torch():
    with pytest.raises(RuntimeError, match="cuda"):
        ensure_supported(Backend.TORCH)


@pytest.mark.skipif(not is_apple_silicon(), reason="MLX requires Apple Silicon")
def test_create_pipeline_builds_mlx_engine_stub(tmp_path):
    m = ModelSpec(name="x", path=tmp_path, backend=Backend.MLX)
    engine = create_pipeline(m)
    assert engine.backend == "mlx"
    with pytest.raises(NotImplementedError):
        engine.generate({}, width=512, height=512)
