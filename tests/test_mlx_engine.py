"""Integration test for the MLX backend — needs Apple Silicon + the fp8 weights.

Run with: pytest tests/test_mlx_engine.py -m integration
"""

from pathlib import Path

import pytest

from imgen.config import ModelSpec
from imgen.engine.factory import create_pipeline
from imgen.platform import Backend, is_apple_silicon

MODEL = Path("/Volumes/PRO-G40/data/models/image-gen/ideogram-4-fp8")

CAPTION = {
    "high_level_description": "a red cube on a white studio table",
    "style_description": {
        "aesthetics": "clean studio product shot",
        "color_palette": ["#FF0000", "#FFFFFF"],
    },
    "compositional_deconstruction": {
        "background": "white seamless backdrop",
        "elements": [{"type": "obj", "bbox": [300, 300, 724, 724], "desc": "a glossy red cube"}],
    },
}


@pytest.mark.integration
@pytest.mark.skipif(
    not is_apple_silicon() or not MODEL.exists(),
    reason="needs Apple Silicon + downloaded fp8 weights",
)
def test_mlx_engine_generates_image():
    engine = create_pipeline(ModelSpec(name="fp8", path=MODEL, backend=Backend.MLX))
    result = engine.generate(CAPTION, width=512, height=512, preset="V4_TURBO_12", seed=0)
    assert result.backend == "mlx"
    assert result.seed == 0
    assert result.image.size == (512, 512)


@pytest.mark.skipif(not is_apple_silicon(), reason="MLX backend needs Apple Silicon + mflux")
def test_mlx_engine_clear_error_on_wrong_layout(tmp_path):
    # An empty/MLXBits-style dir fails mflux's fast layout check (no weight load);
    # MlxEngine should raise a clear, actionable error rather than mflux's raw one.
    from imgen.engine.mlx_engine import MlxEngine

    with pytest.raises(RuntimeError, match="official fp8 checkpoint layout"):
        MlxEngine(ModelSpec(name="bad", path=tmp_path, backend=Backend.MLX))
