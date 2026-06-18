"""Integration test for the MLX backend — needs Apple Silicon + the fp8 weights.

Run with: pytest tests/test_mlx_engine.py -m integration
"""

from pathlib import Path

import pytest

from imagegen.config import ModelSpec
from imagegen.engine.factory import create_pipeline
from imagegen.platform import Backend, is_apple_silicon

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
