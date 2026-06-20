import json


def test_build_summary_shape():
    from imagegen.metadata import build_summary

    result = {
        "seed": 42,
        "width": 768,
        "height": 512,
        "preset": "V4_TURBO_12",
        "backend": "mlx",
        "duration_s": 1.5,
        "caption": {"high_level_description": "a cat"},
    }
    s = build_summary("/tmp/o.png", result, model="ideogram4", prompt="a cat")
    assert s == {
        "seed": 42,
        "width": 768,
        "height": 512,
        "preset": "V4_TURBO_12",
        "backend": "mlx",
        "duration_s": 1.5,
        "out": "/tmp/o.png",
        "model": "ideogram4",
        "prompt": "a cat",
        "caption": {"high_level_description": "a cat"},
    }


def test_write_sidecar(tmp_path):
    from imagegen.metadata import write_sidecar

    out = tmp_path / "o.png"
    write_sidecar(str(out), {"model": "ideogram4", "out": str(out)})
    meta = json.loads((tmp_path / "o.png.json").read_text())
    assert meta["model"] == "ideogram4"
