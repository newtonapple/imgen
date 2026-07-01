# tests/test_image_output.py
from pathlib import Path
from PIL import Image
from imgen.image_output import resolve_format, save_image


def test_resolve_format_explicit_wins():
    assert resolve_format("out.png", "webp") == "webp"
    assert resolve_format("out.webp", "png") == "png"


def test_resolve_format_infers_from_extension():
    assert resolve_format("a/b/c.webp", None) == "webp"
    assert resolve_format("a/b/c.PNG", None) == "png"


def test_resolve_format_defaults_png():
    assert resolve_format("noext", None) == "png"
    assert resolve_format("weird.jpg", None) == "png"


def _sample() -> Image.Image:
    im = Image.new("RGB", (8, 8))
    im.putpixel((0, 0), (10, 20, 30))
    im.putpixel((7, 7), (200, 100, 50))
    return im


def test_save_image_webp_is_lossless(tmp_path: Path):
    im = _sample()
    p = tmp_path / "x.webp"
    save_image(im, str(p), "webp")
    with Image.open(p) as got:
        assert got.format == "WEBP"
        assert got.convert("RGB").tobytes() == im.convert("RGB").tobytes()


def test_save_image_png_matches_pillow_default(tmp_path: Path):
    im = _sample()
    p1, p2 = tmp_path / "a.png", tmp_path / "b.png"
    save_image(im, str(p1), "png")
    im.save(str(p2))  # today's exact call
    assert p1.read_bytes() == p2.read_bytes()
