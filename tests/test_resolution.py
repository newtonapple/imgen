import pytest
from imagegen.engine.resolution import resolve_size, aspect_ratio


@pytest.mark.parametrize(
    "w,h,exp",
    [
        (1024, 1024, (1024, 1024)),
        (1000, 1000, (1008, 1008)),  # round to nearest 16
        (10, 10, (256, 256)),  # floor at 256
        (831, 1473, (832, 1472)),
    ],
)
def test_resolve_size(w, h, exp):
    assert resolve_size(w, h) == exp


@pytest.mark.parametrize(
    "w,h,exp",
    [
        (1024, 1024, "1:1"),
        (576, 1024, "9:16"),
        (1920, 1080, "16:9"),
    ],
)
def test_aspect_ratio(w, h, exp):
    assert aspect_ratio(w, h) == exp
