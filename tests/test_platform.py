from imgen.platform import (
    Backend,
    default_backend,
    is_apple_silicon,
    platform_summary,
)


def test_summary_has_expected_keys():
    s = platform_summary()
    assert set(s) >= {"system", "machine", "apple_silicon", "cuda", "default_backend"}


def test_default_backend_matches_platform():
    expected = Backend.MLX if is_apple_silicon() else Backend.TORCH
    assert default_backend() == expected
    assert platform_summary()["default_backend"] == expected.value
