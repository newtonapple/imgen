import pytest
from imgen.caption import CaptionError, model_caption, validate_caption

GOOD = {
    "high_level_description": "a cat",
    "style_description": {"aesthetics": "warm"},
    "compositional_deconstruction": {"background": "room", "elements": []},
}


def test_good_caption_has_no_issues():
    assert validate_caption(GOOD) == []


def test_missing_keys_reported():
    issues = validate_caption({"high_level_description": "x"})
    assert any("style_description" in i for i in issues)
    assert any("compositional_deconstruction" in i for i in issues)


def test_raise_on_issues():
    with pytest.raises(CaptionError):
        validate_caption({}, raise_on_issues=True)


def test_warn_does_not_raise():
    assert validate_caption({}, raise_on_issues=False)  # non-empty list, no raise


def test_model_caption_drops_non_schema_keys():
    full = {**GOOD, "aspect_ratio": "9:16", "extra": 1}
    out = model_caption(full)
    assert out == GOOD  # only the three schema keys remain
    assert "aspect_ratio" not in out and "extra" not in out


def test_model_caption_keeps_only_present_schema_keys():
    out = model_caption({"high_level_description": "x", "aspect_ratio": "1:1"})
    assert out == {"high_level_description": "x"}
