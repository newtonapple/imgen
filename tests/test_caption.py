import pytest
from imagegen.caption import validate_caption, CaptionError

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
