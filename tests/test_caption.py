import pytest
from imgen.caption import (
    CaptionError,
    model_caption,
    reorder_caption_keys,
    validate_caption,
)

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


# --- reorder_caption_keys -------------------------------------------------

SCRAMBLED = {
    "aspect_ratio": "1:1",
    "style_description": {
        "medium": "graphic_design",
        "lighting": "flat",
        "art_style": "poster",
        "aesthetics": "cozy",
        "color_palette": ["#F4F1EA"],
    },
    "compositional_deconstruction": {
        "elements": [
            {"desc": "a cabin", "bbox": [0, 0, 100, 100], "type": "obj"},
            {"desc": "label", "text": "ANIME", "type": "text", "bbox": [0, 0, 50, 50]},
        ],
        "background": "off-white sheet",
    },
    "high_level_description": "six-panel sheet",
}


def test_reorder_style_description_non_photo():
    out = reorder_caption_keys(SCRAMBLED)
    assert list(out["style_description"]) == [
        "aesthetics",
        "lighting",
        "medium",
        "art_style",
        "color_palette",
    ]


def test_reorder_compositional_and_elements():
    out = reorder_caption_keys(SCRAMBLED)
    cd = out["compositional_deconstruction"]
    assert list(cd) == ["background", "elements"]
    assert list(cd["elements"][0]) == ["type", "bbox", "desc"]  # obj order, missing keys skipped
    assert list(cd["elements"][1]) == ["type", "bbox", "text", "desc"]  # text order


def test_reorder_matches_official_verifier_clean():
    # After reorder, model_caption output must satisfy the schema-order contract.
    out = model_caption(SCRAMBLED)
    assert list(out) == [
        "high_level_description",
        "style_description",
        "compositional_deconstruction",
    ]
    assert list(out["style_description"])[0] == "aesthetics"
    assert list(out["compositional_deconstruction"])[0] == "background"


def test_reorder_does_not_mutate_input():
    import copy

    snapshot = copy.deepcopy(SCRAMBLED)
    reorder_caption_keys(SCRAMBLED)
    assert SCRAMBLED == snapshot
    assert list(SCRAMBLED["style_description"])[0] == "medium"  # original order intact


def test_reorder_ambiguous_style_left_untouched():
    # Neither photo nor art_style (or both) → ambiguous; leave order for the verifier.
    cap = {"style_description": {"medium": "x", "aesthetics": "y"}}
    assert list(reorder_caption_keys(cap)["style_description"]) == ["medium", "aesthetics"]


def test_reorder_unknown_keys_appended_last():
    cap = {"style_description": {"weird": 1, "medium": "x", "art_style": "y", "aesthetics": "z"}}
    keys = list(reorder_caption_keys(cap)["style_description"])
    assert keys == ["aesthetics", "medium", "art_style", "weird"]
