"""Light, dependency-free Ideogram-4 caption validation (works on both backends)."""

from __future__ import annotations

REQUIRED_KEYS = ("high_level_description", "style_description", "compositional_deconstruction")


class CaptionError(ValueError):
    pass


def validate_caption(caption: dict, *, raise_on_issues: bool = False) -> list[str]:
    issues: list[str] = []
    if not isinstance(caption, dict):
        issues.append(f"caption must be a dict, got {type(caption).__name__}")
    else:
        for key in REQUIRED_KEYS:
            if key not in caption:
                issues.append(f"missing required key: {key}")
    if issues and raise_on_issues:
        raise CaptionError("; ".join(issues))
    return issues


def model_caption(caption: dict) -> dict:
    """The caption as the image model expects it: only the schema root keys.

    Magic-prompt output can carry extra reasoning fields (notably ``aspect_ratio``,
    which our template uses to anchor bbox layout but the model doesn't consume —
    it warns on unknown root keys). The image dimensions are passed separately as
    width/height, so we drop anything outside the schema before generation.
    """
    return {k: caption[k] for k in REQUIRED_KEYS if k in caption}
