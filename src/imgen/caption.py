"""Light, dependency-free Ideogram-4 caption validation (works on both backends)."""

from __future__ import annotations

from typing import Any

REQUIRED_KEYS = ("high_level_description", "style_description", "compositional_deconstruction")


class CaptionError(ValueError):
    pass


def validate_caption(caption: dict[str, Any], *, raise_on_issues: bool = False) -> list[str]:
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


# Canonical key orders the Ideogram-4 caption verifier enforces (and the official
# pipeline reorders captions to before generation). Mirrored here so we stay
# dependency-free (the `ideogram4` package is CUDA-only; this module runs on MLX too).
_STYLE_ORDER_PHOTO = ("aesthetics", "lighting", "photo", "medium", "color_palette")
_STYLE_ORDER_NON_PHOTO = ("aesthetics", "lighting", "medium", "art_style", "color_palette")
_CD_ORDER = ("background", "elements")
_ELEMENT_ORDER_OBJ = ("type", "bbox", "desc", "color_palette")
_ELEMENT_ORDER_TEXT = ("type", "bbox", "text", "desc", "color_palette")


def _ordered(d: dict[str, Any], order: tuple[str, ...]) -> dict[str, Any]:
    """Return a copy of ``d`` with known keys in ``order``, unknown keys appended."""
    known = [k for k in order if k in d]
    extra = [k for k in d if k not in order]
    return {k: d[k] for k in (*known, *extra)}


def reorder_caption_keys(caption: dict[str, Any]) -> dict[str, Any]:
    """Reorder a caption's nested keys to the canonical schema order.

    Ideogram 4 was trained on captions in a fixed key order, and its
    ``CaptionVerifier`` enforces it; the official pipeline reorders every caption
    before generation. The LLM that writes our captions does not reliably emit
    that order, so we normalise ``style_description``, ``compositional_deconstruction``,
    and each element here. Returns a new dict; the input is not mutated.
    """
    if not isinstance(caption, dict):
        return caption
    out = dict(caption)

    sd = out.get("style_description")
    if isinstance(sd, dict) and (("photo" in sd) != ("art_style" in sd)):  # exactly one
        order = _STYLE_ORDER_PHOTO if "photo" in sd else _STYLE_ORDER_NON_PHOTO
        out["style_description"] = _ordered(sd, order)

    cd = out.get("compositional_deconstruction")
    if isinstance(cd, dict):
        cd = _ordered(cd, _CD_ORDER)
        elements = cd.get("elements")
        if isinstance(elements, list):
            cd["elements"] = [
                _ordered(
                    e, _ELEMENT_ORDER_TEXT if e.get("type") == "text" else _ELEMENT_ORDER_OBJ
                )
                if isinstance(e, dict) and e.get("type") in ("obj", "text")
                else e
                for e in elements
            ]
        out["compositional_deconstruction"] = cd

    return out


def model_caption(caption: dict[str, Any]) -> dict[str, Any]:
    """The caption as the image model expects it: only the schema root keys, in
    canonical key order.

    Magic-prompt output can carry extra reasoning fields (notably ``aspect_ratio``,
    which our template uses to anchor bbox layout but the model doesn't consume —
    it warns on unknown root keys). The image dimensions are passed separately as
    width/height, so we drop anything outside the schema before generation. We also
    reorder nested keys to the canonical order the model was trained on (see
    ``reorder_caption_keys``) — the LLM does not emit it reliably, and the official
    pipeline normalises it too.
    """
    return reorder_caption_keys({k: caption[k] for k in REQUIRED_KEYS if k in caption})
