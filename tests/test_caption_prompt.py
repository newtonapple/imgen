import json

import pytest

from imgen.magic_prompt import _caption_prompt as cp

CAPTION = {
    "high_level_description": "a cat",
    "style_description": {},
    "compositional_deconstruction": {},
}


def test_build_instruction_has_prompt_and_aspect_ratio():
    instr = cp.build_instruction("a cat", width=576, height=1024)
    assert "a cat" in instr
    assert '"9:16"' in instr  # 576x1024 -> 9:16
    assert "576x1024" in instr


def test_build_instruction_target_elements_block():
    instr = cp.build_instruction("a cat", width=512, height=512, target_elements=5)
    assert "5" in instr and "elements" in instr.lower()


def test_extract_json_unwraps_fence():
    assert (
        json.loads(cp.extract_json("```json\n" + json.dumps(CAPTION) + "\n```"))[
            "high_level_description"
        ]
        == "a cat"
    )


def test_finalize_caption_sets_aspect_ratio():
    cap = cp.finalize_caption(json.dumps(CAPTION), width=576, height=1024)
    assert cap["aspect_ratio"] == "9:16"
    assert cap["high_level_description"] == "a cat"


def test_run_magic_prompt_retries_once_on_bad_json_then_succeeds():
    calls = []

    def call_model(_instruction):
        calls.append(1)
        return "not json" if len(calls) == 1 else json.dumps(CAPTION)

    cap = cp.run_magic_prompt(call_model, prompt="a cat", width=512, height=512)
    assert cap["high_level_description"] == "a cat"
    assert len(calls) == 2  # retried once


def test_run_magic_prompt_raises_after_second_bad_json():
    with pytest.raises(RuntimeError, match="invalid JSON"):
        cp.run_magic_prompt(lambda _i: "still not json", prompt="x", width=512, height=512)
