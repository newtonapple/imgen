"""CLI entrypoint.

Subcommands:
  platform      – report detected platform + default backend.
  magic-prompt  – expand a text prompt to a structured caption JSON.
  generate      – generate an image from a pre-built caption JSON file.
  run           – magic-prompt + generate in one shot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .platform import platform_summary


# ---------------------------------------------------------------------------
# Hookable factory functions (monkeypatched in tests)
# ---------------------------------------------------------------------------

def _build_provider(model: str):
    """Build a MagicPromptProvider for the given model string."""
    from .magic_prompt.cli_provider import CliMagicPromptProvider

    return CliMagicPromptProvider(model=model)


def _build_engine(model_path: str | None):
    """Build an ImageEngine from a model path (or IMAGEGEN_WEIGHTS_ROOT)."""
    import os

    from .config import ModelSpec
    from .engine.factory import create_pipeline

    if model_path:
        spec = ModelSpec.from_path(model_path)
    else:
        root = os.environ.get("IMAGEGEN_WEIGHTS_ROOT")
        if not root:
            raise RuntimeError(
                "Provide --model-path or set IMAGEGEN_WEIGHTS_ROOT."
            )
        spec = ModelSpec.from_path(root)
    return create_pipeline(spec)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imagegen")
    sub = parser.add_subparsers(dest="cmd")

    # -- platform ------------------------------------------------------------
    sub.add_parser("platform", help="show detected platform + default backend")

    # -- magic-prompt --------------------------------------------------------
    mp = sub.add_parser("magic-prompt", help="expand a prompt to a caption JSON")
    mp.add_argument("prompt", help="the text prompt to expand")
    mp.add_argument("--width", type=int, default=1024, help="canvas width (default 1024)")
    mp.add_argument("--height", type=int, default=1024, help="canvas height (default 1024)")
    mp.add_argument(
        "--target-elements",
        type=int,
        default=0,
        dest="target_elements",
        help="target element count for compositional_deconstruction (0=auto)",
    )
    mp.add_argument(
        "--model",
        default="codex - gpt-5.5",
        help="magic-prompt model string (default: 'codex - gpt-5.5')",
    )
    mp.add_argument(
        "--out",
        default=None,
        help="write caption JSON to this file (default: print to stdout)",
    )

    # -- generate ------------------------------------------------------------
    gen = sub.add_parser("generate", help="generate an image from a caption JSON file")
    gen.add_argument(
        "--caption",
        required=True,
        help="path to caption JSON file",
    )
    gen.add_argument("--width", type=int, default=1024)
    gen.add_argument("--height", type=int, default=1024)
    gen.add_argument("--preset", default="V4_DEFAULT_20")
    gen.add_argument("--seed", type=int, default=None, help="RNG seed (omit = random)")
    gen.add_argument(
        "--model-path",
        default=None,
        dest="model_path",
        help="path to model weights (overrides IMAGEGEN_WEIGHTS_ROOT)",
    )
    gen.add_argument("--out", required=True, help="output image path")

    # -- run -----------------------------------------------------------------
    run = sub.add_parser("run", help="magic-prompt + generate in one shot")
    run.add_argument("prompt", help="the text prompt")
    run.add_argument("--width", type=int, default=1024)
    run.add_argument("--height", type=int, default=1024)
    run.add_argument("--preset", default="V4_DEFAULT_20")
    run.add_argument("--seed", type=int, default=None, help="RNG seed (omit = random)")
    run.add_argument(
        "--target-elements",
        type=int,
        default=0,
        dest="target_elements",
    )
    run.add_argument(
        "--model",
        default="codex - gpt-5.5",
        help="magic-prompt model string",
    )
    run.add_argument(
        "--model-path",
        default=None,
        dest="model_path",
        help="path to model weights (overrides IMAGEGEN_WEIGHTS_ROOT)",
    )
    run.add_argument("--out", required=True, help="output image path")
    run.add_argument(
        "--caption",
        default=None,
        help="save intermediate caption JSON to this path",
    )

    return parser


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ---- platform ----------------------------------------------------------
    if args.cmd == "platform":
        json.dump(platform_summary(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    # ---- magic-prompt ------------------------------------------------------
    if args.cmd == "magic-prompt":
        provider = _build_provider(args.model)
        caption = provider.expand(
            args.prompt,
            width=args.width,
            height=args.height,
            target_elements=args.target_elements,
        )
        caption_json = json.dumps(caption, indent=2)
        if args.out:
            Path(args.out).write_text(caption_json)
        else:
            sys.stdout.write(caption_json)
            sys.stdout.write("\n")
        return 0

    # ---- generate ----------------------------------------------------------
    if args.cmd == "generate":
        if args.seed is None:
            sys.stderr.write(
                "warning: no --seed given; re-seeding will likely change the image substantially\n"
            )
        caption = json.loads(Path(args.caption).read_text())
        engine = _build_engine(args.model_path)
        result = engine.generate(
            caption,
            width=args.width,
            height=args.height,
            preset=args.preset,
            seed=args.seed,
        )
        result.image.save(args.out)
        print(
            json.dumps(
                {
                    "seed": result.seed,
                    "width": result.width,
                    "height": result.height,
                    "preset": result.preset,
                    "backend": result.backend,
                    "duration_s": result.duration_s,
                    "out": args.out,
                },
                indent=2,
            )
        )
        return 0

    # ---- run ---------------------------------------------------------------
    if args.cmd == "run":
        if args.seed is None:
            sys.stderr.write(
                "warning: no --seed given; re-seeding will likely change the image substantially\n"
            )
        from .pipeline import Pipeline

        provider = _build_provider(args.model)
        engine = _build_engine(args.model_path)
        pipeline = Pipeline(engine=engine, magic_prompt=provider)
        result = pipeline.run(
            args.prompt,
            width=args.width,
            height=args.height,
            preset=args.preset,
            seed=args.seed,
            target_elements=args.target_elements,
        )
        if args.caption:
            Path(args.caption).write_text(json.dumps(result.caption, indent=2))
        result.image.save(args.out)
        print(
            json.dumps(
                {
                    "seed": result.seed,
                    "width": result.width,
                    "height": result.height,
                    "preset": result.preset,
                    "backend": result.backend,
                    "duration_s": result.duration_s,
                    "out": args.out,
                },
                indent=2,
            )
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
