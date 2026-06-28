"""PyTorch backend (CUDA) via the official `ideogram4` package.

Loads the Ideogram-4 pipeline once (warm) from a local snapshot directory and
generates one image per `generate` call. We use the package's own loader because
it handles both the nf4 (bitsandbytes) and fp8 builds correctly; diffusers'
loader is NOT used — its bnb path mishandles meta tensors on the CUDA 13 /
torch 2.12 stack ("Cannot copy out of meta tensor"), and it can't load the
custom fp8 build at all.

The package's `from_pretrained` only accepts a HF repo id (it calls
`hf_hub_download`). To load our pre-downloaded LOCAL weights we set
``weights_repo`` to the local directory and resolve `hf_hub_download` against
that directory (no network, no HF cache) — `transformers`' AutoTokenizer /
AutoModel already accept the local dir + subfolder. See
`_patch_local_hub_download`.

All heavy imports (torch, ideogram4) are lazy (inside `__init__`/`generate`) so
importing this module — or the factory — never pulls torch into the Mac/MLX env.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..caption import model_caption, validate_caption
from ..config import ModelSpec
from .base import GenerationResult

_SEED_MAX = 2**31 - 1

# Preset names exposed to the CLI; the actual sampler params come from the
# package's own `PRESETS` table at generate time (its own guidance order).
PRESET_NAMES = ("V4_TURBO_12", "V4_DEFAULT_20", "V4_QUALITY_48")


def _patch_local_hub_download() -> None:
    """Make the `ideogram4` package resolve `hf_hub_download` against a local dir.

    ``weights_repo`` is set to a local path, but the package still calls
    `hf_hub_download(repo_id, filename)` for the transformer / vae / text-encoder
    config. We return the local file when it exists, and raise
    `EntryNotFoundError` (NOT a network call) when it doesn't, so the package's
    single-file fallback works. Idempotent; non-dir repo ids fall back to the
    real downloader (online mode for users who didn't pre-download).
    """
    import ideogram4.pipeline_ideogram4 as pkg
    from huggingface_hub.errors import EntryNotFoundError

    if getattr(pkg.hf_hub_download, "_imgen_local", False):
        return
    orig = pkg.hf_hub_download

    def _local(repo_id: str, filename: str, **kwargs: Any) -> str:
        local = Path(repo_id) / filename
        if local.exists():
            return str(local)
        if Path(repo_id).is_dir():
            raise EntryNotFoundError(f"{filename} not found in local weights dir {repo_id}")
        return str(orig(repo_id=repo_id, filename=filename, **kwargs))  # online fallback

    _local._imgen_local = True  # type: ignore[attr-defined]
    pkg.hf_hub_download = _local


def build_caption_prompt(caption: dict[str, Any] | str) -> tuple[dict[str, Any], str]:
    """Torch-free: validate (warn-not-fail) and return (caption_dict, prompt JSON).

    The prompt is the schema-root caption (drops non-schema keys like aspect_ratio),
    matching the MLX backend. Kept pure so it is unit-testable without importing
    torch/ideogram4.
    """
    caption_dict = caption if isinstance(caption, dict) else json.loads(caption)
    validate_caption(caption_dict)  # warn-not-fail (same contract as MLX)
    # ensure_ascii=False keeps non-Latin text (e.g. Japanese) as literal UTF-8;
    # the package's caption verifier warns on \uXXXX escapes, and the model
    # tokenizes literal UTF-8 better.
    return caption_dict, json.dumps(model_caption(caption_dict), ensure_ascii=False)


class TorchEngine:
    backend = "torch"

    def __init__(
        self,
        model: ModelSpec,
        device: str | None = None,
        *,
        quantize: int | None = None,  # MLX-only on-load concept; accepted and ignored
        **options: Any,
    ) -> None:
        self.model = model
        self.device = device or "cuda"
        self.options = options
        # Lazy heavy imports — keep torch/ideogram4 out of the Mac/MLX env.
        import torch

        _patch_local_hub_download()
        from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig

        try:
            self._pipe = Ideogram4Pipeline.from_pretrained(
                config=Ideogram4PipelineConfig(weights_repo=str(model.path)),
                device=self.device,
                dtype=torch.bfloat16,
            )
        except Exception as exc:  # noqa: BLE001 — surface a clear, actionable error
            raise RuntimeError(
                f"Could not load an Ideogram-4 pipeline from {model.path}: {exc}\n"
                "The torch backend needs an Ideogram-4 snapshot dir (model_index.json + "
                "transformer/ + text_encoder/ + tokenizer/ + vae/), e.g. the nf4 or fp8 "
                "build. nf4 loads via bitsandbytes."
            ) from exc

    def generate(
        self,
        caption: dict[str, Any] | str,
        *,
        width: int,
        height: int,
        preset: str = "V4_DEFAULT_20",
        seed: int | None = None,
        progress: "Callable[[int, int], None] | None" = None,
    ) -> GenerationResult:
        from ideogram4 import PRESETS

        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset!r}. Choose from {sorted(PRESETS)}.")
        caption_dict, prompt = build_caption_prompt(caption)
        if seed is None:
            seed = random.randint(0, _SEED_MAX)
        params = PRESETS[preset]
        cb = (lambda done, total: progress(done, total)) if progress is not None else None
        t0 = time.time()
        images = self._pipe(
            prompt,
            height=height,
            width=width,
            num_steps=params.num_steps,
            guidance_schedule=params.guidance_schedule,
            mu=params.mu,
            std=params.std,
            seed=seed,
            raise_on_caption_issues=False,  # warn-not-fail; magic-prompt output conforms
            callback=cb,
        )
        return GenerationResult(
            image=images[0],
            seed=seed,
            width=width,
            height=height,
            preset=preset,
            caption=caption_dict,
            backend="torch",
            duration_s=time.time() - t0,
        )
