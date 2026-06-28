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
        import torch
        from ideogram4 import PRESETS
        from ideogram4.scheduler import get_schedule_for_resolution, make_step_intervals

        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset!r}. Choose from {sorted(PRESETS)}.")
        params = PRESETS[preset]
        num_steps = params.num_steps
        caption_dict, prompt = build_caption_prompt(caption)
        if seed is None:
            seed = random.randint(0, _SEED_MAX)

        pipe = self._pipe
        device = pipe.device
        prompts = [prompt]
        pipe._verify_prompts(prompts, raise_on_issues=False)

        t0 = time.time()
        # --- setup (mirrors Ideogram4Pipeline.__call__) ---
        schedule = get_schedule_for_resolution((height, width), known_mean=params.mu, std=params.std)
        step_intervals = make_step_intervals(num_steps).to(device)

        guidance_schedule = params.guidance_schedule
        if guidance_schedule is not None:
            gw_per_step = torch.as_tensor(guidance_schedule, dtype=torch.float32, device=device)
            if gw_per_step.shape != (num_steps,):
                raise ValueError(
                    f"guidance_schedule must have shape ({num_steps},), got {tuple(gw_per_step.shape)}")
        else:
            gw_per_step = torch.full((num_steps,), 7.0, dtype=torch.float32, device=device)

        inputs = pipe._build_inputs(prompts, height=height, width=width)
        batch_size = len(prompts)
        num_image_tokens = inputs["num_image_tokens"]
        grid_h, grid_w = inputs["grid_h"], inputs["grid_w"]
        max_text_tokens = inputs["max_text_tokens"]
        latent_dim = pipe.conditional_transformer.config.in_channels

        llm_features = pipe._encode_text(
            inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"])

        neg_position_ids = inputs["position_ids"][:, max_text_tokens:]
        neg_segment_ids = inputs["segment_ids"][:, max_text_tokens:]
        neg_indicator = inputs["indicator"][:, max_text_tokens:]
        neg_llm_features = torch.zeros(
            batch_size, num_image_tokens, llm_features.shape[-1],
            dtype=llm_features.dtype, device=device)

        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        z = torch.randn(
            batch_size, num_image_tokens, latent_dim,
            dtype=torch.float32, device=device, generator=generator)
        text_z_padding = torch.zeros(
            batch_size, max_text_tokens, latent_dim, dtype=torch.float32, device=device)

        # --- denoise loop (ours; progress fires natively) ---
        with torch.no_grad():
            for i in range(num_steps - 1, -1, -1):
                t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
                s_val = float(schedule(step_intervals[i].unsqueeze(0)).item())
                t = torch.full((batch_size,), t_val, dtype=torch.float32, device=device)

                pos_z = torch.cat([text_z_padding, z], dim=1)
                pos_out = pipe.conditional_transformer(
                    llm_features=llm_features, x=pos_z, t=t,
                    position_ids=inputs["position_ids"], segment_ids=inputs["segment_ids"],
                    indicator=inputs["indicator"])
                pos_v = pos_out[:, max_text_tokens:]

                neg_v = pipe.unconditional_transformer(
                    llm_features=neg_llm_features, x=z, t=t,
                    position_ids=neg_position_ids, segment_ids=neg_segment_ids,
                    indicator=neg_indicator)

                gw_i = gw_per_step[i]
                v = gw_i * pos_v + (1.0 - gw_i) * neg_v
                z = z + v * (s_val - t_val)
                if progress is not None:
                    progress(num_steps - i, num_steps)   # 1-based completed steps

        images = pipe._decode(z, grid_h=grid_h, grid_w=grid_w)
        return GenerationResult(
            image=images[0], seed=seed, width=width, height=height, preset=preset,
            caption=caption_dict, backend="torch", duration_s=time.time() - t0)
