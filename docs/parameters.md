# Parameters & Model Reference

How to read this: the **CLI parameters** are what you set; they map onto the
**generation parameters** the engine passes to the model. The **model components**
section explains the pieces (transformer, text encoder, VAE) you may have seen as
separate nodes in ComfyUI.

## CLI parameters

Subcommands: `platform`, `magic-models`, `magic-prompt`, `generate`, `run`, `serve`.

| Parameter | Subcommands | Type / default | Meaning |
|---|---|---|---|
| `prompt` (positional) | magic-prompt, run | str | The plain-language idea. The magic-prompt LLM expands it into the structured JSON caption. |
| `--width` / `--height` | magic-prompt, generate, run | int, `1024` | Output size in px. Rounded to a multiple of 16, floored at 256 (range 256–2048). Sets the aspect ratio the layout is composed for. |
| `--preset` | generate, run | `V4_DEFAULT_20` \| `V4_TURBO_12` \| `V4_QUALITY_48` | Sampler bundle — step count + guidance schedule + noise-schedule (`mu`/`std`). See *Presets* below. |
| `--seed` | generate, run | int, *random* | RNG seed. Same seed + caption + params ⇒ reproducible (and keeps composition stable when editing). Omit ⇒ random (a warning is printed, since re-seeding changes the image substantially). |
| `--magic-model` | magic-prompt, run, serve | str, `"codex - gpt-5.5"` | Which LLM turns text → JSON caption. Run `imagegen magic-models` to list choices (`codex - <model>` or `pi - <provider> - <model>`). |
| `--target-elements` | magic-prompt, run | int, `0` (=auto) | Force ~N entries in `compositional_deconstruction.elements`. 0 lets the LLM choose. |
| `--model-path` | generate, run, serve | path | Path to the model weights dir (the `ideogram-4-fp8` checkpoint). Falls back to `IMAGEGEN_WEIGHTS_ROOT`. |
| `--backend` | generate, run | `mlx` \| `torch`, *auto* | Inference backend. Auto-detected from the platform (MLX on Apple Silicon, PyTorch/CUDA on the Spark). |
| `--quantize` | generate, run, serve | `4` \| `8`, *none* | MLX only: quantize the fp8 weights to N bits on load. `8` = int8 (faster, ~same memory); `4` = 4-bit (faster, ~½ transformer memory, some quality loss); default keeps fp8. |
| `--out` | magic-prompt, generate, run | path | Output: image file (generate/run) or caption JSON (magic-prompt; defaults to stdout if omitted). |
| `--caption` | generate (required), run | path | generate: the input caption JSON. run: optional path to save the intermediate caption. |
| `--worker` | generate, run | socket path | Delegate the job to an already-running warm worker (see `serve`) instead of loading the model in-process. |
| `--socket` | serve (required) | path | Unix socket the worker listens on. |

## Generation parameters (what the presets/engine set)

A **preset** bundles the sampler settings so you don't tune them individually:

| Preset | Steps | Guidance (CFG) schedule | `mu` | `std` | Use for |
|---|---|---|---|---|---|
| `V4_TURBO_12` | 12 | 11 steps @ ~7, 1 polish @ ~3 | 0.5 | 1.75 | fast iteration / previews |
| `V4_DEFAULT_20` | 20 | 18 @ ~7, 2 polish @ ~3 | 0.0 | 1.75 | balanced (default) |
| `V4_QUALITY_48` | 48 | 45 @ ~7, 3 polish @ ~3 | 0.0 | 1.5 | final / high-fidelity |

- **steps** — denoising iterations; more = higher quality, slower (turbo ≈ ¼ the time of quality).
- **guidance (CFG)** — per-step prompt-adherence strength; high for most steps, a few low-guidance "polish" steps at the end.
- **`mu` / `std`** — shape of the flow-matching noise schedule (where denoising effort is concentrated).

The image **dimensions** (`--width`/`--height`) are passed to the model directly — they are *not* part of the caption (the caption's `aspect_ratio` field, if present, is dropped before generation).

## Magic-prompt (text → JSON caption)

`magic-prompt` / the first half of `run` call an LLM to convert your prompt into the
**Ideogram-4 structured caption** (the only schema the model accepts):

- `high_level_description` — one-sentence scene summary.
- `style_description` — aesthetics, lighting, medium, `photo`/`art_style`, color palette.
- `compositional_deconstruction` — `background` + `elements[]` (each with `type`, `bbox`,
  `desc`, optional palette). `--target-elements` influences how many elements.

Provider is chosen by `--magic-model` (codex CLI or pi CLI today; OpenRouter/local vLLM are future backends).

## Model components (the pieces, incl. "Qwen" and "VAE")

Ideogram-4 is not one file — it's several components. In ComfyUI you load them as
separate nodes; in this pipeline they all live inside the **one** `ideogram-4-fp8/`
directory and load together:

| Component | ComfyUI file / our subdir | What it does |
|---|---|---|
| **Text encoder** | `qwen3vl_8b_fp8_scaled` / `text_encoder/` + `tokenizer/` | **Qwen3-VL-8B** (vision tower stripped) — encodes the JSON caption into the conditioning the diffusion model follows. This is the "Qwen" piece; it is **not** a VAE. |
| **Diffusion transformer (conditional)** | `ideogram4_fp8_scaled` / `transformer/` | The DiT that denoises the latent, guided by the text conditioning. |
| **Diffusion transformer (unconditional)** | `ideogram4_unconditional_fp8_scaled` / `unconditional_transformer/` | Second DiT for **asymmetric classifier-free guidance** (the unconditional pass drops text tokens — that's the "guidance" knob, not a negative prompt). |
| **VAE** | `flux2-vae` / `vae/` | **Flux.2-architecture VAE** — decodes the denoised latent into the final pixel image (and defines the latent space the diffusion runs in). |
| **Scheduler** | `Ideogram4Scheduler` / `scheduler/` | Flow-matching noise schedule (the `mu`/`std` from the preset). |

So the two things you saw: **Qwen3-VL = the text/prompt encoder**, **Flux2 VAE = the image decoder** — different jobs, both required.

## mflux knobs we don't expose (yet)

The MLX backend (mflux) also supports `--num-inference-steps` / `--guidance` (override
the preset), `--lora-paths` / `--lora-scales`, and `--low-ram` / `--mlx-cache-limit-gb`.
We can surface any of these if needed; the presets are the intended interface.
