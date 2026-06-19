# Parameters & Model Reference

How to read this: the **CLI parameters** are what you set; they map onto the
**generation parameters** the engine passes to the model. The **model components**
section explains the pieces (transformer, text encoder, VAE) you may have seen as
separate nodes in ComfyUI.

## CLI parameters

Commands: `ig gen`, `ig serve`, `ig model`, `ig platform`.

### `ig gen` — generate an image

| Parameter | Type / default | Meaning |
|---|---|---|
| `-p` / `--prompt` | str | Plain-text prompt to expand into a structured JSON caption. Optional if `--caption` is provided. |
| `--width` | int, `1024` | Output width in px. Rounded to a multiple of 16, floored at 256 (range 256–2048). |
| `--height` | int, `1024` | Output height in px. Rounded to a multiple of 16, floored at 256 (range 256–2048). |
| `--seed` | int, *random* | RNG seed. Same seed + caption + params ⇒ reproducible. Omit ⇒ random. |
| `--out` | path | Output image file. |
| `--model-path` | path | Path to the model weights dir (the `ideogram-4-fp8` checkpoint). Falls back to `IMAGEGEN_WEIGHTS_ROOT`. |
| `--backend` | `mlx` \| `torch`, *auto* | Inference backend. Auto-detected from the platform. |
| `MODEL` (positional) | str | Model to use (e.g., `ideogram4`). |

### `ig gen` — ideogram4 model options (after `--`)

| Option | Type / default | Meaning |
|---|---|---|
| `--preset` | `V4_DEFAULT_20` \| `V4_TURBO_12` \| `V4_QUALITY_48` | Sampler bundle — step count + guidance schedule + noise-schedule. See *Presets* below. |
| `--quantize` | `4` \| `8`, *none* | Quantize the fp8 weights to N bits on load. `8` = int8 (faster); `4` = 4-bit (fastest, ~½ transformer memory); default keeps fp8. |
| `--magic-model` | str, `"codex - gpt-5.5"` | Which LLM turns text → JSON caption (`codex - <model>` or `pi - <provider> - <model>`). |
| `--target-elements` | int, `0` (=auto) | Force ~N entries in `compositional_deconstruction.elements`. 0 lets the LLM choose. |
| `--caption` | path | Path to a prebuilt caption JSON file. If provided, `--prompt` is ignored. |

### `ig serve` — warm a worker on a Unix socket

| Parameter | Type / default | Meaning |
|---|---|---|
| `--socket` | path (required) | Unix socket the worker listens on. |
| `--log` | `debug` \| `info` \| `warning` \| `error`, *info* | Log level. |
| `--model-path` | path | Path to the model weights dir. Falls back to `IMAGEGEN_WEIGHTS_ROOT`. |
| `--backend` | `mlx` \| `torch`, *auto* | Inference backend. Auto-detected from the platform. |
| `MODEL` (positional) | str | Model to use (e.g., `ideogram4`). |

### `ig serve` — ideogram4 model options (after `--`)

Same as `ig gen` (see above): `--preset`, `--quantize`, `--magic-model`, `--target-elements`, `--caption`.

### `ig model` — inspect and configure models

| Subcommand | Arguments | Meaning |
|---|---|---|
| `ig model list` | — | List all available models. |
| `ig model show` | `<model>` | Show options, defaults, and path for a specific model (e.g., `ideogram4`). |
| `ig model set-path` | `<model> <path>` | Register the path to a model's weights directory. Persists to config. |

### `ig platform` — detect platform and default backend

No arguments. Prints the detected platform (Apple Silicon / Linux) and default backend (MLX / PyTorch).

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
