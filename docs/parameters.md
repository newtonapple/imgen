# Parameters & Model Reference

How to read this: the **CLI parameters** are what you set; they map onto the
**generation parameters** the engine passes to the model. The **model components**
section explains the pieces (transformer, text encoder, VAE) you may have seen as
separate nodes in ComfyUI.

The CLI uses a **model-first grammar**: `ig <model> <action> [options]`.
Build config (weights path, backend, quantize, magic-prompt provider/model) is set
once with `ig <model> config set` and persisted to `~/.config/ig/config.toml`.
Per-request options (prompt, width, height, seed, preset, caption) are passed at
gen time.

## CLI commands

### `ig <model> gen` — generate an image

Example: `ig ideogram4 gen -p "a cat" -w 768 -h 768 --seed 42 -o out.png --preset V4_DEFAULT_20`

`gen` routes the request through the **warm daemon** (auto-started if not running).
Generation progress is streamed to **stderr**; on completion the metadata JSON is
printed to **stdout** and also written as a sidecar file at `<out>.json`.

| Parameter | Type / default | Meaning |
|---|---|---|
| `-p` / `--prompt` | str | Plain-text prompt to expand into a structured JSON caption. Optional if `--caption` is provided. |
| `-w` / `--width` | int, `1024` | Output width in px. Rounded to a multiple of 16, floored at 256 (range 256–2048). |
| `-h` / `--height` | int, `1024` | Output height in px. Rounded to a multiple of 16, floored at 256 (range 256–2048). |
| `--seed` | int, *random* | RNG seed. Same seed + caption + params ⇒ reproducible. Omit ⇒ random. |
| `-o` / `--out` | path (required) | Output image file. |
| `--preset` | `V4_DEFAULT_20` \| `V4_TURBO_12` \| `V4_QUALITY_48` | Sampler bundle — step count + guidance schedule + noise-schedule. See *Presets* below. |
| `--target-elements` | int, `0` (=auto) | Force ~N entries in `compositional_deconstruction.elements`. 0 lets the LLM choose. |
| `--caption` | path | Path to a prebuilt caption JSON file. If provided, `--prompt` is ignored. |

**Output sidecar** (`<out>.json`) and stdout JSON fields:

| Field | Meaning |
|---|---|
| `caption` | The full magic-prompt structured JSON caption sent to the model. |
| `prompt` | The original plain-text prompt (null when `--caption` was used). |
| `model` | Model name (e.g. `ideogram4`). |
| `seed` | Actual seed used (useful when you did not specify `--seed`). |
| `width` / `height` | Actual image dimensions in px. |
| `preset` | Sampler preset used. |
| `backend` | Inference backend (`mlx` or `torch`). |
| `duration_s` | Total generation time in seconds. |
| `out` | Path to the output image file. |

### `ig <model> config` — get/set persisted build config

Example: `ig ideogram4 config set magic-provider openrouter`

| Subcommand | Arguments | Meaning |
|---|---|---|
| `ig <model> config set` | `<key> <value>` | Set a build config value (persists to `~/.config/ig/config.toml`). |
| `ig <model> config set-key` | `<provider> <api-key>` | Store an API key in `~/.config/ig/secrets.toml` (mode 600, never committed). |
| `ig <model> config show` | — | Print the current config for this model. |

**Config keys for `ideogram4`:**

| Key | Values | Meaning |
|---|---|---|
| `weights-path` | path | Path to the model weights directory (the `ideogram-4-fp8` checkpoint). |
| `backend` | `mlx` \| `torch` | Inference backend. Auto-detected from the platform if not set. |
| `quantize` | `4` \| `8` \| *(empty to clear)* | Quantize the fp8 weights to N bits on load. `8` = int8; `4` = 4-bit; default keeps fp8. |
| `magic-provider` | `codex` \| `claude` \| `pi` \| `openai` \| `anthropic` \| `openrouter` | Which provider turns text → JSON caption. |
| `magic-model` | str | Model id for the chosen provider. |

### `ig <model> serve` — start the warm daemon

Examples:
```
ig ideogram4 serve           # foreground (Ctrl-C to stop)
ig ideogram4 serve --detach  # background (alias: -d)
```

| Parameter | Type / default | Meaning |
|---|---|---|
| `--detach` / `-d` | flag, off | Run the daemon in the background (returns immediately). |

Build config (backend, weights path, quantize, magic-prompt provider) is read from
`ig ideogram4 config` — no per-serve flags needed.  The socket, registry, and log
are placed under `IG_RUNTIME_DIR` (default `~/.cache/ig`) and are managed
automatically.

If a daemon for that model is already running, `serve` exits with an error and
shows the PID.  Run `ig <model> stop` before starting a new one.

### `ig <model> stop` — stop the daemon

No arguments.  Sends SIGTERM to the daemon and cleans up the registry.  Prints a
message whether or not a daemon was running.

### `ig model` — inspect models and manage daemons

| Subcommand | Arguments | Meaning |
|---|---|---|
| `ig model list` | — | List all available models with their daemon status (PID if running). |
| `ig model show` | `<model>` | Show options, defaults, and config keys for a specific model. |
| `ig model stop-all` | — | Stop every running daemon. |

### `ig platform` — detect platform and default backend

No arguments. Prints the detected platform (Apple Silicon / Linux) and default backend (MLX / PyTorch).

## Daemon runtime

`gen`, `serve`, and `stop` all use a shared **runtime directory** (default
`~/.cache/ig`, override with `IG_RUNTIME_DIR`) that holds:

| Path | Purpose |
|---|---|
| `$IG_RUNTIME_DIR/daemons/<model>.sock` | Unix domain socket the daemon listens on. |
| `$IG_RUNTIME_DIR/daemons/<model>.json` | Live registry record (PID, socket path, backend, state). |
| `$IG_RUNTIME_DIR/logs/<model>.log` | Stdout/stderr of the detached daemon process. |

There is **one daemon per model** (one GPU → one job at a time); concurrent `gen`
calls queue automatically.  If the socket path would exceed the platform limit
(104 bytes on macOS, 108 on Linux) `ig` exits with an error — set `IG_RUNTIME_DIR`
to a shorter path in that case.

> **Coming in Phase 2b** (not yet available): `ig <model> gen --queue` (async
> fire-and-forget), `ig model jobs` (list pending/running jobs), `ig model clean`
> (purge stale job data).

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

The image **dimensions** (`-w`/`-h`) are passed to the model directly — they are *not* part of the caption (the caption's `aspect_ratio` field, if present, is dropped before generation).

## Magic-prompt (text → JSON caption)

`magic-prompt` / the first half of `run` call an LLM to convert your prompt into the
**Ideogram-4 structured caption** (the only schema the model accepts):

- `high_level_description` — one-sentence scene summary.
- `style_description` — aesthetics, lighting, medium, `photo`/`art_style`, color palette.
- `compositional_deconstruction` — `background` + `elements[]` (each with `type`, `bbox`,
  `desc`, optional palette). `--target-elements` influences how many elements.

### Providers

Two config keys select the provider and model: `magic-provider` and `magic-model`.

| Family | Providers | Key required? | Key source |
|---|---|---|---|
| CLI (shell out) | `codex`, `claude`, `pi` | No | — |
| HTTP | `openai`, `anthropic`, `openrouter` | Yes | `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY` env var, or stored with `ig ideogram4 config set-key <provider> <key>` |

The default provider/model is `codex` + `gpt-5.5` when nothing is configured.

Use `ig ideogram4 config set magic-provider <p>` and `ig ideogram4 config set magic-model <m>`
to persist a new default. Use `ig ideogram4 config set-key <provider> <key>` to store an
HTTP API key in `~/.config/ig/secrets.toml` (mode 600, never committed to version control).

**`openrouter`** accepts a comma-separated model list for `magic-model` (e.g. `openrouter/free,openai/gpt-4o`) — this becomes an OpenRouter `models[]` fallback chain.

**`pi`** takes `magic-model "<pi-provider>/<model>"` (reads `~/.pi/agent/models.json`; override with `PI_MODELS_JSON` env var).

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
