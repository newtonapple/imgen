# imagegen

Pure-Python pipeline for **Ideogram 4** image generation:
plain prompt → magic-prompt → structured JSON caption → image, with edit/regenerate.

Built to replace a ComfyUI workflow with our own code. Pluggable inference
backends, selected per platform:

| Platform | Backend | How |
| --- | --- | --- |
| Apple Silicon (dev/test) | **MLX** | [mflux](https://github.com/filipstrand/mflux) |
| DGX Spark (prod, CUDA) | **PyTorch** | official [`ideogram4`](https://github.com/ideogram-oss/ideogram4) pipeline |

## Architecture

A factory takes a **model** (weights on disk) + an **inference backend** and
builds a warm, load-once **inference pipeline** (`ImageEngine`):

```python
from imagegen import ModelSpec, create_pipeline

model = ModelSpec.from_path("/Volumes/PRO-G40/data/models/image-gen/ideogram-4-fp8")
engine = create_pipeline(model)          # backend auto-detected (MLX here)
result = engine.generate(caption, width=1024, height=1024, preset="V4_DEFAULT_20")
result.image.save("out.png")
```

The same engine instance is what a worker holds warm and calls per job; the CLI
is just one caller.

## Weights

Model weights are **never committed and never live in iCloud** — they sit on an
external volume / the Spark and are referenced by path (`ModelSpec.path`, or the
`IMAGEGEN_WEIGHTS_ROOT` env var).

## Install

`make install` creates the virtualenv at `~/.venvs/imagegen` (kept **outside**
this iCloud-synced repo, so uv clones packages from its global cache with no byte
duplication and iCloud never syncs it; override with `IMAGEGEN_VENV`) and installs
the package (editable) plus the platform backend extra — which also installs the
`ig` CLI:

```bash
make install    # ~/.venvs/imagegen + imagegen[mlx] on Apple Silicon, imagegen[cuda] on Linux
make test       # unit tests (no weights needed)
make fmt        # format the code in place (ruff format)
make style      # format in place, then lint (ruff format + ruff check)
make lint       # ruff check + ruff format --check + mypy (no changes)
make platform   # print detected platform + default backend
make clean      # remove the venv
```

To use the library/CLI directly (the `ig` console script ships with the
package via `[project.scripts]`):

```bash
pip install ".[mlx]"     # Apple Silicon (MLX backend)
pip install ".[cuda]"    # Linux / DGX Spark (PyTorch backend)
ig --help
```

## CLI Usage

```bash
# one-time: tell ig where the weights are
ig ideogram4 config set weights-path /Volumes/PRO-G40/data/models/image-gen/ideogram-4-fp8

# generate an image — the warm daemon is auto-started if it isn't running
# progress streams to stderr; metadata JSON is printed to stdout
ig ideogram4 gen -p "a ginger cat wizard" -w 768 -h 768 --seed 42 -o out.png \
   --preset V4_DEFAULT_20
# → writes out.png  +  out.png.json  (includes caption, prompt, seed, duration, …)

# from a prebuilt caption
ig ideogram4 gen -o out.png --caption caption.json

# daemon lifecycle (one daemon per model, one job at a time)
ig ideogram4 serve             # foreground (Ctrl-C to stop)
ig ideogram4 serve --detach    # background (alias: -d)
ig ideogram4 stop              # stop this model's daemon
ig model stop-all              # stop every running daemon

# list / inspect models and daemon status
ig model list
ig model show ideogram4

ig platform
```

### Warm-daemon model

`ig <model> gen` always routes through a **warm daemon** — a background process that
loads the model once and stays resident, accepting one job at a time.  Concurrent
`gen` calls queue (the daemon serialises them).

- **Auto-start** — if no daemon is running, `gen` spawns one and waits up to
  15 minutes for the model to load before sending the job.
- **`ig <model> serve`** starts the daemon in the foreground; `--detach` / `-d`
  backgrounds it.  Starting a second `serve` while one is already running prints an
  error and tells you to run `ig <model> stop` first.
- **`ig <model> stop`** sends SIGTERM and cleans up the registry entry.
- **`ig model stop-all`** stops every daemon.  `ig model list` shows running status
  (`idle` / `busy`, or `crashed` with a log path if a daemon died mid-job).
- **Sidecar** — after each generation `gen` writes `<out>.json` next to the image,
  containing: `caption` (the magic-prompt structured JSON), `prompt`, `model`,
  `seed`, `width`, `height`, `preset`, `backend`, `duration_s`, `out`.  The same
  object is printed to stdout as JSON.

**Runtime directory** (`IG_RUNTIME_DIR`, default `~/.cache/ig`) holds the daemon
socket, per-model JSON registry file, and log file.  Set `IG_RUNTIME_DIR` to a
shorter path if the default exceeds the platform's Unix-socket path limit
(104 bytes on macOS, 108 on Linux).

**Background (async) generation** — pass `--queue` to fire-and-forget without
blocking:

```bash
ig ideogram4 gen -p "a cat" -w 768 -h 768 -o out.png --queue
# prints: job <id> → out.png   (poll: ig model jobs <id>)

ig model jobs            # list all jobs (queued / running / done / failed)
ig model jobs <id>       # show one job's record + its log path
ig model clean           # remove finished-job records/logs + dead-daemon logs
ig model clean --all     # also truncate logs of running daemons
ig model clean --older-than 7   # only prune jobs finished > 7 days ago
```

### Model & precision

There is **one model directory** — the official `ideogram-ai/ideogram-4-fp8` checkpoint
(it serves both the Mac via mflux and the Spark via the PyTorch pipeline). Precision
variants come from quantizing it **on load**, configured via `ig ideogram4 config set quantize`:

- *(default)* — native fp8
- `quantize 8` — int8 (equivalent to the MLXBits "q8" build)
- `quantize 4` — 4-bit (smallest/fastest)

```bash
ig ideogram4 config set quantize 8
```

> The pre-converted `MLXBits/ideogram-4-mlx*` builds are **not** loadable by released
> mflux (their flat-layout loader lives only in an unmerged PR) — use fp8 + `quantize`.

### Magic-prompt provider & model

Build config (magic-prompt provider, model, backend, weights path) is set once via
`ig ideogram4 config set` — not as per-gen flags.

```bash
# free OpenRouter pool (rotating) — store key + persist as default
ig ideogram4 config set-key openrouter sk-or-...
ig ideogram4 config set magic-provider openrouter
ig ideogram4 config set magic-model openrouter/free

# thereafter the default is openrouter/free — no flags needed at gen time
ig ideogram4 gen -p "a cat" -w 768 -h 768 -o out.png
```

- **CLI providers** (`codex`, `claude`, `pi`) shell out to a local CLI and need **no API key**.
- **HTTP providers** (`openai`, `anthropic`, `openrouter`) need a key stored with
  `ig ideogram4 config set-key <provider> <key>` (written to `~/.config/ig/secrets.toml`, mode 600, never committed).
- `openrouter` `magic-model` accepts a comma-separated list (e.g. `a/x,b/y`) → an OpenRouter `models[]` fallback chain. `pi` takes `"<pi-provider>/<model>"`.
- With nothing configured the default is `codex` + `gpt-5.5`.

Run `ig model show ideogram4` to see all available config keys and gen options.
The full parameter, preset, and model-component reference follows below.

## Status

MLX backend (Apple Silicon via mflux) is implemented and validated end-to-end.
PyTorch/CUDA backend is pending Spark bring-up.

---

## Parameter & model reference

How to read this: the **CLI parameters** are what you set; they map onto the
**generation parameters** the engine passes to the model. The **model components**
section explains the pieces (transformer, text encoder, VAE) you may have seen as
separate nodes in ComfyUI.

The CLI uses a **model-first grammar**: `ig <model> <action> [options]`.
Build config (weights path, backend, quantize, magic-prompt provider/model) is set
once with `ig <model> config set` and persisted to `~/.config/ig/config.toml`.
Per-request options (prompt, width, height, seed, preset, caption) are passed at
gen time.

### CLI commands

#### `ig <model> gen` — generate an image

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
| `--queue` / `-q` | flag, off | Run in the background; returns a job id to poll with `ig model jobs`. |

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

#### `ig <model> config` — get/set persisted build config

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

#### `ig <model> serve` — start the warm daemon

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

#### `ig <model> stop` — stop the daemon

No arguments.  Sends SIGTERM to the daemon and cleans up the registry.  Prints a
message whether or not a daemon was running.

#### `ig model` — inspect models and manage daemons

| Subcommand | Arguments | Meaning |
|---|---|---|
| `ig model list` | — | List all available models with their daemon status (PID + `idle`/`busy`, or `crashed`). |
| `ig model show` | `<model>` | Show options, defaults, and config keys for a specific model. |
| `ig model jobs` | `[<id>]` | List background (`--queue`) jobs, or show one job by id (with its log path). |
| `ig model clean` | `[--all] [--older-than DAYS]` | Remove finished-job records/logs and dead-daemon logs. `--all` also truncates running daemons' logs; `--older-than` only prunes jobs finished more than N days ago. |
| `ig model stop-all` | — | Stop every running daemon. |

#### `ig platform` — detect platform and default backend

No arguments. Prints the detected platform (Apple Silicon / Linux) and default backend (MLX / PyTorch).

### Daemon runtime

`gen`, `serve`, and `stop` all use a shared **runtime directory** (default
`~/.cache/ig`, override with `IG_RUNTIME_DIR`) that holds:

| Path | Purpose |
|---|---|
| `$IG_RUNTIME_DIR/daemons/<model>.sock` | Unix domain socket the daemon listens on. |
| `$IG_RUNTIME_DIR/daemons/<model>.json` | Live registry record (PID, socket path, backend, state). |
| `$IG_RUNTIME_DIR/logs/<model>.log` | Stdout/stderr of the detached daemon process. |
| `$IG_RUNTIME_DIR/jobs/<id>.json` + `<id>.log` | Per-job record + log for `--queue` background jobs. |

There is **one daemon per model** (one GPU → one job at a time); concurrent `gen`
calls queue automatically.  If the socket path would exceed the platform limit
(104 bytes on macOS, 108 on Linux) `ig` exits with an error — set `IG_RUNTIME_DIR`
to a shorter path in that case.

**Background jobs:** `ig <model> gen --queue` spawns a detached copy of the
wait-mode `gen` (the socket backlog is the queue — no daemon-side scheduler). The
detached process streams progress into `$IG_RUNTIME_DIR/jobs/<id>.log`, writes the
same `<out>.json` sidecar when done, and updates `$IG_RUNTIME_DIR/jobs/<id>.json`
(`queued` → `running` → `done`/`failed`). `ig model jobs [<id>]` reads those
records; `ig model clean` prunes finished ones.

### Generation parameters (what the presets/engine set)

A **preset** bundles the sampler settings so you don't tune them individually:

| Preset | Steps | Guidance (CFG) schedule | `mu` | `std` | Use for |
|---|---|---|---|---|---|
| `V4_TURBO_12` | 12 | 1 step @ 3.0, then 11 @ 7.0 | 0.5 | 1.75 | fast iteration / previews |
| `V4_DEFAULT_20` | 20 | 2 steps @ 3.0, then 18 @ 7.0 | 0.0 | 1.75 | balanced (default) |
| `V4_QUALITY_48` | 48 | 3 steps @ 3.0, then 45 @ 7.0 | 0.0 | 1.5 | final / high-fidelity |

- **steps** — denoising iterations; more = higher quality, slower (turbo ≈ ¼ the time of quality).
- **guidance (CFG)** — per-step prompt-adherence strength. Each preset starts with a
  few **low-guidance (3.0)** steps so the global composition settles naturally, then
  clamps to **strong guidance (7.0)** for the rest to lock in prompt fidelity and
  detail. Higher is not "better": past the sweet spot guidance over-saturates and
  distorts, which is why the steady-state sits at 7.0.
- **`mu` / `std`** — shape of the flow-matching noise schedule (where denoising effort is concentrated).

The image **dimensions** (`-w`/`-h`) are passed to the model directly — they are *not* part of the caption (the caption's `aspect_ratio` field, if present, is dropped before generation).

### Magic-prompt (text → JSON caption)

`magic-prompt` / the first half of `run` call an LLM to convert your prompt into the
**Ideogram-4 structured caption** (the only schema the model accepts):

- `high_level_description` — one-sentence scene summary.
- `style_description` — aesthetics, lighting, medium, `photo`/`art_style`, color palette.
- `compositional_deconstruction` — `background` + `elements[]` (each with `type`, `bbox`,
  `desc`, optional palette). `--target-elements` influences how many elements.

#### Providers

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

### Model components (the pieces, incl. "Qwen" and "VAE")

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

### mflux knobs we don't expose (yet)

The MLX backend (mflux) also supports `--num-inference-steps` / `--guidance` (override
the preset), `--lora-paths` / `--lora-scales`, and `--low-ram` / `--mlx-cache-limit-gb`.
We can surface any of these if needed; the presets are the intended interface.
