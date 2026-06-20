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
- **`ig model stop-all`** stops every daemon.  `ig model list` shows running status.
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
# prints: queued job <id>

ig model jobs            # list all jobs (queued / running / done / failed)
ig model jobs <id>       # show details + sidecar path for one job
ig model clean           # remove completed job records older than 7 days
ig model clean --all     # remove all completed/failed records
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

Full parameter reference (every CLI flag, presets, and the model components —
Qwen3-VL text encoder, Flux2 VAE, transformers): **[docs/parameters.md](docs/parameters.md)**.

## Status

MLX backend (Apple Silicon via mflux) is implemented and validated end-to-end.
PyTorch/CUDA backend is pending Spark bring-up.
