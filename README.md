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

model = ModelSpec.from_path("/Volumes/PRO-G40/data/models/image-gen/ideogram-4-mlx-q8")
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
`imagegen` CLI:

```bash
make install    # ~/.venvs/imagegen + imagegen[mlx] on Apple Silicon, imagegen[cuda] on Linux
make test       # unit tests (no weights needed)
make fmt        # format the code in place (ruff format)
make lint       # ruff check + ruff format --check (no changes)
make platform   # print detected platform + default backend
make clean      # remove the venv
```

To use the library/CLI directly (the `imagegen` console script ships with the
package via `[project.scripts]`):

```bash
pip install ".[mlx]"     # Apple Silicon (MLX backend)
pip install ".[cuda]"    # Linux / DGX Spark (PyTorch backend)
imagegen --help
```

## CLI Usage

```bash
# List selectable magic-prompt providers/models
imagegen magic-models

# End-to-end: prompt -> magic-prompt -> image
imagegen run "a ginger cat wizard" \
  --magic-model "codex - gpt-5.5" \
  --width 1024 --height 1024 --preset V4_DEFAULT_20 --seed 42 \
  --model-path /Volumes/PRO-G40/data/models/image-gen/ideogram-4-fp8 \
  --out out.png

# Just expand a prompt to caption JSON (inspect the structured output)
imagegen magic-prompt "a ginger cat wizard" \
  --magic-model "codex - gpt-5.5" \
  --out caption.json

# Generate from an existing caption JSON (int8 via --quantize 8; default keeps fp8)
imagegen generate \
  --caption caption.json \
  --seed 42 \
  --model-path /Volumes/PRO-G40/data/models/image-gen/ideogram-4-fp8 \
  --quantize 8 \
  --out out.png

# Show detected platform + default backend
imagegen platform
```

### Model & precision

There is **one model directory** — the official `ideogram-ai/ideogram-4-fp8` checkpoint
(it serves both the Mac via mflux and the Spark via the PyTorch pipeline). Precision
variants come from quantizing it **on load** with `--quantize`:

- *(default)* — native fp8
- `--quantize 8` — int8 (equivalent to the MLXBits "q8" build)
- `--quantize 4` — 4-bit (smallest/fastest)

> The pre-converted `MLXBits/ideogram-4-mlx*` builds are **not** loadable by released
> mflux (their flat-layout loader lives only in an unmerged PR) — use fp8 + `--quantize`.

### Selecting a magic-prompt provider/model

`imagegen magic-models` lists all available choices.  Pass one to `--magic-model`:

- `codex - gpt-5.5` / `codex - gpt-5.4` / `codex - gpt-5.4-mini` — local Codex CLI
- `pi - <provider> - <model-id>` — pi CLI (reads `~/.pi/agent/models.json`;
  override path with `PI_MODELS_JSON` env var)

Full parameter reference (every CLI flag, presets, and the model components —
Qwen3-VL text encoder, Flux2 VAE, transformers): **[docs/parameters.md](docs/parameters.md)**.

## Status

Phase 1 scaffold: platform detection, config/model spec, the `ImageEngine`
interface, and the factory are in place. Backend `generate()` bodies are stubbed
pending the MLX spike (Mac) and Spark bring-up (CUDA).
