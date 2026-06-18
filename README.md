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

## Setup

`scripts/setup.sh` detects the platform and installs the matching extra, into a
venv kept outside this (iCloud-synced) repo:

```bash
make setup      # -> [mlx] on Apple Silicon, [cuda] on Linux
make test       # unit tests (no weights needed)
make platform   # print detected platform + default backend
```

## Status

Phase 1 scaffold: platform detection, config/model spec, the `ImageEngine`
interface, and the factory are in place. Backend `generate()` bodies are stubbed
pending the MLX spike (Mac) and Spark bring-up (CUDA).
