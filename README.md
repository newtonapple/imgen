# imgen

A pluggable, pure-Python **image-generation** pipeline: plain prompt → magic-prompt
→ structured caption → image, with edit/regenerate. Models plug into a registry
behind a common interface and run on a per-platform inference backend — **Ideogram 4**
is the first supported model.

Built to replace a ComfyUI workflow with our own code. Pluggable inference
backends, selected per platform:

| Platform | Backend | How |
| --- | --- | --- |
| Apple Silicon | **MLX** | [mflux](https://github.com/filipstrand/mflux) |
| Linux / CUDA | **PyTorch** | official [`ideogram4`](https://github.com/ideogram-oss/ideogram4) package |

## Architecture

A factory takes a **model** (weights on disk) + an **inference backend** and
builds a warm, load-once **inference pipeline** (`ImageEngine`):

```python
from imgen import ModelSpec, create_pipeline

model = ModelSpec.from_path("/Volumes/PRO-G40/data/models/image-gen/ideogram-4-fp8")
engine = create_pipeline(model)          # backend auto-detected (MLX here)
result = engine.generate(caption, width=1024, height=1024, preset="V4_DEFAULT_20")
result.image.save("out.png")
```

The same engine instance is what a worker holds warm and calls per job; the CLI
is just one caller.

## Project layout

```
.
├── AGENTS.md            # agent development guide (CLAUDE.md is a symlink to this)
├── README.md
├── Makefile             # install / test / lint / style targets
├── pyproject.toml       # package metadata, deps, tool config
├── scripts/setup.sh     # `make install`: build the venv + install imgen[extra]
├── src/imgen/
│   ├── cli/             # `ig` command-line interface (Click groups + action bodies)
│   ├── models/          # model registry + plugins — the extension point (`Model` Protocol)
│   ├── engine/          # inference backends (MLX / PyTorch-CUDA) + create_pipeline factory
│   ├── magic_prompt/    # text -> structured JSON caption providers (Ideogram-4-specific)
│   ├── pipeline.py      # Pipeline = engine + magic_prompt — the warm, load-once unit
│   ├── worker.py        # warm worker: one job at a time, NDJSON over a Unix socket
│   ├── daemon.py        # one daemon per model: registry, liveness, auto-start, stop
│   ├── jobs.py          # `--queue` background jobs: records, detached runner, clean
│   └── config.py        # Config/Secrets (TOML) + runtime dirs / socket-path helpers
└── tests/               # pytest, one test_*.py per module (offline; real GPU behind @integration)
```

(`docs/` holds tracked model & CLI reference docs; local planning notes under
`docs/design/`, `docs/plans/`, `docs/superpowers/`, etc. are gitignored — see
`.gitignore`.)

**Adding a model:** implement the `Model` Protocol in `models/<name>.py`, call
`models.register(...)` at import, and import the module so it registers. The CLI
builds the `ig <name>` group automatically. Reuse the `engine/` backends and the
`pipeline.py` / `worker.py` / `daemon.py` machinery — only add a new engine or
magic-prompt mechanism if the model genuinely needs one. Keep `engine/`, `worker`,
`daemon`, `jobs`, and `cli/` model-agnostic; model-specific logic lives in `models/`
and (where applicable) `magic_prompt/`.

## Weights

Model weights are **never committed** — they sit on an external volume and are
referenced by path (`ModelSpec.path`, or the `IMGEN_WEIGHTS_ROOT` env var).

## Install

`make install` creates the virtualenv at `~/.venvs/imgen` (kept **outside** the
repo, so uv clones packages from its global cache with no byte duplication;
override with `IMGEN_VENV`) and installs the package (editable) plus the
platform backend extra — which also installs the `ig` CLI:

```bash
make install    # ~/.venvs/imgen + imgen[mlx] on Apple Silicon, imgen[cuda] on Linux
make test       # unit tests (no weights needed)
make test-integration  # integration tests (real API/GPU; self-skip when no key/weights)
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
pip install ".[cuda]"    # Linux / CUDA (PyTorch backend)
ig --help
```

## Quickstart

```bash
# one-time: tell ig where the weights are
ig ideogram4 config set weights-path /Volumes/PRO-G40/data/models/image-gen/ideogram-4-fp8

# generate — the warm daemon is auto-started if it isn't running;
# progress streams to stderr, metadata JSON to stdout + <out>.json sidecar
ig ideogram4 gen -p "a ginger cat wizard" -w 768 -h 768 --seed 42 -o out.png \
   --preset V4_DEFAULT_20

ig ideogram4 serve --detach   # start the warm daemon in the background
ig ideogram4 stop             # stop this model's daemon
ig model list                 # daemon status (idle / busy / crashed)
ig platform                   # detected platform + default backend
```

Run `ig model show ideogram4` to see all config keys and gen options for a model.

## Docs

- **[`docs/cli.md`](docs/cli.md)** — full CLI reference: `gen` / `config` / `serve` /
  `stop` / `model` / `platform`, the warm-daemon model, override precedence, and
  magic-prompt provider plumbing. Model-agnostic.
- **[`docs/models/`](docs/models/)** — per-model references (config keys, presets,
  caption schema, model components, backend notes):
  - **[Ideogram 4](docs/models/ideogram4.md)** — `ideogram4` package/nf4 on CUDA (Linux)
    and MLX (Apple Silicon, via mflux) backends; on-load `quantize` (MLX only).

Local planning notes (design specs, plans, handoffs) live under `docs/design/`,
`docs/plans/`, `docs/superpowers/`, etc. and are **not tracked** — see `.gitignore`.

## Status

Both backends are implemented: MLX (Apple Silicon, via mflux) and PyTorch/CUDA
(Linux, via the official `ideogram4` package). The CUDA backend loads a local
Ideogram-4 snapshot directly; the nf4 build is validated end-to-end on the DGX
Spark (fp8 should work via the same package and is pending confirmation).
