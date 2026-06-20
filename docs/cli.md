# `ig` CLI reference

`ig` is the shipped command-line interface for imgen (entry point
`imgen.cli:main`, installed via `[project.scripts]`). It uses a **model-first
grammar**: `ig <model> <action> [options]` — each model is a Click group, and the
group is built automatically from the model registry, so a newly-registered model
gets an `ig <model>` group for free.

This document covers the **model-agnostic** CLI: command structure, the warm-daemon
model, override precedence, and output. Model-specific detail — config keys, gen
options, presets, caption schema, magic-prompt — lives in each model's own doc under
[`docs/models/`](models/).

> **Format conventions** — options are listed as bullets: the flag is bold, then a
> parenthetical `*(type, default X)*`, then the description. Defaults appear next to
> the flag they belong to. Tables are used only where a tight grid genuinely helps.

## Command overview

```
ig
├── <model>                              # one group per registered model (e.g. ideogram4)
│   ├── gen                              # generate an image (warm daemon, auto-start)
│   ├── config
│   │   ├── set <key> <value>            # persist build config to config.toml
│   │   ├── set-key <provider> <key>     # store an API key in secrets.toml
│   │   └── show                         # print this model's current config
│   ├── serve [--detach|-d]              # start the warm daemon (foreground or background)
│   └── stop                             # stop this model's daemon
├── model                                # inspect models & manage daemons
│   ├── list                             # models + daemon status (idle / busy / crashed)
│   ├── show <model>                     # options, defaults, config keys
│   ├── jobs [<id>]                      # background (--queue) job records
│   ├── clean [--all] [--older-than N]   # prune finished jobs / dead-daemon logs
│   └── stop-all                         # stop every running daemon
└── platform                             # detected platform + default backend
```

Run `ig model show <model>` to print all config keys and gen options a model
exposes.

## Quickstart

The example below uses **ideogram4** as the concrete model — substitute your
model's group name. See [`docs/models/ideogram4.md`](models/ideogram4.md) for
ideogram4-specific config keys, presets, and magic-prompt setup.

```bash
# one-time: tell ig where the weights are (key name is model-specific)
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

## Build config and override precedence

Build config (weights path, backend, and any model-specific knobs) is set **once**
with `ig <model> config set` and persisted to `~/.config/ig/config.toml`. The load
config is overridable per run by a **model-group option** or an **`IG_*` env var**,
with precedence:

**option > env var > `config set` > built-in default** (`api_key` excepted — keys
are stored with `config set-key` or `*_API_KEY` env vars, never overridden by a
flag).

These build the model, so they take effect when the daemon is (auto-)started; if a
daemon is already warm the override is ignored with a warning (`ig <model> stop` to
rebuild).

> **The available build keys, their allowed values, and their `IG_*` env-var names
> are model-specific** — see the model's own doc
> (e.g. [`docs/models/ideogram4.md#config-keys`](models/ideogram4.md#config-keys)).
> Models may expose model-specific overrides on top of the common ones (load config
> like weights/backend, plus model-specific knobs such as ideogram4's quantize and
> magic-prompt provider/model).

## Commands

### `ig <model> gen` — generate an image

Routes the request through the **warm daemon** (auto-started if it isn't running).
Progress streams to **stderr**; on completion the metadata JSON is printed to
**stdout** and written as a sidecar at `<out>.json`.

**Shared options** (every model):

- **`-o`**, **`--out`** *(path, required)* — output image file.
- **`-q`**, **`--queue`** *(flag)* — run in the background; prints a job id to poll
  with `ig model jobs`.

**Model-specific options** — each model supplies its own gen options (typically
prompt, dimensions, seed, preset, caption, and any model-specific knobs). See the
model's own doc for the full list and allowed values
(e.g. [`docs/models/ideogram4.md#gen-options`](models/ideogram4.md#gen-options)).

**Build overrides (before the action):** `ig <model> --<build-key> … gen` (or the
matching `IG_*` env var) override the daemon's load config for this start. See
*Build config and override precedence* above and the model's doc for its keys.

**Example** (ideogram4):

```bash
ig ideogram4 gen -p "a ginger cat wizard" -w 768 -h 768 --seed 42 \
   -o out.png --preset V4_DEFAULT_20
# from a prebuilt caption (skips magic-prompt)
ig ideogram4 gen -o out.png --caption caption.json
# fire-and-forget
ig ideogram4 gen -p "a cat" -w 768 -h 768 -o out.png --queue
```

**Output — `<out>.json` sidecar** (the same object is printed to stdout as JSON;
models may add model-specific fields):

- **`caption`** — the structured JSON caption sent to the model (model-specific
  schema — see the model's doc).
- **`prompt`** — the original plain-text prompt (`null` when `--caption` was used).
- **`model`** — model name (e.g. `ideogram4`).
- **`seed`** — actual seed used (useful when the seed option was omitted).
- **`width`**, **`height`** — actual image dimensions in px.
- **`preset`** — sampler preset used (model-specific; absent if the model has none).
- **`backend`** — inference backend (`mlx` or `torch`).
- **`duration_s`** — total generation time in seconds.
- **`out`** — path to the output image file.

### `ig <model> config` — get/set persisted build config

Persists build config to `~/.config/ig/config.toml`. The set of keys and their
allowed values are **model-specific** — see the model's own doc
(e.g. [`docs/models/ideogram4.md#config-keys`](models/ideogram4.md#config-keys)).
Run `ig model show <model>` to print all available config keys and gen options.

**Subcommands:**

- **`ig <model> config set`** *(args: `<key> <value>`)* — set a build config value.
- **`ig <model> config set-key`** *(args: `<provider> <api-key>`)* — store an API
  key in `~/.config/ig/secrets.toml` (mode 600, never committed). Used by models
  that call out to an HTTP API (e.g. ideogram4's magic-prompt providers).
- **`ig <model> config show`** — print the current config for this model.

**Example:**

```bash
ig ideogram4 config set weights-path /Volumes/PRO-G40/data/models/image-gen/ideogram-4-fp8
ig ideogram4 config show
```

### `ig <model> serve` — start the warm daemon

Loads the model once and stays resident, accepting one job at a time via the
daemon socket. Build config is read from `ig <model> config` — no per-serve flags
needed. The socket, registry, and log are placed under `IG_RUNTIME_DIR` (default
`~/.cache/ig`) and managed automatically. If a daemon for that model is already
running, `serve` exits with an error and shows the PID — run `ig <model> stop`
first.

**Options:**

- **`-d`**, **`--detach`** *(flag, default off)* — run the daemon in the background
  (returns immediately).

**Example:**

```bash
ig ideogram4 serve           # foreground (Ctrl-C to stop)
ig ideogram4 serve --detach   # background (alias: -d)
```

### `ig <model> stop` — stop the daemon

No arguments. Sends SIGTERM to the daemon and cleans up the registry. Prints a
message whether or not a daemon was running.

### `ig model` — inspect models and manage daemons

**Subcommands:**

- **`ig model list`** — list all available models with their daemon status (PID +
  `idle`/`busy`, or `crashed` with a log path if a daemon died mid-job).
- **`ig model show`** *(arg: `<model>`)* — show options, defaults, and config keys
  for a specific model.
- **`ig model jobs`** *(arg: optional `<id>`)* — list background (`--queue`) jobs,
  or show one job by id (with its log path).
- **`ig model clean`** *(flags: `--all`, `--older-than DAYS`)* — remove finished-job
  records/logs and dead-daemon logs. `--all` also truncates running daemons' logs;
  `--older-than` only prunes jobs finished more than N days ago.
- **`ig model stop-all`** — stop every running daemon.

### `ig platform` — detect platform and default backend

No arguments. Prints the detected platform (Apple Silicon / Linux) and default
backend (MLX / PyTorch).

## Warm-daemon model

`ig <model> gen` always routes through a **warm daemon** — a background process that
loads the model once and stays resident, accepting one job at a time. Concurrent
`gen` calls queue automatically (one GPU → one job; the socket backlog is the
queue).

- **Auto-start** — if no daemon is running, `gen` spawns one and waits up to 15
  minutes for the model to load before sending the job.
- **`ig <model> serve`** starts the daemon in the foreground; `--detach` / `-d`
  backgrounds it. Starting a second `serve` while one is running prints an error and
  tells you to run `ig <model> stop` first.
- **`ig <model> stop`** sends SIGTERM and cleans up the registry entry.
- **`ig model stop-all`** stops every daemon. **`ig model list`** shows running
  status (`idle` / `busy`, or `crashed` with a log path if a daemon died mid-job).
- **Sidecar** — after each generation `gen` writes `<out>.json` next to the image
  (see the `gen` output fields above).

### Runtime directory

`gen`, `serve`, and `stop` all use a shared **runtime directory** (default
`~/.cache/ig`, override with `IG_RUNTIME_DIR`):

- `$IG_RUNTIME_DIR/daemons/<model>.sock` — Unix domain socket the daemon listens on.
- `$IG_RUNTIME_DIR/daemons/<model>.json` — live registry record (PID, socket path,
  backend, state).
- `$IG_RUNTIME_DIR/logs/<model>.log` — stdout/stderr of the detached daemon process.
- `$IG_RUNTIME_DIR/jobs/<id>.json` + `<id>.log` — per-job record + log for `--queue`
  background jobs.

There is **one daemon per model**. If the socket path would exceed the platform
limit (104 bytes on macOS, 108 on Linux) `ig` exits with an error — set
`IG_RUNTIME_DIR` to a shorter path in that case.

### Background (async) generation

Pass `--queue` to fire-and-forget without blocking. The detached process streams
progress into `$IG_RUNTIME_DIR/jobs/<id>.log`, writes the same `<out>.json` sidecar
when done, and updates `$IG_RUNTIME_DIR/jobs/<id>.json` (`queued` → `running` →
`done`/`failed`).

```bash
ig ideogram4 gen -p "a cat" -w 768 -h 768 -o out.png --queue
# prints: job <id> → out.png   (poll: ig model jobs <id>)

ig model jobs            # list all jobs (queued / running / done / failed)
ig model jobs <id>       # show one job's record + its log path
ig model clean           # remove finished-job records/logs + dead-daemon logs
ig model clean --all     # also truncate logs of running daemons
ig model clean --older-than 7   # only prune jobs finished > 7 days ago
```