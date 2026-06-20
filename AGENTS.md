# Working in this repo (agents)

How we develop here. Read the **README first** — especially *Architecture*,
*Project layout*, and *Parameter & model reference*. This file is *how we work*;
the README is *what it is*. Keep this file short; link to README sections rather
than restating them.

## Always
- **TDD + green gate.** Write the test, then the code. Run `make style && make test`
  while iterating; the merge gate is `make lint` (ruff + `ruff format --check` +
  mypy strict) **and** `make test` (see README → *Install* for targets). Never
  commit on a red gate.
- **Pythonic & succinct.** Idiomatic, fully type-annotated (mypy strict), small
  focused modules. Match surrounding style and naming. Prefer clear over clever.
- **Docs current.** In the *same* change that alters behavior, flags, or structure,
  update the README (and its *Project layout*). Docs must reflect the repo's actual
  state — no stale or aspirational claims.

## Patterns (don't reinvent)
- Follow existing patterns; do **not** copy-paste, fork, or invent a parallel way to
  do something that already exists. New models plug into the `Model` registry
  (README → *Project layout* → "Adding a model"); new caption providers / inference
  backends extend `magic_prompt/` and `engine/`. Reuse `pipeline.py`, `worker.py`,
  `daemon.py`, `metadata.py` instead of duplicating them.
- **Design for multiple pipelines.** Keep `engine/`, `worker`, `daemon`, `jobs`, and
  `cli/` model-agnostic; model-specific logic belongs in `models/` (and where
  applicable `magic_prompt/`). If a change hard-codes one model's assumptions into a
  shared module, push it back into the model plugin.

## After big features — review & refactor
- Re-read the whole diff with fresh eyes. **Merge code with similar shape/structure**,
  extract shared helpers, and **delete dead code aggressively**.
- Keep structure **minimal**: implement only what was requested, remove anything
  unused, and avoid speculative abstraction.

## Dependencies
- Periodically check `pyproject.toml` deps are up-to-date; bump and re-run the gate.
- **Remove unused dependencies immediately** (runtime and dev alike).
