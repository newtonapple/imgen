# Working in this repo (agents)

How we develop here. Read the **README first** — especially *Architecture*, *Project layout*, and
*Docs* — then the model-agnostic CLI reference in `docs/cli.md` and per-model
references under `docs/models/`. This file is *how we work*; the README and
`docs/` are *what it is*. Keep this file short; link to those rather than
restating them.

## Branches & commits
- **Never auto-commit on `main`.** Only commit when you are on a branch off `main`;
  if you find yourself on `main`, stop and create a branch (or worktree) first.
- **Before starting a feature, ASK where to develop** — a feature branch, a
  worktree created with worktrunk (`wt`), or in the current branch without
  committing (leave the changes for the user to review). Don't assume; wait for
  the answer.
- Prefer a feature branch or a `wt` worktree for any non-trivial change; use a
  worktree (`wt`) when explicitly instructed.
- **Use the `worktrunk` skill for all worktrunk / `wt` operations** (creating,
  listing, switching, and cleaning up worktrees).

## Always
- **TDD + green gate.** Write the test, then the code. Run `make style && make test`
  while iterating; the merge gate is `make lint` (ruff + `ruff format --check` +
  mypy strict) **and** `make test` (see README → *Install* for targets). Never
  commit on a red gate.
- **Pythonic & succinct.** Idiomatic, fully type-annotated (mypy strict), small
  focused modules. Match surrounding style and naming. Prefer clear over clever.
- **Docs current.** In the *same* change that alters behavior, flags, or structure,
  update the README (and its *Project layout*) and the relevant `docs/` page
  (`docs/cli.md` for shared CLI, `docs/models/<model>.md` for model-specific
  detail). Docs must reflect the repo's actual state — no stale or aspirational
  claims.

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
