# Model docs

Each model has its own reference under `docs/models/`. These docs cover the
model-specific parts — config keys, presets, caption schema, model components, and
backend-specific notes — while the model-agnostic CLI lives in
[`docs/cli.md`](../cli.md).

## Models

- **[Ideogram 4](ideogram4.md)** — the first supported model. One fp8 checkpoint
  serves both the MLX (Apple Silicon, via mflux) and PyTorch/CUDA backends; precision comes from on-load `quantize`.

## Adding a model doc

When you register a new model (implement the `Model` Protocol in
`models/<name>.py`, call `models.register(...)` at import — see the README's
*Project layout* → "Adding a model"), add a `docs/models/<name>.md` covering its
config keys, presets (if any), caption schema, and model components, then add a line
to the list above. Keep the model-agnostic CLI reference in `docs/cli.md` and link to
it rather than restating shared material.