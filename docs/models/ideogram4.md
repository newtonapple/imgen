# Ideogram 4

**Ideogram 4** is the first supported imgen model. One model directory — the
official `ideogram-ai/ideogram-4-fp8` checkpoint — serves both backends:

| Platform | Backend | How |
| --- | --- | --- |
| Apple Silicon | **MLX** | [mflux](https://github.com/filipstrand/mflux) |
| Linux / CUDA | **PyTorch** | official [`ideogram4`](https://github.com/ideogram-oss/ideogram4) pipeline |

Precision variants come from quantizing the fp8 checkpoint **on load** (see
*Config keys* → `quantize`).

> **Status:** the MLX (Apple Silicon) path is complete and validated end-to-end.
> The PyTorch/CUDA backend is pending bring-up (`engine/torch_engine.py` is a
> stub).

## Config keys

Set with `ig ideogram4 config set <key> <value>` (persists to
`~/.config/ig/config.toml`). See [`docs/cli.md`](../cli.md) for the generic config
mechanism and override precedence.

- **`weights-path`** *(path)* — path to the `ideogram-4-fp8` checkpoint directory.
- **`backend`** *(`mlx` | `torch`, auto-detected from platform if unset)* — inference
  backend.
- **`quantize`** *(`4` | `8` | empty to clear)* — quantize the fp8 weights to N bits
  on load. `8` = int8 (≈ the `MLXBits` "q8" build); `4` = 4-bit (smallest/fastest);
  empty (default) keeps native fp8.
- **`magic-provider`** *(provider name)* — which provider turns text → JSON caption.
  See [Magic-prompt](#magic-prompt).
- **`magic-model`** *(model id)* — model id for the chosen provider.

```bash
ig ideogram4 config set weights-path /Volumes/PRO-G40/data/models/image-gen/ideogram-4-fp8
ig ideogram4 config set quantize 8
ig ideogram4 config set backend mlx
```

> **The `MLXBits/ideogram-4-mlx*` builds are not loadable** by released mflux — their
> flat-layout loader lives only in an unmerged PR. Use the fp8 checkpoint + `quantize`.

### Build overrides & env vars

The load config keys above are overridable per run by a model-group option or an
`IG_*` env var, with precedence **option > env var > `config set` > built-in**
(see [`docs/cli.md#build-config-and-override-precedence`](../cli.md#build-config-and-override-precedence)).
They build the model, so they take effect when the daemon is (auto-)started; if a
daemon is already warm the override is ignored with a warning (`ig ideogram4 stop`
to rebuild).

- **`--weights-path`** / `IG_WEIGHTS_PATH`
- **`--backend`** / `IG_BACKEND`
- **`--quantize`** / `IG_QUANTIZE`
- **`--mp` / `--magic-provider`** / `IG_MAGIC_PROVIDER` — daemon-default magic-prompt
  provider (serve-compatible).
- **`--mm` / `--magic-model`** / `IG_MAGIC_MODEL` — daemon-default magic-prompt model.

```bash
ig ideogram4 --quantize 8 gen -p "a cat" -o out.png   # option (this start)
IG_QUANTIZE=8 ig ideogram4 gen -p "a cat" -o out.png  # env var
```

## Gen options

`ig ideogram4 gen` accepts the shared options (`-o/--out`, `-q/--queue`; see
[`docs/cli.md`](../cli.md#ig-model-gen)) plus these ideogram4-specific options:

- **`-p`**, **`--prompt`** *(str)* — plain-text prompt to expand into a structured
  JSON caption. Optional if `--caption` is given.
- **`-w`**, **`--width`** *(int, default 1024)* — output width in px. Rounded to the
  nearest multiple of 16 (minimum 256), with a stderr warning if that changes the
  value (the model requires multiples of 16). The model supports 256–2048 with
  aspect ≤ 6:1; values beyond that are not clamped and may be rejected by the
  backend.
- **`-h`**, **`--height`** *(int, default 1024)* — output height in px. Same as
  width.
- **`--seed`** *(int, default random)* — RNG seed. Same seed + caption + params ⇒
  reproducible; omit for random.
- **`--preset`** *(name, default `V4_DEFAULT_20`)* — sampler bundle. See [Presets](#presets).
- **`--target-elements`** *(int, default 0 = auto)* — force ~N entries in the
  caption's `compositional_deconstruction.elements` (see [Caption schema](#caption-schema)).
- **`--caption`** *(path)* — prebuilt caption JSON file. If set, `--prompt` is
  ignored and magic-prompt is skipped.
- **`--mp`**, **`--magic-provider`** *(provider, default from config)* — per-request
  magic-prompt provider for this generation only. See [Magic-prompt](#magic-prompt).
- **`--mm`**, **`--magic-model`** *(str, default from config)* — per-request
  magic-prompt model for this generation only.

```bash
ig ideogram4 gen -p "a ginger cat wizard" -w 768 -h 768 --seed 42 \
   -o out.png --preset V4_DEFAULT_20
```

## Presets

A preset bundles the sampler settings (steps + guidance schedule + `mu`/`std`) so
you don't tune them individually. Set with `ig ideogram4 gen --preset <name>`.

- **`V4_TURBO_12`** — fast iteration / previews. 12 steps; guidance 3.0 for the
  first step then 7.0; `mu` 0.5, `std` 1.75.
- **`V4_DEFAULT_20`** *(default)* — balanced. 20 steps; guidance 3.0 for the first 2
  steps then 7.0; `mu` 0.0, `std` 1.75.
- **`V4_QUALITY_48`** — final / high-fidelity. 48 steps; guidance 3.0 for the first 3
  steps then 7.0; `mu` 0.0, `std` 1.5.

**What each knob means:**

- **steps** — denoising iterations; more = higher quality, slower (turbo ≈ ¼ the
  time of quality).
- **guidance (CFG)** — per-step prompt-adherence strength. Each preset starts with a
  few **low-guidance (3.0)** steps so the global composition settles naturally, then
  clamps to **strong guidance (7.0)** to lock in fidelity and detail. Higher is not
  "better": past the sweet spot guidance over-saturates and distorts, which is why
  the steady-state sits at 7.0.
- **`mu` / `std`** — shape of the flow-matching noise schedule (where denoising effort
  is concentrated).

The image **dimensions** (`-w`/`-h`) are passed to the model directly — they are
**not** part of the caption (the caption's `aspect_ratio` field, if present, is
dropped before generation).

## Caption schema

The **Ideogram-4 structured caption** — the only schema the model accepts — has
these fields:

- **`high_level_description`** — one-sentence scene summary.
- **`style_description`** — aesthetics, lighting, medium, `photo`/`art_style`, color
  palette.
- **`compositional_deconstruction`** — `background` + `elements[]`:
  - each element has `type`, `bbox`, `desc`, and an optional palette.
  - **`--target-elements`** *(int, default 0 = auto)* — force ~N entries in
    `elements[]`. `0` lets the LLM choose.

**Reference:** the authoritative schema, field order, `bbox` coordinate system
(normalised 0–1000, `[y1, x1, y2, x2]`), and `color_palette` rules are in
Ideogram's official [Prompting Guide](https://github.com/ideogram-oss/ideogram4/blob/main/docs/prompting.md).
Snake_case only; key order is part of the contract.

> The `elements[].bbox` field is captured in the sidecar specifically to later draw
> the layout on the image (caption-overlay feature, not yet implemented).

## Magic-prompt

Ideogram 4 is trained **exclusively on structured JSON captions** (see [Caption
schema](#caption-schema)) — not free-form text. A plain sentence works, but
underperforms, because the model expects every object, style, and layout decision
spelled out as typed JSON fields. Hand-writing that JSON for every request is
tedious, so `gen` runs a **magic-prompt** step first: an LLM expands your
plain-text prompt into the full structured caption (the three top-level keys,
`bbox` positions, `color_palette`, per-element `text`/`desc`, etc.), and *that*
JSON is what the model actually consumes.

That expansion is exactly the job an LLM is good at — decomposing a scene into the
schema's slots, committing to concrete values (one medium, one typeface, fixed
colours), and preserving quoted text verbatim — which is **why an LLM provider is
required** here. Ideogram's own reference is a hosted magic-prompt API; rather than
hard-depend on it, imgen makes the provider **pluggable**: pick any of the CLI or
HTTP providers below and it produces the same caption contract. Skip the expansion
entirely by passing a prebuilt caption with `--caption`.

Two config keys select the provider and model: `magic-provider` and `magic-model`
(set with `ig ideogram4 config set`).

The provider/model can be overridden at **two** levels (`--mp` = `--magic-provider`,
`--mm` = `--magic-model`), with precedence:
**per-request `gen --mp/--mm` > group `--mp/--mm` / `IG_MAGIC_*` > `config set` >
`codex`**. The per-request override applies to just that generation with no daemon
restart; the group-level override applies at daemon build time (serve-compatible).

| Family | Providers | Key required? | Key source |
| --- | --- | --- | --- |
| CLI (shell out) | `codex`, `claude`, `pi` | No | — |
| HTTP chat | `openai`, `anthropic`, `openrouter` | Yes | `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY` env var, or `ig ideogram4 config set-key <provider> <key>` |
| Ideogram hosted | `ideogram` | Yes | `IDEOGRAM_API_KEY` env var, or `ig ideogram4 config set-key ideogram <key>` |

- The default provider/model is `codex` + `gpt-5.5` when nothing is configured.
- `ig ideogram4 config set-key <provider> <key>` stores an HTTP API key in
  `~/.config/ig/secrets.toml` (mode 600, never committed). API keys are the one
  setting **not** overridable by a plain option.
- **`openrouter`** accepts a comma-separated model list for `magic-model`
  (e.g. `openrouter/free,openai/gpt-4o`) → an OpenRouter `models[]` fallback chain.
- **`pi`** takes `magic-model "<pi-provider>/<model>"` (reads
  `~/.pi/agent/models.json`; override with `PI_MODELS_JSON`).
- **`ideogram`** calls Ideogram's own hosted magic-prompt API — the reference
  implementation. **Get an API key at
  [developer.ideogram.ai](https://developer.ideogram.ai)** (log in → API Dashboard
  / [ideogram.ai/manage-api](https://ideogram.ai/manage-api) → accept the Developer
  API Agreement → add payment info → create a key), then store it with
  `ig ideogram4 config set-key ideogram <key>` (or the `IDEOGRAM_API_KEY` env var).
  The prompt **expansion itself is free** — imgen only calls the magic-prompt
  endpoint, not the paid per-image generate endpoint. The endpoint returns the
  structured caption directly, so `--mm`/`magic-model` (always `v4`) and
  `--target-elements` are ignored. **Rate limit:** the default is
  [10 in-flight requests](https://ideogram.ai/features/api-pricing) per API account;
  429s surface as a `rate-limited by ideogram; retry shortly` error.

```bash
# free OpenRouter pool (rotating) — store key + persist as default
ig ideogram4 config set-key openrouter sk-or-...
ig ideogram4 config set magic-provider openrouter
ig ideogram4 config set magic-model openrouter/free

# per-request override — no daemon restart
ig ideogram4 gen -p "a cat" -o out.png --mm anthropic/claude-haiku-4-5
```

**Reference:** Ideogram's open-source [magic-prompt system prompt](https://github.com/ideogram-oss/ideogram4/blob/main/src/ideogram4/magic_prompt_system_prompts/v1.txt)
defines the exact caption contract our providers produce (single-line minified
JSON, the three top-level keys, `bbox` strategy, text-handling rules). The `ideogram`
provider *is* that hosted magic-prompt API; the CLI and HTTP-chat providers are
drop-in alternatives that produce the same contract.

## Model components

Ideogram-4 is not one file — it's several components. In ComfyUI you load them as
separate nodes; in this pipeline they all live inside the **one** `ideogram-4-fp8/`
directory and load together. The two pieces you may have seen named separately:
**Qwen3-VL = the text/prompt encoder**, **Flux2 VAE = the image decoder** — different
jobs, both required.

| Component | ComfyUI file / our subdir | What it does |
| --- | --- | --- |
| **Text encoder** | `qwen3vl_8b_fp8_scaled` / `text_encoder/` + `tokenizer/` | **Qwen3-VL-8B** (vision tower stripped) — encodes the JSON caption into the conditioning the diffusion model follows. This is the "Qwen" piece; it is **not** a VAE. |
| **Diffusion transformer (conditional)** | `ideogram4_fp8_scaled` / `transformer/` | The DiT that denoises the latent, guided by the text conditioning. |
| **Diffusion transformer (unconditional)** | `ideogram4_unconditional_fp8_scaled` / `unconditional_transformer/` | Second DiT for **asymmetric classifier-free guidance** (the unconditional pass drops text tokens — that's the "guidance" knob, not a negative prompt). |
| **VAE** | `flux2-vae` / `vae/` | **Flux.2-architecture VAE** — decodes the denoised latent into the final pixel image (and defines the latent space the diffusion runs in). |
| **Scheduler** | `Ideogram4Scheduler` / `scheduler/` | Flow-matching noise schedule (the `mu`/`std` from the preset). |

## mflux knobs we don't expose (yet)

The MLX backend (mflux) also supports `--num-inference-steps` / `--guidance`
(override the preset — note that passing steps flattens the preset's guidance
schedule), `--lora-paths` / `--lora-scales`, and `--low-ram` /
`--mlx-cache-limit-gb`. We can surface any of these if needed; the presets are the
intended interface.