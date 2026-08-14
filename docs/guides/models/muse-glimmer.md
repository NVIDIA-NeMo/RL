# Muse Glimmer

This page collects NeMo RL guidance for post-training Muse Glimmer, a ~29.8B model
built on the Onyx architecture. Use it to set up the environment, choose a starting
recipe, and understand the settings that are specific to this model.

> [!IMPORTANT]
> **Early access.** Muse Glimmer runs end-to-end in NeMo RL and short GRPO runs
> have been numerically validated, but long-run convergence has not been established.

## Support Status

Model support is tracked in two stages:

| Stage | Meaning |
| --- | --- |
| **Functionally Ready** | Runnable end-to-end and numerically validated with an initial training run. |
| **Long-Run Convergence Validated** | Trains stably over a full-length run with a healthy, reproducible reward curve. |

Muse Glimmer is **Functionally Ready**.

## What's Supported

| Model | Modality | Training backend | Parallelism | Inference | Precision |
| --- | --- | --- | --- | --- | --- |
| Muse Glimmer 29B | LLM (text-only path) | AutoModel (DTensor) | FSDP2 + CP | vLLM | BF16 |

Notes:

- **Training** runs on the AutoModel (DTensor) backend. The Megatron backend is not
  supported for this architecture.
- **Context Parallel (CP)** is supported and is what makes long sequences possible;
  see [Choose a Recipe](#choose-a-recipe). Tensor Parallel and Pipeline Parallel are
  not used on the training side — absorb memory pressure into CP and FSDP2 instead.

## Build the Environment

Published NeMo RL containers do not include the Muse Glimmer runtime. AutoModel must
come from a local source; vLLM installs directly from a prebuilt nightly wheel, so no
vLLM source checkout is needed.

### 1. Clone AutoModel into `3rdparty/`

- **AutoModel** — Muse Glimmer support is not yet on
  [main](https://github.com/NVIDIA-NeMo/Automodel); it lives on
  [`huiyingl/feat/muse-glimmer-support`](https://github.com/NVIDIA-NeMo/Automodel/tree/huiyingl/feat/muse-glimmer-support).
  The submodule is pinned to that branch until it merges upstream — see the
  `.gitmodules` note for this submodule.
- **vLLM** — `pyproject.toml` pins the `vllm` extra directly to the nightly wheel
  built from vLLM main at
  [6adad0876](https://github.com/vllm-project/vllm/commit/6adad08767583f52eb4d2122111af0bf638ed5e6),
  the commit that merged Muse Glimmer support
  ([nightly build directory layout](https://docs.vllm.ai/en/latest/contributing/ci/nightly_builds/#directory-structure)).
  This installs as a regular (non-editable) package — there is nothing to clone.

```bash
git submodule update --init 3rdparty/Automodel-workspace/Automodel
```

### 2. Force a worker-venv rebuild

Ray worker virtual environments are cached, so they will not pick up the local
AutoModel source or a `pyproject.toml` dependency bump unless you ask for a rebuild:

```bash
export NRL_FORCE_REBUILD_VENVS=true

# Multi-node only. Every node's venv builder contends on one lock in the shared
# uv cache while vLLM's metadata is built; uv's default 300 s lock timeout is
# shorter than that takes, and the waiting nodes die with "Failed to acquire
# lock on the distribution cache".
export UV_LOCK_TIMEOUT=3600
```

> [!NOTE]
> The vLLM wheel is fetched from `https://wheels.vllm.ai` during `uv sync`/`uv lock`.
> If compute nodes have no external network, warm the shared `UV_CACHE_DIR` first by
> running `uv sync` (or `uv lock`) once from a node that does have network access —
> subsequent syncs on compute nodes reuse the cached wheel.

## Get the Weights

The recipes default `policy.model_name` and `policy.tokenizer.name` to the released
HF Hub checkpoint, `meta-models/Muse-Glimmer-30B`. To use a local checkpoint instead,
override both keys:

```bash
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-muse-glimmer-30b-4n8g-fsdp2cp4-automodel-6k.yaml \
  policy.model_name=/your/path/to/muse-glimmer-30b \
  policy.tokenizer.name=/your/path/to/muse-glimmer-30b
```

or edit `policy.model_name` and `policy.tokenizer.name` in the YAML directly.

## Example Recipes

Both are GRPO on DAPO-Math-17K with AIME-2024 validation, AutoModel (DTensor) training
with colocated vLLM generation, on 4 nodes x 8 GPUs. Recipe YAML files under
`examples/configs/recipes/` are the source of truth.

| Seq | CP | Tokens/rank | `max_new_tokens` | Recipe |
|---|---|---|---|---|
| 6144 | 4 | 1536 | 4096 | [`…-30b-4n8g-fsdp2cp4-automodel-6k.yaml`](../../../examples/configs/recipes/llm/grpo-muse-glimmer-30b-4n8g-fsdp2cp4-automodel-6k.yaml) |
| 4096 | 1 | 4096 | 2048 | [`…-30b-4n8g-fsdp2-automodel-4k.yaml`](../../../examples/configs/recipes/llm/grpo-muse-glimmer-30b-4n8g-fsdp2-automodel-4k.yaml) |

These are GRPO, not DAPO: no dynamic sampling, no overlong reward shaping, symmetric
`ratio_clip` at 0.2, and sequence-level loss. The base config `grpo_math_1B.yaml`
already defaults to all of that, so the recipes mostly just avoid overriding it —
`loss_fn.token_level_loss: false` is the one value they state explicitly.

> [!IMPORTANT]
> `max_new_tokens` is `max_total_sequence_length - data.max_input_seq_length` (2048).
> Keep that relationship if you change the sequence length. If `max_new_tokens` is
> larger than the context leaves room for, vLLM silently caps generation per-sample at
> `max_model_len - input_length`, so the effective budget varies with prompt length
> instead of being uniform.

## Choose a Recipe

**Start with the 6k (cp4) recipe.** It is better than the 4k variant on every measured
axis — see [Reference Results](#reference-results): roughly 0.69 peak validation
accuracy against 0.505, a `gen_kl_error` that stays flat instead of drifting upward,
and entropy that recovers rather than collapsing.

Use the 4k recipe when you want the shorter context or fewer moving parts. It runs
without context parallelism, so there is no ring-attention communication and no
per-CP-step `dq`/`dk`/`dv` copies — but it is the *tighter* of the two on memory.

### Why context length matters here

Muse Glimmer produces long reasoning traces. When the sequence budget binds, the policy
learns to stop early rather than to reason better, and reward decouples from
correctness. `train/truncation_rate` early in training runs around **0.6 at 4k** versus
**0.34 at 6k**, and the accuracy gap follows.

> [!NOTE]
> Diagnose truncation pressure with `train/truncation_rate` and
> `train/mean_gen_tokens_per_sample`, **not** mean generation length alone. When the cap
> binds, mean generation length *falls*, which reads like "the model does not need the
> tokens" and means the opposite.

## Launch

```bash
export NRL_FORCE_REBUILD_VENVS=true

# 6k, CP4 -- recommended default (pulls meta-models/Muse-Glimmer-30B from HF Hub)
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-muse-glimmer-30b-4n8g-fsdp2cp4-automodel-6k.yaml

# 4k, no CP -- shorter context, tighter on memory
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-muse-glimmer-30b-4n8g-fsdp2-automodel-4k.yaml
```

On Slurm, keep `cluster.num_nodes` in step with what you request from the scheduler.
A mismatch wedges Ray placement rather than failing cleanly:

```bash
uv run examples/run_grpo.py --config <recipe> cluster.num_nodes=4
```

## Reference Results

### Training curves

Both runs use the released checkpoint on 4 nodes x 8 GPUs. Each curve covers roughly
the first 70 steps of a 300-step target, so these are **partial runs**, not completed
ones.

**6k, CP4** — the recommended recipe.

![Muse Glimmer 6k GRPO training curves](../../assets/muse-glimmer/muse_glimmer_6k_grpo_curve.png)

Validation accuracy rises from 0.42 to a peak near **0.69** by step 15 and holds
0.63-0.67. `truncation_rate` falls from 0.34 to roughly 0.05-0.15. `gen_kl_error` is
**flat** at ~5.5e-04 throughout. Entropy dips, then recovers to 0.68 around step 43
before settling near 0.47 — the policy keeps exploring rather than collapsing.

**4k, no CP** — shorter context.

![Muse Glimmer 4k GRPO training curves](../../assets/muse-glimmer/muse_glimmer_4k_grpo_curve.png)

Validation accuracy rises from 0.22 to about **0.505** by step 55-60 and then flattens.
`truncation_rate` starts near 0.6 — well above the 6k run — and settles around 0.15.
`gen_kl_error` **climbs** from 5.5e-04 to 1.4e-03, and entropy falls monotonically from
0.38 to 0.15.

## Known Issues

- **Long-run convergence is not validated.** Current evidence covers short runs only.
