# Qwen3.8 FlashNext

This page collects NeMo RL guidance for post-training Qwen3.8 FlashNext 180B,
also known as BrightDelta. Use it to set up the environment, choose a starting
recipe, and understand the settings that are specific to this model.

> [!IMPORTANT]
> **Early access.** Qwen3.8 FlashNext 180B runs end-to-end in NeMo RL and short
> GRPO runs have been numerically validated, but long-run convergence has not
> been established.

## Support Status

Model support is tracked in two stages:

| Stage | Meaning |
| --- | --- |
| **Functionally Ready** | Runnable end-to-end and numerically validated with an initial training run. |
| **Long-Run Convergence Validated** | Trains stably over a full-length run with a healthy, reproducible reward curve. |

Qwen3.8 FlashNext 180B is **Functionally Ready**.

## What's Supported

| Model | Modality | Training backend | Parallelism | Inference | Precision |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8 FlashNext 180B | LLM (text-only path) | AutoModel (DTensor) | FSDP2 + EP | vLLM | BF16 |

Notes:

- **Training** runs on the AutoModel (DTensor) backend with TP1 and EP64.
- **Inference** uses eight colocated vLLM TP8/EP8 replicas on the 8-node,
  64-GPU allocation.

## Build the Environment

Published NeMo RL containers do not yet include the complete Qwen3.8 FlashNext
runtime. Use the pinned AutoModel submodule together with a vLLM build that
contains Qwen3.8 FlashNext support.

### 1. Clone the sources into `3rdparty/`

Sources:

- **AutoModel** — the revision pinned by this repository's AutoModel submodule.
- **vLLM** — `<partner vLLM repository>`, branch `qwen38next`, or a container
  with the same runtime baked in.

```bash
git submodule update --init 3rdparty/Automodel-workspace/Automodel

mkdir -p 3rdparty/vLLM-workspace
git clone <partner-vllm-repository-url> 3rdparty/vLLM-workspace/vllm
git -C 3rdparty/vLLM-workspace/vllm checkout qwen38next
```

### 2. Force a worker-venv rebuild

Ray worker virtual environments are cached, so they will not pick up the local
AutoModel and vLLM sources unless you ask for a rebuild:

```bash
export NRL_FORCE_REBUILD_VENVS=true

# Skip vLLM's CUDA build. Without this the editable install runs cmake, whose
# FetchContent step clones dependencies from GitHub -- which fails on compute
# nodes with no external network. Stage the wheel on a shared filesystem first;
# a URL only works if the compute nodes can reach it.
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION=/path/to/vllm-<version>.whl

# Multi-node only. Every node's venv builder contends on one lock in the shared
# uv cache while vLLM's metadata is built; uv's default 300 s lock timeout can
# expire before the waiting nodes acquire it.
export UV_LOCK_TIMEOUT=3600
```

## Get the Weights

The recipe ships with a placeholder checkpoint path. Point the model and
tokenizer at your local checkpoint directory. The matching vLLM compatibility
config is already included in the repository:

```bash
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen3.8-flash-next-dapo-8n8g-automodel.yaml \
  policy.model_name=/your/path/to/qwen3.8-flash-next-180b \
  policy.tokenizer.name=/your/path/to/qwen3.8-flash-next-180b
```

or edit the checkpoint path in the YAML directly.

## Example Recipes

The recipe is DAPO-style GRPO on DAPO-Math-17K with AIME-2024 validation,
AutoModel (DTensor) training with colocated vLLM generation, on 8 nodes x 8 GPUs.
Recipe YAML files under `examples/configs/recipes/` are the source of truth.

The YAML defaults to the currently validated 4k launch configuration. A 9k
configuration may run out of memory.

| Validated seq | Training EP | Rollout TP/EP | `max_new_tokens` | Recipe |
|---|---|---|---|---|
| 4096 | 64 | 8/8 | 3072 | [`grpo-qwen3.8-flash-next-dapo-8n8g-automodel.yaml`](../../../../examples/configs/recipes/llm/grpo-qwen3.8-flash-next-dapo-8n8g-automodel.yaml) |

This is a DAPO-style recipe with overlong reward shaping, asymmetric clipping,
and token-level loss. Dynamic sampling is disabled in the current recipe.

> [!IMPORTANT]
> In the validated 4k launch, `max_new_tokens` is
> `max_total_sequence_length - data.max_input_seq_length` (4096 - 1024 = 3072).
> Keep that relationship if you change the sequence length. If
> `max_new_tokens` is larger than the context leaves room for, vLLM silently
> caps generation per sample at `max_model_len - input_length`, so the effective
> budget varies with prompt length instead of being uniform.

## Choose a Recipe

Use the 4k, 8-node launch below as the starting point for full-model training.

### Why context length matters here

> [!NOTE]
> Context-length comparisons will be added after reference runs are available.

## Launch

```bash
export NRL_FORCE_REBUILD_VENVS=true

uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen3.8-flash-next-dapo-8n8g-automodel.yaml \
  policy.model_name=/your/path/to/qwen3.8-flash-next-180b \
  policy.tokenizer.name=/your/path/to/qwen3.8-flash-next-180b
```

On Slurm, keep `cluster.num_nodes` in step with what you request from the scheduler.
A mismatch wedges Ray placement rather than failing cleanly:

```bash
uv run examples/run_grpo.py --config <recipe> cluster.num_nodes=8
```

## Reference Results

### Training curves

> [!NOTE]
> Reference training curves and metrics will be added after the production run
> completes.

## Known Issues

- **Long-run convergence is not validated.** Current evidence covers short runs only.
- **The 9k configuration may OOM.** Use the default 4k recipe unless
  additional memory headroom has been established for the target system.
