# Muse Glimmer

This page collects NeMo RL guidance for post-training Muse Glimmer, a ~29.8B model
built on the Onyx architecture. Use it to set up the environment, choose a starting
recipe, and understand the settings that are specific to this model.

> [!IMPORTANT]
> **Early access.** Muse Glimmer runs end-to-end in NeMo RL and short GRPO/DAPO runs
> have been numerically validated, but long-run convergence has not been established.
> The AutoModel dependency is a private repository (see
> [Build the Environment](#build-the-environment)), so this branch is not yet usable
> without access to it.

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
- **Generation** runs on vLLM with tensor parallel 4. Onyx has 2 KV heads and vLLM's
  `OnyxAttention` allows `tp_size % num_kv_heads == 0` by replicating KV heads, so
  TP4 is legal even though 4 does not divide 2.
- **The vLLM build is text-only.** It warns and skips the vision tower; the recipes
  set `policy.generation.vllm_kwargs.language_model_only: true` to match, and freeze
  the vision and audio towers on the training side.

## Build the Environment

Published NeMo RL containers do not include the Muse Glimmer runtime. Both AutoModel
and vLLM must come from local sources.

### 1. Clone the sources into `3rdparty/`

Sources:

- **AutoModel** — a git submodule on this branch, pinned to the branch carrying the
  Onyx training path. The repository is private, cloned over SSH; make sure your key
  is loaded (`ssh-add -l`) first. See
  [Experiment with Custom vLLM](../use-custom-vllm.md#ssh-setup-for-private-repositories)
  if you need to set up an agent.
- **vLLM** — [main branch](https://github.com/vllm-project/vllm), where Muse Glimmer
  support is merged.

```bash
git submodule update --init 3rdparty/Automodel-workspace/Automodel

mkdir -p 3rdparty/vLLM-workspace
git clone https://github.com/vllm-project/vllm.git 3rdparty/vLLM-workspace/vllm
```

`pyproject.toml` installs vLLM from `3rdparty/vLLM-workspace/vllm` as an editable
source, so `uv sync` fails if that directory is empty.

If you check out a vLLM revision other than the one `uv.lock` was resolved against,
re-run `uv lock` and keep the `flashinfer-*` and `nvidia-cutlass-dsl` pins in the
`vllm` extra in sync with that revision's `requirements/cuda.txt`.

### 2. Force a worker-venv rebuild

Ray worker virtual environments are cached, so they will not pick up the local
AutoModel and vLLM sources unless you ask for a rebuild:

```bash
export NRL_FORCE_REBUILD_VENVS=true

# Skip vLLM's CUDA build. Without this the editable install runs cmake, whose
# FetchContent step clones cutlass from GitHub -- which fails on compute nodes
# with no external network ("Failed to clone repository"). Stage the wheel on a
# shared filesystem first; a URL only works if the compute nodes can reach it.
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION=/path/to/vllm-<version>-cp38-abi3-manylinux_2_28_x86_64.whl

# Multi-node only. Every node's venv builder contends on one lock in the shared
# uv cache while vLLM's metadata is built; uv's default 300 s lock timeout is
# shorter than that takes, and the waiting nodes die with "Failed to acquire
# lock on the distribution cache".
export UV_LOCK_TIMEOUT=3600
```

Skipping the CUDA build is safe because Muse Glimmer support is pure Python on
top of upstream vLLM — `git diff` of `csrc/`, `CMakeLists.txt` and `cmake/`
against the upstream base commit is empty, and that base already requires torch
2.13, so the prebuilt extensions link the same ABI. Take the wheel for the
upstream base commit from `https://wheels.vllm.ai/<commit>/`, and re-stage it
whenever you move the vLLM checkout.

> [!IMPORTANT]
> **The first rebuild is slow on a cold cache, and the venvs belong on
> node-local disk.** This branch is on torch 2.13, which no prebuilt
> `flash-attn` wheel targets, so `flash-attn`, Transformer Engine, `mamba-ssm`,
> `causal-conv1d` and `nv-grouped-gemm` compile from source the first time. On a
> shared filesystem that is pathologically slow: measured throughput on Lustre
> was ~0.4 MB/s, against 257 packages in 16 s on node-local disk. Point
> `UV_PROJECT_ENVIRONMENT` at local disk (or run inside a writable container,
> where `/opt/ray_venvs` is already node-local) and keep `UV_CACHE_DIR` on the
> shared filesystem so the built wheels persist across jobs. With a warm cache a
> full 4-node rebuild plus two training steps takes about 15 minutes.
>
> Once a run has built the venvs, save the container so later jobs reuse them
> instead of rebuilding — `_env_builder` early-returns when the venv's `python`
> already exists, so a baked container skips this step entirely.

> [!WARNING]
> Keep the **policy** worker venv on whatever `nemo-automodel` pins for
> `transformers`. It is tempting to force the vLLM build's newer `transformers` into
> every worker venv, but AutoModel's Onyx forward depends on transformers internals;
> a mismatched policy venv produces `gen_kl_error` around 3.5 **before any weight
> update**, which silently destroys the checkpoint over the following steps. The
> vLLM/transformers pairing evidence comes from generation tests, where both live in
> the same venv — it does not transfer to the policy venv.

## Get the Weights

The recipes ship with a placeholder path. Point both keys at your local checkpoint
directory:

```bash
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/dapo-muse-glimmer-29b-8n8g-fsdp2cp8-automodel-16k.yaml \
  policy.model_name=/your/path/to/muse-glimmer-29b \
  policy.tokenizer.name=/your/path/to/muse-glimmer-29b
```

or edit `policy.model_name` and `policy.tokenizer.name` in the YAML directly.

## Example Recipes

All three are DAPO on DAPO-Math-17K with AIME-2024 validation, 8 nodes x 8 GPUs,
AutoModel (DTensor) training with colocated vLLM generation. They differ only in
sequence length and context-parallel degree. Recipe YAML files under
`examples/configs/recipes/` are the source of truth.

| Sequence length | CP | Tokens/rank | vLLM `gpu_memory_utilization` | Recipe |
|---|---|---|---|---|
| 4096 | 2 | 2048 | 0.5 | [`dapo-muse-glimmer-29b-8n8g-fsdp2cp2-automodel-4k.yaml`](../../../examples/configs/recipes/llm/dapo-muse-glimmer-29b-8n8g-fsdp2cp2-automodel-4k.yaml) |
| 8192 | 4 | 2048 | 0.5 | [`dapo-muse-glimmer-29b-8n8g-fsdp2cp4-automodel-8k.yaml`](../../../examples/configs/recipes/llm/dapo-muse-glimmer-29b-8n8g-fsdp2cp4-automodel-8k.yaml) |
| 16384 | 8 | 2048 | 0.6 | [`dapo-muse-glimmer-29b-8n8g-fsdp2cp8-automodel-16k.yaml`](../../../examples/configs/recipes/llm/dapo-muse-glimmer-29b-8n8g-fsdp2cp8-automodel-16k.yaml) |

## Choose a Recipe

**Start with the 16k (cp8) recipe.** Context length is the single biggest quality
lever for this model. Use 8k or 4k only when you do not have the GPU-hours: the
throughput cost of 16k is real, roughly 8 steps/hour versus 25 at cp4/8k.

### Why context length matters here

Muse Glimmer produces long reasoning traces, and DAPO applies an overlong penalty to
truncated rollouts. When the sequence budget binds, the policy learns to stop early
rather than to reason better, and reward decouples from correctness.
`train/truncation_rate` at step 5 falls **0.553 / 0.231 / 0.076** across the 4k / 8k /
16k recipes — at 4k more than half of all rollouts are truncated.

> [!NOTE]
> Diagnose truncation pressure with `train/truncation_rate` and
> `train/max_gen_tokens_per_sample`, **not** mean generation length. DAPO's overlong
> penalty makes the mean generation length *fall* when the cap binds, which reads
> like "the model does not need the tokens" and means the opposite.

### Why every recipe sits at 2048 tokens per rank

The head is unusually memory-hungry: the vocabulary is 202048 entries and the final
soft cap in AutoModel's Onyx `model.py` runs

```python
logits = soft_cap * torch.tanh(logits.float() * multiplier / soft_cap)
```

which materializes roughly four full-vocab fp32 tensors — about 4.67 GiB each at
~6.2k tokens. Without context parallelism the training backward OOMs at both 8192 and
16384, and **more nodes does not help**: under FSDP2 the per-GPU peak is set by tokens
in the largest microbatch, a single sequence cannot be split across ranks, and
sharding only shrinks parameters and optimizer state.

Context parallelism is the lever that does work, because it splits a single sequence
across ranks. But it is **not memory neutral**: Transformer Engine's ring-attention
backward keeps per-CP-step `dq`/`dk`/`dv` buffers, roughly `cp_size` copies. `cp4` at
16384 — 4096 tokens/rank, the same per-rank load that fits without CP — OOMs in
`fused_attn_bwd`. Budget about **half** the per-rank tokens you would use at `cp=1`.

CP also constrains the rest of the config: it is rejected for VLMs and for sequence
packing, and it forces `attn_implementation=sdpa`. The recipes already set
`policy.sequence_packing.enabled: false` and `attn_implementation: sdpa`; leave both
alone.

## Launch

```bash
export NRL_FORCE_REBUILD_VENVS=true

# 16k, CP8 -- recommended default
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/dapo-muse-glimmer-29b-8n8g-fsdp2cp8-automodel-16k.yaml \
  policy.model_name=/your/path/to/muse-glimmer-29b \
  policy.tokenizer.name=/your/path/to/muse-glimmer-29b

# 8k, CP4
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/dapo-muse-glimmer-29b-8n8g-fsdp2cp4-automodel-8k.yaml \
  policy.model_name=/your/path/to/muse-glimmer-29b \
  policy.tokenizer.name=/your/path/to/muse-glimmer-29b

# 4k, CP2 -- low-memory fallback
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/dapo-muse-glimmer-29b-8n8g-fsdp2cp2-automodel-4k.yaml \
  policy.model_name=/your/path/to/muse-glimmer-29b \
  policy.tokenizer.name=/your/path/to/muse-glimmer-29b
```

On Slurm, keep `cluster.num_nodes` in step with what you request from the scheduler.
A mismatch wedges Ray placement rather than failing cleanly:

```bash
uv run examples/run_grpo.py --config <recipe> cluster.num_nodes=8
```

### Check the run is healthy before letting it run long

`train/gen_kl_error` measures whether the policy and the generation stack agree on the
same distribution. On a healthy run it sits around **5e-4** from step 1 and drifts
only slowly. A value of **1 or more at step 1** means the very first refit disagreed:
training is optimising against meaningless logprobs and will destroy the checkpoint.

Run 2 steps and read `train/gen_kl_error` after **any** container or dependency
change. It takes minutes. A generation-only smoke test cannot catch this, because it
never exercises the policy path.

> [!NOTE]
> Read metrics from Weights & Biases or the TensorBoard event files, not from the Ray
> driver log. `RAY_LOG_SYNC_FREQUENCY` re-syncs the driver log from the head node, so
> a mid-run read can show step counts and values that vanish a minute later.

## Reference Results

Smoke run on the released checkpoint, 4 nodes x 8 GPUs, the 4k CP2 recipe, 2 steps,
worker venvs rebuilt from this branch:

| Metric | Value |
| --- | --- |
| `train/gen_kl_error` | 5.43e-04 |
| `train/token_mult_prob_error` | 1.025 |
| `train/truncation_rate` (4k) | 0.471 |

`gen_kl_error` at 5.43e-04 is the number that matters: it says the policy and the
generation stack agree, so training is optimising against meaningful logprobs.

*Long-run training curves and validation accuracy will be added once a full run on
the released checkpoint completes.*

## Known Issues

- **Long-run convergence is not validated.** Current evidence covers short runs only.
- **The AutoModel dependency is a private repository.** `3rdparty/Automodel-workspace/Automodel`
  points at a private fork over SSH; cloning it requires access. This must be
  repointed at [NVIDIA-NeMo/Automodel](https://github.com/NVIDIA-NeMo/Automodel)
  before the branch can be used publicly.
- **Never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** with colocated
  vLLM generation. vLLM's sleep mode needs the cumem allocator, so every engine worker
  overrides the setting with `_set_allocator_settings("expandable_segments:False")`
  and then dies in `compile_or_warm_up_model`. The variable is process-global and
  cannot be scoped to the policy worker alone.
- **Two benign log lines, do not chase them.** `undefined symbol: _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE`
  from optional prebuilt kernels (vLLM's soft-import probe logs it at WARNING and
  falls back correctly), and `CUDA Error: invalid argument at cumem_allocator.cpp`
  emitted after "Max number of steps has been reached", i.e. during vLLM teardown.
