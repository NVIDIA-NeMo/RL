# Nemotron 3.5 Super VL

This page collects NeMo RL guidance for post-training Nemotron 3.5 Super VL
(`NemotronH_Omni_Reasoning_V3`), a hybrid Mamba + Attention MoE model with
512 routed experts (22 active) and a RADIO vision tower, on the AutoModel
(DTensor) backend. Use it to set up the environment, launch the DAPO math
recipe, and understand the settings that are specific to this model.

> [!IMPORTANT]
> **Early access.** The text-only DAPO recipe runs end-to-end (resume from
> checkpoint included) and validation accuracy rises over the first tens of
> steps, but no run has yet been taken to full convergence. Multimodal (image)
> RL on this model is not yet supported on the AutoModel path; see
> [Known Issues](#known-issues).

## Support Status

| Stage | Meaning |
| --- | --- |
| **Functionally Ready** | Runnable end-to-end and numerically validated with an initial training run. |
| **Long-Run Convergence Validated** | Trains stably over a full-length run with a healthy, reproducible reward curve. |

Nemotron 3.5 Super VL is **Functionally Ready** (text-only).

## What's Supported

| Model | Modality | Training backend | Parallelism | Inference | Precision |
| --- | --- | --- | --- | --- | --- |
| Nemotron 3.5 Super VL (120B-A12B) | LLM (text-only path) | AutoModel (DTensor) | FSDP2 + EP | vLLM (Super VL fork) | BF16 compute, FP32 master |

Notes:

- **Training** runs on the AutoModel (DTensor) backend with FSDP2 over all
  parameters and expert parallelism (DeepEP) for the routed experts. The Megatron
  backend for this model lives on the `super-v3.5-posttraining` branch and is not
  covered here.
- **Generation** uses a vLLM fork; stock vLLM 0.25.1 does not register the
  `NemotronH_Omni_Reasoning_V3` architecture. See
  [Build the Environment](#build-the-environment).
- The vision tower is constructed on both the training and generation side, but
  the recipe drives the model text-only: the tokenizer path (no processor) is
  used, so the DTensor worker runs with `is_vlm=false` and vLLM never receives
  images.

## Build the Environment

Published NeMo RL containers do not include the Nemotron 3.5 Super VL runtime.
Both AutoModel and vLLM come from pinned sources in this branch.

### 1. Pinned sources

- **AutoModel** — the submodule is pinned to
  [`6ae5ca23`](https://github.com/NVIDIA-NeMo/Automodel/commit/6ae5ca23895d1e06c81cc6103732a20510833fda)
  on `main`, which includes Nemotron 3.5 Super VL support
  ([#3801](https://github.com/NVIDIA-NeMo/Automodel/pull/3801)) and the
  `fc2_latent_proj` dtype fix
  ([#3828](https://github.com/NVIDIA-NeMo/Automodel/pull/3828)) that NeMo RL's
  FSDP mixed-precision policy requires. This pin moves the `automodel` extra to
  `transformers==5.15.1`.
- **vLLM** — `pyproject.toml` installs the `vllm` extra from the
  [`super_vl_rl_v0.25.1`](https://github.com/TomerBN-Nvidia/vllm/tree/super_vl_rl_v0.25.1)
  fork at `6f11a9c5` as a direct git requirement. The fork only changes the
  Python layer (architecture registration, transformers-native RADIO checkpoint
  layout, Mamba prefix-cache fixes), so the build reuses the upstream v0.25.1
  precompiled wheel (`VLLM_USE_PRECOMPILED=1`,
  `VLLM_PRECOMPILED_WHEEL_COMMIT=752a3a50`, the v0.25.1 tag on
  `wheels.vllm.ai`). Nothing to clone.
- **FlashInfer** stays on upstream 0.6.13.

```bash
git submodule update --init 3rdparty/Automodel-workspace/Automodel
```

### 2. Force a worker-venv rebuild

Ray worker virtual environments are cached, so they will not pick up the
AutoModel source or the `pyproject.toml` dependency bump unless you ask for a
rebuild:

```bash
export NRL_FORCE_REBUILD_VENVS=true

# Multi-node only. Every node's venv builder contends on one lock in the shared
# uv cache while vLLM's metadata is built.
export UV_LOCK_TIMEOUT=3600
```

> [!NOTE]
> The precompiled vLLM wheel is fetched from `https://wheels.vllm.ai` during
> `uv sync`/`uv lock`. If compute nodes have no external network, warm the shared
> `UV_CACHE_DIR` once from a node that does, or bake the rebuilt venvs into a
> container image (`enroot`/Pyxis `--container-save`) and launch from that image
> without `NRL_FORCE_REBUILD_VENVS`. After rebuilding inside a container,
> regenerate `/opt/nemo_rl_container_fingerprint` with
> `python tools/generate_fingerprint.py` so the version check passes.

## Get the Weights

The recipe defaults `policy.model_name` and `policy.tokenizer.name` to the
HF Hub checkpoint `nvidia/nemotron-3.5-super-pre-ea-text-08282026` (gated). To
use a local checkpoint instead, override both keys:

```bash
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/dapo-nemotron3.5-super-vl-120BA12B-16n8g-automodel.yaml \
  policy.model_name=/your/path/to/nemotron-3.5-super-vl \
  policy.tokenizer.name=/your/path/to/nemotron-3.5-super-vl
```

The checkpoint ships as 63 safetensors shards (~232 GB, BF16) with remote code
(`modeling_nemotron_h_omni.py`, `modeling_radio.py`); `trust_remote_code` is
always enabled by NeMo RL.

## Example Recipe

DAPO on DAPO-Math-17K with AIME-2024 validation, AutoModel (DTensor) training
with colocated vLLM generation. The recipe YAML under
`examples/configs/recipes/` is the source of truth.

| Algo | Seq | Train EP | vLLM TP/EP | `max_new_tokens` | Nodes | Recipe |
|---|---|---|---|---|---|---|
| DAPO | 9216 | 8 | 8 / 8 | 8192 | 16 x 8 GPUs | [`dapo-nemotron3.5-super-vl-120BA12B-16n8g-automodel.yaml`](../../../../examples/configs/recipes/llm/dapo-nemotron3.5-super-vl-120BA12B-16n8g-automodel.yaml) |

The recipe mirrors the Nemotron 3.5 Lightning DAPO recipe: dynamic sampling
(`batch_multiplier: 3`), Clip-Higher (`ratio_clip_max: 0.28`), overlong
filtering and soft overlong reward shaping, `reference_policy_kl_penalty: 0`,
FusedAdam with FP32 master weights, activation checkpointing, and
`moe_parallelizer.ignore_router_for_ac: true` (required with activation
checkpointing on this MoE: the BF16 router top-k is nondeterministic on
recompute).

### Model-specific settings

These are set in the recipe and are required for this model:

| Setting | Value | Why |
|---|---|---|
| `policy.hf_config_overrides.num_nextn_predict_layers` | `0` | The checkpoint ships one MTP layer (`mtp.*` tensors). AutoModel builds it by default; this drops it on the training side. Keep `generation.vllm_kwargs.speculative_config` unset for the same reason. |
| `policy.generation.vllm_cfg.skip_tokenizer_init` | `false` | vLLM's multimodal encoder budget calls the tokenizer during engine init for this architecture; the text-only default (`true`) fails with `You cannot pass text prompts when skip_tokenizer_init=True`. |
| `policy.generation.vllm_kwargs.skip_mm_profiling` | `true` | No images are ever sent; skip reserving encoder-cache memory. |
| `policy.generation.vllm_kwargs.mamba_ssm_cache_dtype` | `float32` | Matches the checkpoint's `mamba_ssm_cache_dtype`. |
| `policy.dtensor_cfg.automodel_kwargs.force_hf` | unset | The custom AutoModel implementation and its state-dict adapter are required for EP and per-tensor refit. |
| `policy.dtensor_cfg.env_vars.PYTORCH_CUDA_ALLOC_CONF` | unset | See [Known Issues](#known-issues): `expandable_segments:True` breaks the colocated IPC refit on some clusters. |

### Parallelism

- **Training EP must not exceed the GPUs per node.** The AutoModel DeepEP
  dispatcher (V1 `Buffer` API) assumes an expert-parallel group of up to 8 ranks
  is intranode and exchanges CUDA IPC memory handles. If the group spans nodes
  (for example EP=8 on 4-GPU GB200 nodes), `Buffer` initialization fails with
  `CUDA error ... deep_ep.cpp 'invalid resource handle'`. On 4-GPU nodes use
  `expert_parallel_size: 4`.
- **vLLM TP and EP live on the same GPUs.** `expert_parallel_size ==
  tensor_parallel_size` runs one engine per node with dense layers TP-sharded
  and experts EP-sharded (`enable_expert_parallel`). Set both to the GPUs per
  node.
- FSDP2 shards all parameters (including experts) over the full world, so
  persistent memory per GPU does not depend on EP; EP only changes the
  transient unsharded expert weights and all-to-all traffic.

### Memory

Per-GPU persistent state is FP32 master weights, FP32 gradients, and BF16 Adam
moments (about 1.45 TB total for 121B parameters), sharded over the world. At
64 GPUs that is about 23 GB per GPU plus activations; 4 nodes of 4 GPUs is the
practical minimum, and a single 4-GPU node OOMs at the first refit.

### Wall-clock and checkpointing

Generation dominates (about 85% of step time). On 64 GB200 GPUs a step takes
about 15 minutes with one generation batch and about 30 minutes when dynamic
sampling needs a second batch, which becomes common once validation accuracy
climbs. Checkpoints (FP32 master + optimizer) are about 1.8 TB each; the recipe
keeps `keep_top_k: 2` ranked by `val:accuracy`, and
`checkpoint_must_save_by: 00:03:20:00` leaves enough margin under a 4-hour
Slurm limit for the timeout save to complete. Resume is automatic from the
latest checkpoint in `checkpointing.checkpoint_dir`; chaining Slurm jobs with the
same name under `ray.sub`'s `--dependency=singleton` runs them back to back.

## Launch

```bash
export NRL_FORCE_REBUILD_VENVS=true

# DAPO, 16 nodes x 8 GPUs (recipe default)
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/dapo-nemotron3.5-super-vl-120BA12B-16n8g-automodel.yaml

# Same recipe on 4-GPU nodes (e.g. GB200 NVL72): 16 nodes x 4 GPUs, EP/TP = 4
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/dapo-nemotron3.5-super-vl-120BA12B-16n8g-automodel.yaml \
  cluster.gpus_per_node=4 \
  policy.dtensor_cfg.expert_parallel_size=4 \
  policy.generation.vllm_cfg.tensor_parallel_size=4 \
  policy.generation.vllm_cfg.expert_parallel_size=4
```

On Slurm, keep `cluster.num_nodes` and `cluster.gpus_per_node` in step with what
you request from the scheduler.

## Reference Results

TBD.

## Known Issues

- **`expandable_segments:True` breaks the colocated IPC refit on some
  clusters.** With that allocator setting PyTorch exports CUDA IPC handles as
  file descriptors fetched via `pidfd_getfd`; on clusters where that syscall path
  is unavailable the trainer-to-vLLM weight transfer fails with
  `pidfd_getfd: Bad file descriptor` and every key is reported missing. The
  recipe deliberately leaves `PYTORCH_CUDA_ALLOC_CONF` unset (default allocator,
  legacy `cudaIpcMemHandle` sharing).
- **Training EP across nodes is limited by DeepEP V1's 8-peer intranode
  assumption.** Larger EP on 4-GPU nodes needs the `hybridep` or `torch`
  dispatcher (`policy.dtensor_cfg.automodel_kwargs.backend.dispatcher`) or a
  DeepEP build with MNNVL enabled; not validated here.
- **Vision weights are not refit into vLLM.** The AutoModel Omni state-dict
  adapter's per-tensor HF conversion does not re-fuse the RADIO q/k/v
  projections for transformers-native RADIO checkpoints, so vLLM's vision tower
  keeps its dummy initialization. Harmless for the text-only recipe; blocks
  image RL until fixed.
- **Benign warning at load:** `Checkpoint key mismatch ... missing=80
  ...experts.{down_projs,gate_and_up_projs}`. The routed experts are written in
  place through strided views and the Omni adapter wrapper does not forward the
  in-place bookkeeping to the checkpoint loader; the weights are loaded.
- **No run has been taken to full convergence.**
