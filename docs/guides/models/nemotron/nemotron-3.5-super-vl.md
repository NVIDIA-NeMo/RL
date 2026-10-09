# Nemotron 3.5 Super VL

This page collects NeMo RL guidance for text post-training of Nemotron 3.5
Super VL (`NemotronH_Omni_Reasoning_V3`), a hybrid Mamba + Attention MoE model
with 512 routed experts (22 active) and a RADIO vision tower, on the AutoModel
(DTensor) backend. Use it to launch the text DAPO recipe and understand the
settings that are specific to this model.

> [!IMPORTANT]
> **Early access.** The text DAPO recipe runs end-to-end on 16 x 4-GPU nodes
> with checkpoint resume and reaches 0.76 AIME-2024 accuracy by step 40 (see
> [Reference Results](#reference-results)). No run has been taken to full
> convergence yet, and fewer than 16 nodes is not supported; see
> [Known Issues](#known-issues). Image (VLM) post-training for this model needs
> a newer vLLM than the one pinned here and is not covered by this page.

## Support Status

| Stage | Meaning |
| --- | --- |
| **Functionally Ready** | Runnable end-to-end and numerically validated with an initial training run. |
| **Long-Run Convergence Validated** | Trains stably over a full-length run with a healthy, reproducible reward curve. |

Nemotron 3.5 Super VL is **Functionally Ready** for text (DAPO) post-training.

## What's Supported

| Model | Modality | Training backend | Parallelism | Inference | Precision |
| --- | --- | --- | --- | --- | --- |
| Nemotron 3.5 Super VL (120B-A12B) | LLM (text) | AutoModel (DTensor) | FSDP2 + EP | vLLM | BF16 compute, FP32 master |

Notes:

- **Training** runs on the AutoModel (DTensor) backend with FSDP2 over all
  parameters and expert parallelism (HybridEP dispatcher) for the routed
  experts. The Megatron backend for this model lives on the
  `super-v3.5-posttraining` branch
  (Megatron-Bridge `3961f399` or later registers `NemotronH_Omni_Reasoning_V3`)
  and is not covered here.
- **Generation** uses the pinned stock vLLM, which registers
  `NemotronH_Omni_Reasoning_V3` and loads the checkpoint's RADIO weights from
  disk (`load_format=auto`) for the vision tower that the text recipe never
  exercises. The language model is refit from the trainer every step.
- The DAPO recipe drives the model text-only: the tokenizer path (no processor)
  is used, so the DTensor worker runs with `is_vlm=false` and vLLM never
  receives images.

## Environment

AutoModel support for this model landed in the `r0.6.0` branch with
[#3874](https://github.com/NVIDIA-NeMo/Automodel/pull/3874) (cherry-pick of
[#3801](https://github.com/NVIDIA-NeMo/Automodel/pull/3801)); it also carries
the `fc2_latent_proj` dtype fix (part of the same #3801 squash) that NeMo RL's
FSDP mixed-precision policy requires, and
[#4211](https://github.com/NVIDIA-NeMo/Automodel/pull/4211), which keeps the
Mamba `A_log` / `dt_bias` / `D` parameters in fp32 storage after FSDP2
sharding (required for checkpoint resume on torch >= 2.11). The Automodel
submodule pin on NeMo RL `main` (`b916107a5`) includes all of them. vLLM and the
rest of the dependencies are the standard NeMo RL pins.

Containers built before this pin ship worker venvs whose editable
`nemo_automodel` still points at the image's older Automodel checkout, and the
per-job `uv sync` does not re-point it. Check out the submodule and force a
worker-venv rebuild on the first run (it reinstalls from the image's uv cache,
about one to two minutes per node, no downloads):

```bash
git submodule update --init 3rdparty/Automodel-workspace/Automodel
export NRL_FORCE_REBUILD_VENVS=true
export UV_LOCK_TIMEOUT=3600   # multi-node: venv builders share one uv cache lock
```

Without the rebuild the policy worker falls back to the generic transformers
loader and fails at model construction with
`NemotronH_Omni_Reasoning_V3.__init__() got an unexpected keyword argument
'num_nextn_predict_layers'`.

> [!NOTE]
> After rebuilding inside a container you intend to save, regenerate
> `/opt/nemo_rl_container_fingerprint` with `python tools/generate_fingerprint.py`
> so the version check passes without `NRL_FORCE_REBUILD_VENVS`.

## Example Recipe

AutoModel (DTensor) training with colocated vLLM generation. The recipe YAML
under `examples/configs/recipes/` is the source of truth.

| Algo | Data | Seq | Train EP | vLLM TP/EP | `max_new_tokens` | Nodes | Recipe |
|---|---|---|---|---|---|---|---|
| DAPO (text) | DAPO-Math-17K / AIME-2024 | 9216 | 4 | 4 / 4 | 8192 | 16 x 4 GPUs | [`dapo-nemotron3.5-super-vl-120BA12B-16n4g-automodel.yaml`](../../../../examples/configs/recipes/llm/dapo-nemotron3.5-super-vl-120BA12B-16n4g-automodel.yaml) |

The recipe is sized for 16 x 4-GPU GB200 nodes: train `expert_parallel_size: 4`
and vLLM `tensor_parallel_size: 4` / `expert_parallel_size: 4` (one vLLM engine
per node). See [Parallelism](#parallelism) for why EP must stay within a node.

It mirrors the Nemotron 3.5 Lightning DAPO recipe: dynamic sampling
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
| `policy.generation.vllm_kwargs.skip_mm_profiling` | `true` | No images are ever sent; skip vLLM's multimodal profiling pass and its encoder-cache reservation. |
| `policy.generation.vllm_kwargs.mm_processor_cache_gb` | `0` | Drop vLLM's default 4 GiB host-side multimodal processor cache; host memory is the tight resource for this recipe. |
| `policy.generation.vllm_kwargs.limit_mm_per_prompt.image` | `1` | vLLM sizes the encoder budget from this limit, and an unset modality defaults to 999 images per prompt. |
| `policy.generation.vllm_kwargs.mamba_ssm_cache_dtype` | `float32` | Matches the checkpoint's `mamba_ssm_cache_dtype`. |
| `policy.generation.vllm_cfg.skip_tokenizer_init` | (automatic) | vLLM's multimodal encoder budget calls the tokenizer during engine init for this architecture. `NemotronH_Omni_Reasoning_V3` is listed in `TOKENIZER_REQUIRED_ARCHITECTURES`, so NeMo RL keeps the tokenizer regardless of the text-only default. |
| `policy.dtensor_cfg.automodel_kwargs.force_hf` | unset | The custom AutoModel implementation and its state-dict adapter are required for EP and per-tensor refit. |
| `policy.dtensor_cfg.env_vars.PYTORCH_CUDA_ALLOC_CONF` | unset | See [Known Issues](#known-issues): `expandable_segments:True` breaks the colocated IPC refit on some clusters. |

### Parallelism

- **Training EP stays within a node.** The recipe uses `expert_parallel_size: 4`
  on 4-GPU GB200 nodes with the HybridEP dispatcher
  (`policy.dtensor_cfg.automodel_kwargs.backend.dispatcher: hybridep`), which
  requires `make_sequence_length_divisible_by: 64`. The DeepEP dispatcher also
  works at EP=4 but its V1 `Buffer` API assumes an expert-parallel group of up to
  8 ranks is intranode; EP=8 on 4-GPU nodes fails with
  `CUDA error ... deep_ep.cpp 'invalid resource handle'`.
- **vLLM TP and EP live on the same GPUs.** `expert_parallel_size ==
  tensor_parallel_size` runs one engine per node with dense layers TP-sharded
  and experts EP-sharded (`enable_expert_parallel`). Set both to the GPUs per
  node.
- FSDP2 shards all parameters (including experts) over the full world, so
  persistent memory per GPU does not depend on EP; EP only changes the
  transient unsharded expert weights and all-to-all traffic.

## Launch

The recipe defaults to 16 nodes x 4 GPUs.

```bash
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/dapo-nemotron3.5-super-vl-120BA12B-16n4g-automodel.yaml
```

Resume is automatic from the latest checkpoint in `checkpointing.checkpoint_dir`;
chaining Slurm jobs with the same name under `ray.sub`'s
`--dependency=singleton` runs them back to back. On Slurm, keep
`cluster.num_nodes` and `cluster.gpus_per_node` in step with what you request
from the scheduler.

## Reference Results

### Training curves

The run uses the recipe defaults (16 x 4-GPU GB200 nodes, train EP4, vLLM
TP4/EP4) with the AutoModel (DTensor) backend and colocated vLLM generation,
chained as 4-hour Slurm jobs that resume from the latest checkpoint. Curves are
wandb exports; the x axis is the training step.

**Text DAPO, DAPO-Math-17K / AIME-2024** — 43 steps, `max_new_tokens: 8192`,
starting from `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`.

![Nemotron 3.5 Super VL text DAPO training curves](../../../assets/nemotron/nemotron-3.5-super-vl-text-dapo-16n4g.png)

AIME-2024 validation accuracy climbs from 0.41 at step 0 to **0.76 at step 40**
(0.45 at step 10, 0.48 at step 20, 0.59 at step 30) while the mean validation
response length falls from ~6,450 to ~4,600 tokens and `truncation_rate` drops
from ~0.40 to ~0.15: the policy gets both more accurate and more concise.
Training reward rises from a noisy -0.6 to -0.1 band to 0.2-0.6 after step 20.
`gen_kl_error` drifts slowly from ~0.003 to ~0.0045 as the policy sharpens, the
same order as the drift documented for other models; the isolated
`token_mult_prob_error` spikes (steps 20, 32, 38) are single-step outliers and
`token_mult_prob_error` returns to ~1.03 on the next step. The chain ran as five
4-hour Slurm jobs that resumed from the latest checkpoint.

## Known Issues

- **`expandable_segments:True` breaks the colocated IPC refit on some
  clusters.** With that allocator setting PyTorch exports CUDA IPC handles as
  file descriptors fetched via `pidfd_getfd`; on clusters where that syscall path
  is unavailable the trainer-to-vLLM weight transfer fails with
  `pidfd_getfd: Bad file descriptor` and every key is reported missing. The
  recipe deliberately leaves `PYTORCH_CUDA_ALLOC_CONF` unset (default allocator,
  legacy `cudaIpcMemHandle` sharing).
- **Training EP larger than the GPUs per node is not validated.** The recipe
  keeps EP=4 on 4-GPU nodes; cross-node EP with HybridEP has not been tested
  for this model.
- **Host memory is the limit for node counts below 16.** Per 4-GPU GB200 node
  (942 GB) the colocated run holds the pinned vLLM sleep-mode weight backups
  (~88 GiB per TP worker) plus, while the trainer is offloaded for generation,
  the sharded params + optimizer state of 4 ranks; the checkpoint save adds
  staging on top. On 8 nodes the first checkpoint save is OOM-killed on the
  host; 16 nodes peak at ~840-920 GB including the save.
- **Sequence packing is not validated for this model.** The recipe keeps it
  disabled; sequence packing on AutoModel-native models currently loses
  packed-sequence boundaries (NeMo RL
  [#4167](https://github.com/NVIDIA-NeMo/RL/issues/4167)). Dynamic batching is
  enabled with the inherited 9216-token microbatch budget.
- **Image post-training is out of scope for this pin.** Vision-weight refit and
  the Blackwell vision-encoder attention path need vLLM changes that are not in
  the pinned release; only the text path is supported on NeMo RL `main`.
