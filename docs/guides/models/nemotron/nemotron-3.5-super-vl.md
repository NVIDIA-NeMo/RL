# Nemotron 3.5 Super VL

This page collects NeMo RL guidance for text post-training of Nemotron 3.5
Super VL (`NemotronH_Omni_Reasoning_V3`), a hybrid Mamba + Attention MoE model
with 512 routed experts (22 active) and a RADIO vision tower, on the AutoModel
(DTensor) backend. Use it to launch the text DAPO recipe and understand the
settings that are specific to this model.

> [!IMPORTANT]
> **Early access.** The text DAPO recipe runs end-to-end on 16 x 4-GPU nodes
> with checkpoint resume and reaches 0.79 AIME-2024 accuracy by step 50 (see
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
  parameters and expert parallelism (DeepEP) for the routed experts. The
  Megatron backend for this model lives on the `super-v3.5-posttraining` branch
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
the `fc2_latent_proj` dtype fix
([#3828](https://github.com/NVIDIA-NeMo/Automodel/pull/3828)) that NeMo RL's
FSDP mixed-precision policy requires. The submodule pin on this branch includes
both. vLLM and the rest of the dependencies are the standard NeMo RL pins.

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
| `policy.generation.vllm_cfg.skip_tokenizer_init` | `false` | vLLM's multimodal encoder budget calls the tokenizer during engine init for this architecture; the text-only default (`true`) fails with `You cannot pass text prompts when skip_tokenizer_init=True`. |
| `policy.generation.vllm_kwargs.skip_mm_profiling` | `true` | No images are ever sent; skip vLLM's multimodal profiling pass and its encoder-cache reservation. |
| `policy.generation.vllm_kwargs.mamba_ssm_cache_dtype` | `float32` | Matches the checkpoint's `mamba_ssm_cache_dtype`. |
| `policy.dtensor_cfg.automodel_kwargs.force_hf` | unset | The custom AutoModel implementation and its state-dict adapter are required for EP and per-tensor refit. |
| `policy.dtensor_cfg.env_vars.PYTORCH_CUDA_ALLOC_CONF` | unset | See [Known Issues](#known-issues): `expandable_segments:True` breaks the colocated IPC refit on some clusters. |

### Parallelism

- **Training EP must not exceed the GPUs per node.** The AutoModel DeepEP
  dispatcher (V1 `Buffer` API) assumes an expert-parallel group of up to 8 ranks
  is intranode and exchanges CUDA IPC memory handles. If the group spans nodes
  (for example EP=8 on 4-GPU GB200 nodes), `Buffer` initialization fails with
  `CUDA error ... deep_ep.cpp 'invalid resource handle'`. The recipe uses
  `expert_parallel_size: 4` for 4-GPU nodes.
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

**Text DAPO, DAPO-Math-17K / AIME-2024** — 50 steps, `max_new_tokens: 8192`,
on a text-SFT early-access checkpoint of the model.

![Nemotron 3.5 Super VL text DAPO training curves](../../../assets/nemotron/nemotron-3.5-super-vl-text-dapo-16n4g.png)

AIME-2024 validation accuracy climbs from 0.53 at step 0 to **0.79 at step 50**
(0.55 at step 20, 0.69 at step 40) while the mean validation response length
falls from ~5,700 to ~4,600 tokens and `truncation_rate` drops from ~0.28 to
~0.15: the policy gets both more accurate and more concise. Training reward
rises from around -0.2 to a noisy 0.4-0.7 band. `gen_kl_error` stays in the
0.003-0.004 range and `token_mult_prob_error` in 1.03-1.04 throughout, so the
trainer and vLLM stay in agreement across the refits; the isolated
`token_mult_prob_error` spike at step 49 coincides with a job boundary and
resume. The chain resumed across five jobs.

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
- **Host memory is the limit for node counts below 16.** Per 4-GPU GB200 node
  (942 GB) the colocated run holds the pinned vLLM sleep-mode weight backups
  (~88 GiB per TP worker) plus, while the trainer is offloaded for generation,
  the sharded params + optimizer state of 4 ranks; the checkpoint save adds
  staging on top. On 8 nodes the first checkpoint save is OOM-killed on the
  host; 16 nodes peak at ~840-920 GB including the save.
- **Sequence packing and dynamic batching are not validated for this model.**
  The recipe keeps both disabled (`train_micro_batch_size: 1`). Sequence packing
  on AutoModel-native models currently loses packed-sequence boundaries
  (NeMo RL [#4167](https://github.com/NVIDIA-NeMo/RL/issues/4167)); keep it off.
- **Image post-training is out of scope for this pin.** Vision-weight refit and
  the Blackwell vision-encoder attention path need vLLM changes that are not in
  the pinned release; only the text path is supported from this branch.
