# DeepSeek V4.1 Flash

This guide describes text-only DAPO training of DeepSeek V4.1 Flash through the
GRPO entry point, using AutoModel for training and vLLM for generation.

> [!IMPORTANT]
> **Early access.** Text-only, single-turn training has undergone short-run
> stability validation for approximately 50 steps at a 3K-token context length
> (3,072 tokens). This does not establish long-run convergence or performance
> across other configurations. Use the pinned dependencies and review
> [Known Limitations](#known-limitations) before changing the recipe.

## Support Status

| Model | Training backend | Training parallelism | Generation backend | Status |
| --- | --- | --- | --- | --- |
| DeepSeek V4.1 Flash | AutoModel / FSDP2 | TP1 + CP1 + EP | vLLM with TP + EP | Functionally Ready |

## Supported Scope

- **Algorithm and data**: DAPO through GRPO, with `DAPOMath17K` training data
  and `DAPOMathAIME2024Deduplicated` validation data.
- **Training**: BF16 compute with FP32 master parameters, activation
  checkpointing, TileLang attention and the Torch expert dispatcher.
- **Memory management**: FSDP CPU parameter offload, frozen host Engram tables
  and `BF16CPUAdamW` with BF16 moment storage and FP32 update arithmetic.
- **Generation**: Colocated vLLM with BF16 dense/expert weights, native MXFP8
  Engram tables and MXFP8 compressed KV. Router replay preserves rollout expert
  selections during training and logprob recomputation.
- **Inputs**: Text-only, with thinking mode disabled by the reference recipe.

The recipe YAML is the source of truth for resource counts, parallelism,
sequence lengths, batch sizes, evaluation and checkpoint settings.

## How to Run

### 1. Prepare the Environment and Checkpoint

This early-access support requires the
[`deepseek-v4.1-support` branch](https://github.com/NVIDIA-NeMo/RL/tree/deepseek-v4.1-support).
Switch to that branch before initializing submodules or installing dependencies.
For a clone of `NVIDIA-NeMo/RL` with the upstream repository named `origin`,
run from the repository root:

```bash
git fetch origin deepseek-v4.1-support
git switch deepseek-v4.1-support
git merge --ff-only origin/deepseek-v4.1-support
git submodule update --init --recursive
uv sync --locked
```

Use the dependency lock and AutoModel submodule pinned by this branch.
If `origin` points to a fork, use your upstream remote in the fetch and merge
commands, and create a local tracking branch from it on the first checkout.

For distributed execution, use a matching CUDA environment on every node.
See [Installation](../../../about/installation.md),
[Dependency Management](../../../design-docs/dependency-management.md) and
[Cluster Setup](../../../cluster.md) for environment and launch setup.

Provide a local DeepSeek V4.1 Flash checkpoint containing its original config,
tokenizer, safetensors index and weight shards:

```bash
export DS41_FULL_CHECKPOINT=/path/to/verified/full-checkpoint
export HF_HOME=/path/to/shared-huggingface-cache
```

The checkpoint must be accessible at the same path on every training and
generation node. Keep the original Engram weight and scale bytes immutable:
the host lookup and rollout initialization both use this source.
Make the configured DAPO/AIME datasets available through the shared cache or
dataset download access. Configure W&B externally, or disable it at launch.

### 2. Choose the Reference Recipe

| Model | Algorithm | Deployment | Recipe |
| --- | --- | --- | --- |
| DeepSeek V4.1 Flash | DAPO | Colocated AutoModel + vLLM | [Streaming CPU Adam recipe](../../../../examples/configs/recipes/llm/dapo-deepseek-v4.1-12n4g-fsdp2tp1-ep48-colocated-stream-adam.yaml) |
| DeepSeek V4.1 Flash | DAPO | Asynchronous, non-colocated AutoModel + vLLM (one-step staleness) | [Async non-colocated recipe](../../../../examples/configs/recipes/llm/dapo-deepseek-v4.1-16n4g-fsdp2tp1-ep48-noncolocated-async-stream-adam.yaml) |

Provision resources matching the YAML. In the colocated recipe training and
generation share devices; in the asynchronous variant
`policy.generation.colocated.resources.num_nodes` generation nodes are carved
out of `cluster.num_nodes`. Review both parallel layouts before changing
resource counts.

### 3. Launch

From the repository root in the prepared distributed environment:

```bash
bash examples/run_deepseek_v41_stream_adam.sh logger.wandb_enabled=false
```

The launcher accepts ordinary configuration overrides. The equivalent direct
entry point is:

```bash
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/dapo-deepseek-v4.1-12n4g-fsdp2tp1-ep48-colocated-stream-adam.yaml \
  logger.wandb_enabled=false
```

See the [DAPO guide](../../dapo.md) and [GRPO guide](../../grpo.md) for algorithm
settings. The launcher does not allocate nodes or start the cluster for you.

## Important Recipe Settings

- **Frozen Engram**: `policy.hf_config_overrides.text_config.engram_host_checkpoint`
  and `policy.generation.vllm_cfg.frozen_engram_checkpoint` resolve to the same
  checkpoint. Only the lookup table is frozen; Engram projections and gates
  remain trainable. Selected rows are decoded on CPU and transferred to GPU.
  The table is excluded from trainer refit payloads.
- **Streaming optimizer**: `policy.dtensor_cfg.cpu_offload: true` is required by
  `BF16CPUAdamW`. Selecting this optimizer enables GPU gradient accumulation,
  scaling and clipping, followed by chunked CPU updates. Separate gradient
  offload, norm and streaming environment switches are not required.
- **Router replay**: `policy.router_replay.enabled: true` records vLLM expert IDs
  and reuses them for logprob and training forwards, including checkpoint
  recomputation. Router scores and mixing weights are recomputed and remain
  differentiable. Keep TP1, CP1 and sequence packing disabled for this path.
- **Generation precision**: Keep BF16 rollout, `expert_dtype: bf16` and the
  rollout `quantization_config: null` override together. This does not remove
  the original checkpoint's quantization metadata on the training side.
  Engram and compressed KV retain their MXFP8 representation.
- **Refit and sleep**: Keep `refit_with_reload_api: false`. CPU parameters are
  staged individually for colocated IPC refit. With `sleep_level: 2`, vLLM
  discards weights; the synchronous GRPO loop marks generation stale after
  rollout or validation so the next generation first refits the policy.
- **Checkpoint restore**: Training checkpoints contain model parameters and
  optimizer state, but not the external frozen Engram table. Restore requires
  the same table source and compatible optimizer shard layout.

## Asynchronous Non-Colocated Variant

`dapo-deepseek-v4.1-16n4g-fsdp2tp1-ep48-noncolocated-async-stream-adam.yaml`
runs the same DAPO recipe with asynchronous generation on dedicated nodes:

- **Layout**: 12 training nodes (EP48, the reference trainer topology) plus
  4 generation nodes, 16 GB200 nodes with 4 GPUs each. One vLLM TP8/EP8 engine
  needs more than 1 TiB of HBM, i.e. two GB200 nodes, and
  `cluster.segment_size: 2` keeps each engine inside one NVLink domain.
- **Keys that differ from the reference recipe**: `grpo.async_grpo.enabled: true`
  with `max_trajectory_age_steps: 1`, `policy.generation.colocated.enabled: false`
  (`resources.num_nodes` is the number of generation nodes) and
  `policy.generation.vllm_cfg.async_engine: true`. Generation runs with CUDA
  graphs (`enforce_eager: false`, no Inductor), `gpu_memory_utilization: 0.90`
  and `max_num_seqs: 96`, because two engines serve all 192 samples of a step.
  `loss_fn.use_importance_sampling_correction` is already `true`.
- **Reward scaling and shaping**: `async_grpo_train` on this branch applies
  `scale_rewards` and `apply_reward_shaping` before the advantage computation,
  at the same point as the synchronous loop, so the DAPO settings of the
  reference recipe stay in effect and `examples/run_grpo.py` no longer rejects
  them in async mode. Dynamic sampling remains unsupported in async mode.
- **Refit**: weights reach vLLM through the collective refit path. With FSDP
  CPU offload each rank onloads its shard, broadcasts it and offloads it again,
  about 14 s per step on GB200.
- **Measured** on 16 x 4 GB200 at branch commit `8e056c2f`, 48 steps at GBS 192,
  validation every 10 steps on the same prompts as the colocated run:

  | Step | Asynchronous non-colocated | Colocated synchronous |
  | --- | --- | --- |
  | 10 | 0.4792 | 0.4792 |
  | 20 | 0.5042 | 0.4958 |
  | 30 | 0.5750 | 0.5792 |
  | 40 | 0.6333 | 0.6125 |

  The generation-KL error stays at 0.0005 to 0.0006. A step takes 127 to 644 s
  (median about 220 s, training plus logprob recomputation; exposed generation
  at most 1.4 s, weight sync about 14 s) against 857 to 1283 s per step for the
  colocated recipe as shipped. Ten further steps with
  `in_flight_weight_updates: true` ran clean (step-10 accuracy 0.4500); the
  switch stays off in the recipe pending a longer run.
- **Checkpointing**: the 12 training nodes of this layout cannot write a
  checkpoint — the save copies the CPU-offloaded optimizer state on top of a
  650–708 GB host-memory steady state and the nodes run out of memory, as the
  colocated 12-node recipe does. With 16 training nodes
  (`policy.dtensor_cfg.expert_parallel_size: 64`, `cluster.num_nodes: 18`,
  generation `resources.num_nodes: 2`, `max_num_seqs: 192`) the same recipe ran
  all 50 steps with a checkpoint and a validation every 10 steps in one
  3 h 47 min job: AIME-2024 0.4417 / 0.5125 / 0.5625 / 0.6250 / 0.6500 at
  steps 10–50, five 4.4 TB saves in 82–133 s that include `replay_buffer.pt`,
  training-node host peak 689 GB, exposed generation still below 1 s with one
  engine. A resume from `step_50` restored the replay buffer and trained two
  further steps.

## Reference Training Curves

The following panels show raw, unsmoothed W&B metrics through training step 47
for the text-only, single-turn recipe with a 3,072-token context limit.
They illustrate short-run behavior rather than long-run convergence.

![DeepSeek V4.1 Flash DAPO training reward, AIME validation accuracy, response length, entropy, generation KL error and gradient norm through step 47](../../../assets/deepseek/deepseek-v4.1-flash-dapo-training.png)

For repeated training steps after resume, the last logged value is shown.
Validation markers show the recorded evaluation points only.

## Known Limitations

- Validation is limited to text-only, single-turn short runs of approximately
  50 steps at a 3K-token context length. Multi-turn training, longer contexts
  and long-run stability have not been established by this validation.
- This guide covers text-only AutoModel training with synchronous, colocated
  vLLM generation and the asynchronous non-colocated variant above. It does
  not establish support for Megatron training, SGLang generation, multimodal
  inputs or other deployment layouts.
- AutoModel router replay currently rejects sequence packing, training TP/CP
  greater than one and a separate reference policy.
- Frozen host Engram requires BF16 rollout and does not support full-model FP8
  layerwise reload. Changing the table source during training is unsupported.
- The streaming gradient path depends on a checked PyTorch FSDP internal hook.
  Unsupported source layouts fail explicitly; upgrading Torch or AutoModel
  requires compatibility validation.
- BF16 moment storage changes optimizer rounding relative to FP32-state AdamW.
  CPU updates also introduce host-memory and transfer costs; no general
  throughput claim is made for this configuration.
- Optimizer restore checks dtype, shape and distributed layout. The optimizer's
  loader does not itself provide resharding across different layouts.
