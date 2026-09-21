# DeepSeek V4.1 Flash

This guide describes text-only DAPO training of DeepSeek V4.1 Flash through the
GRPO entry point, using AutoModel for training and vLLM for generation.

> [!IMPORTANT]
> This is development support for the configuration described below. Functional
> training support does not establish long-run convergence or performance across
> other configurations. Use the pinned dependencies and review
> [Known Limitations](#known-limitations) before changing the recipe.

## Support Status

| Model | Training backend | Training parallelism | Generation backend | Status |
| --- | --- | --- | --- | --- |
| DeepSeek V4.1 Flash | AutoModel / FSDP2 | TP1 + CP1 + EP | vLLM with TP + EP | Development support |

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

Use the dependency lock and AutoModel submodule recorded by the NeMo RL
revision containing this guide. From the repository root:

```bash
git submodule update --init --recursive
uv sync --locked
```

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

Provision resources matching the YAML. Training and generation share devices;
review both parallel layouts before changing resource counts.

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

## Known Limitations

- This guide covers text-only AutoModel training with synchronous, colocated
  vLLM generation. It does not establish support for Megatron training,
  SGLang generation, multimodal inputs or other deployment layouts.
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
