# Training Backends

NeMo RL supports multiple training backends to accommodate different model sizes and hardware configurations.

## Available Backends

- **DTensor (FSDP2)** - PyTorch's next-generation distributed training with improved memory efficiency.
- **Megatron** - NVIDIA's high-performance training framework for scaling to large models (>100B parameters).

## Supported Input Checkpoint Format

At this time, NeMo RL only supports Hugging Face checkpoints as inputs to the training scripts. This applies to both
the `DTensor` backend and the `Megatron` backend.

* `DTensor` uses the Hugging Face checkpoint both to initialize the training backend and to configure `vllm`, ensuring the model implementations match exactly. This is crucial for correctness.
* `Megatron` also uses the Hugging Face checkpoint to configure `vllm`, and performs a one-time conversion to a Megatron-format checkpoint to initialize the training backend.

If you would like to see direct support for Megatron checkpoints, please share your use case on
https://github.com/NVIDIA-NeMo/RL/issues/671.

## Backend Selection

The training backend is automatically determined based on your YAML configuration settings. Here's how to configure each backend.

### Megatron Backend
To enable Megatron-based training:

1. Initialize the NeMo and Megatron submodules by running `git submodule update --init --recursive`
2. Add the `megatron_cfg` key to your policy configuration.
3. Set `policy.megatron_cfg.enabled=True`.
4. Refer to [examples/configs/grpo_math_1B_megatron.yaml](../../examples/configs/grpo_math_1B_megatron.yaml) for a complete configuration example.

_Note_: When using Megatron, the optimizer and learning rate schedule are configured through `policy.megatron_cfg.optimizer` and `policy.megatron_cfg.scheduler`, respectively.

Model-provider options that do not have a dedicated NeMo RL config field can be
passed directly to the Megatron Bridge model config through
`policy.megatron_cfg.model_overrides`:

```yaml
policy:
  megatron_cfg:
    enabled: true
    model_overrides:
      masked_softmax_fusion: false
```

NeMo RL merges these values into a newly constructed model provider before
Megatron Bridge finalizes or instantiates it. That provider is also the model
config persisted in checkpoint `run_config.yaml` files. Unknown provider fields
fail during setup with their full config path.

`model_overrides` is only for model-provider fields without a first-class NeMo
RL setting. A key that duplicates another `megatron_cfg` field is rejected; set
the first-class field directly instead. Valid fields, values, and nesting are
defined by the provider in the installed Megatron Bridge version. Nested
dictionaries follow the hierarchy of nested model-config objects. NeMo
RL-specific settings such as optimizer, scheduler, checkpointing, and
environment variables remain under their existing `megatron_cfg` sections.

#### Fine-grained activation CPU offload

Fine-grained activation offloading asynchronously moves selected module-input
activations to CPU between forward and backward passes to reduce peak GPU
memory. It is distinct from `optimizer_cpu_offload`, which moves optimizer
states rather than activations. The following dense-model configuration is
runnable with the Megatron backend; `core_attn` and `attn_proj` are appropriate
for a dense Qwen model, and `attn_proj` must be paired with `core_attn`.

```yaml
policy:
  megatron_cfg:
    enabled: true
    cuda_graph_impl: transformer_engine
    env_vars:
      NVTE_CPU_OFFLOAD_V1: "1"
    fine_grained_activation_offloading: true
    offload_modules: ["core_attn", "attn_proj"]
```

Activation offloading requires the Transformer Engine model implementation.
CUDA graphs are optional; this example was validated with Transformer Engine
CUDA graphs. If graphs are enabled, pinned MCore permits `transformer_engine`
or `full_iteration` for this dense module pair, but only the former is validated
here. `local` CUDA graphs support only partial MoE offload (`expert_fc1`,
`moe_act`, and `fused_group_mlp`). With the default `cuda_graph_impl: none`, no
graph-specific restriction applies.

Supported module names are `attn_norm`, `qkv_linear`, `core_attn`,
`attn_proj`, `mlp_norm`, `expert_fc1`, `moe_act`, and `fused_group_mlp`.
The last three are MoE-specific. `fused_group_mlp` requires the Transformer
Engine op fuser and cannot be combined with `expert_fc1` or `moe_act`.

Activation checkpointing is not blanket-incompatible with fine-grained
activation offload. However, selective recomputation of the whole MoE module
(`recompute_modules: ["moe"]`) conflicts with MoE-internal offload modules
(`expert_fc1`, `moe_act`, or `fused_group_mlp`), and layer-level
`cpu_offloading` conflicts with fine-grained activation offload. Megatron
Bridge/Megatron-Core setup validation owns the exact compatibility checks for
the pinned versions.

Offloading saves GPU memory but adds CPU transfer and synchronization work, so
benchmark the memory and throughput tradeoff for the target model, sequence
length, hardware, and parallelism configuration.

### DTensor Backend
To enable DTensor (FSDP2) training:

1. Set `policy.dtensor_cfg.enabled=True`.
2. Refer to [examples/configs/grpo_math_1B.yaml](../../examples/configs/grpo_math_1B.yaml) for a configuration example.

For multi-node runs, set `policy.dtensor_cfg.dp_replicate_size > 1` on the DTensor v2 backend to enable Hybrid Sharded Data Parallel (HSDP). This keeps FSDP collectives intra-node and only replicates gradients across nodes. See [HSDP (`dp_replicate_size`)](fsdp2-parallel-plan.md#hsdp-dp_replicate_size).

## Configuration Examples

For comprehensive examples of each algorithm and backend, see the [examples/configs/recipes/llm](https://github.com/NVIDIA-NeMo/RL/tree/main/examples/configs/recipes/llm) folder. This directory contains ready-to-use configurations for various supported combinations.

## Megatron Configuration

The Megatron backend requires a checkpoint directory for storing converted Hugging Face model weights in Megatron format. This directory must be accessible from all nodes in your distributed training setup.

### Environment Variable Priority (Highest to Lowest) ###

1. **`NRL_MEGATRON_CHECKPOINT_DIR`** - The custom checkpoint directory path.
2. [RECOMMENDED] **`HF_HOME/nemo_rl`** - Uses the Hugging Face cache directory, if available.
3. **`~/.cache/huggingface/nemo_rl`** - The default fallback location.

### Configuration Examples ###

```bash
# Option 1: Set custom checkpoint directory
export NRL_MEGATRON_CHECKPOINT_DIR="/shared/nfs/checkpoints/megatron"

# Option 2: Use HuggingFace home directory (recommended for shared setups)
export HF_HOME="/shared/nfs/huggingface"
# This will use /shared/nfs/huggingface/nemo_rl

# Option 3: Use default (no environment variables needed)
# Uses ~/.cache/huggingface/nemo_rl
```

### Best Practices ###

- **Mount the checkpoint directory**: If you are using Docker, make sure the Megatron checkpoint path is covered by `-v`/`--mount`. Similarly, if you are using SLURM+pyxis, ensure `--container-mounts` includes this path.
- **Use shared storage**: Ensure the checkpoint directory is accessible from all nodes (e.g., NFS, shared filesystem).
- **Prefer HF_HOME**: If you already have `HF_HOME` mounted across nodes, this reduces the number of environment variables to manage.
- **Sufficient space**: Ensure adequate disk space for the converted model checkpoints.

### Force Reconvert ###

By default, NeMo RL skips the HF → Megatron conversion if a converted checkpoint already exists at the target path. If you need to force a fresh conversion (e.g., after updating megatron-bridge or changing `hf_config_overrides`), set the following option in your config:

```yaml
policy:
  megatron_cfg:
    force_reconvert_from_hf: True  # Default: False
```

This is equivalent to deleting the converted checkpoint directory and rerunning — the old checkpoint will be overwritten with a freshly converted one.
