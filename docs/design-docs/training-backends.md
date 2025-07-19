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

1. Add the `megatron_cfg` key to your policy configuration.
2. Set `policy.megatron_cfg.enabled=True`.
3. Refer to [examples/configs/grpo_math_1B_megatron.yaml](../../examples/configs/grpo_math_1B_megatron.yaml) for a complete configuration example.

_Note_: When using Megatron, the optimizer and learning rate schedule are configured through `policy.megatron_cfg.optimizer` and `policy.megatron_cfg.scheduler`, respectively.

### DTensor Backend
To enable DTensor (FSDP2) training:

1. Set `policy.dtensor_config.enabled=True`.
2. Refer to [examples/configs/grpo_math_1B.yaml](../../examples/configs/grpo_math_1B.yaml) for a configuration example.

## Backend Priority

**Megatron takes precedence over DTensor.** If both backends are enabled simultaneously (`policy.megatron_cfg.enabled=True` and `policy.dtensor_config.enabled=True`), the Megatron backend will be used.

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

- **Mount in checkpoint directory**: If you are using Docker, make sure the Megatron checkpoint path is covered by `-v`/`--mount`. Similarly, if you are using SLURM+pyxis, ensure `--container-mounts` includes this path.
- **Use shared storage**: Ensure the checkpoint directory is accessible from all nodes (e.g., NFS, shared filesystem).
- **Prefer HF_HOME**: If you already have `HF_HOME` mounted across nodes, this reduces the number of environment variables to manage.
- **Sufficient space**: Ensure adequate disk space for the converted model checkpoints.