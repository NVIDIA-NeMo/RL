# Training Backends

NeMo RL supports multiple training backends to accommodate different model sizes and hardware configurations.

## Available Backends

- **DTensor (FSDP2)** - PyTorch's next-generation distributed training with improved memory efficiency
- **Megatron** - NVIDIA's high-performance training framework for scaling to large models (>100B parameters)

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
