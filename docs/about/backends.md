# Training and Generation Backends

## Training Backends

NeMo RL supports multiple training backends to accommodate different model sizes and hardware configurations:

- **DTensor** - PyTorch's next-generation distributed training with improved memory efficiency (PyTorch-native TP, SP, PP, CP, and FSDP2)
- [**Megatron**](https://github.com/NVIDIA-NeMo/Megatron-Bridge) - NVIDIA's high-performance training framework for scaling to large models with 6D parallelisms

The training backend is automatically determined based on your YAML configuration settings. For detailed information on backend selection, configuration, and examples, see the [Training Backends documentation](../design-docs/training-backends.md).

## Generation Backends

NeMo RL supports multiple generation/rollout backends to accommodate different model sizes and hardware configurations:

- [**vLLM**](https://github.com/vllm-project/vllm) - A high-throughput and memory-efficient popular inference and serving engine
- [**Megatron**](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/inference) - A high-performance Megatron-native inference backend which eliminates weight conversion between training and inference

For detailed information on backend selection, configuration, and examples, see the [Generation Backends documentation](../design-docs/generation.md).

