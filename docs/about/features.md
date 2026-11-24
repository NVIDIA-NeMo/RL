# Features and Roadmap

## Available Now

- **Distributed Training** - Ray-based infrastructure.
- **Environment Support and Isolation** - Support for multi-environment training and dependency isolation between components.
- **Worker Isolation** - Process isolation between RL Actors (no worries about global state).
- **Learning Algorithms** - GRPO/GSPO/DAPO, SFT (with LoRA), DPO, and on-policy distillation.
- **Multi-Turn RL** - Multi-turn generation and training for RL with tool use, games, etc.
- **Advanced Parallelism with Automodel** - PyTorch FSDP2, TP, CP, SP, and EP for efficient training (through NeMo AutoModel).
- **Larger Model Support with Longer Sequences** - Performant parallelism with Megatron Core (TP/PP/CP/SP/EP/FSDP) through NeMo Megatron Bridge.
- **Sequence Packing** - Sequence packing in both Automodel and Megatron Core for large training performance gains.
- **Fast Generation** - vLLM backend for optimized inference.
- **Hugging Face Integration** - Out-of-box support in the Automodel path, with checkpoint conversion available for the Megatron path through Megatron Bridge middleware.
- **End-to-End FP8 Low-Precision Training** - Support for Megatron Core FP8 training and FP8 vLLM generation.
- **Vision Language Models (VLM)** - Support SFT and GRPO on VLMs.
- **Megatron Inference** - Megatron Inference for fast day-0 support for new Megatron models without weight conversion.
- **Async RL** - Support for asynchronous rollouts and replay buffers for off-policy training, and enable a fully asynchronous GRPO.
- **NeMo-Gym Integration** - RL environment integration.
- **GB200** - Container support for GB200.
- **Muon Optimizer** - Muon optimizer support for SFT/RL. See the [Muon guide](../guides/muon-optimizer.md).
- **SGLang Inference** - SGLang rollout backend. See the [Generation design doc](../design-docs/generation.md).
- **Speculative Decoding** - Eagle3 draft models for rollout acceleration. See the [Eagle3 guide](../guides/eagle3-speculative-decoding.md).
- **GDPO** - See [Multi-reward support (GDPO)](../guides/environments.md#multi-reward-support-gdpo) and `examples/configs/gdpo_math_1B.yaml`.
- **Nemotron-Super** - See the [Nemotron 3 Super guide](../guides/models/nemotron/nemotron-3-super.md).

## Planned / In Progress

- **Improved Native Performance** - Improved training time for native PyTorch models.
- **Improved Large MoE Performance** - Improved Megatron Core training and generation performance.
- **New Models** - Qwen3-Next.
- **Expanded Algorithms** - Broader LoRA coverage for GRPO and DPO.
- **Resiliency** - Fault tolerance and auto-scaling support.
