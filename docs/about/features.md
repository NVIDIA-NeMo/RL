# Features and Roadmap

## Available Now

- **Distributed Training** - Ray-based infrastructure.
- **Environment Support and Isolation** - Support for multi-environment training and dependency isolation between components.
- **Worker Isolation** - Process isolation between RL Actors (no worries about global state).
- **Learning Algorithms** - GRPO/GSPO/DAPO, SFT (with LoRA), DPO, and on-policy distillation.
- **Multi-Turn RL** - Multi-turn generation and training for RL with tool use, games, etc.
- **Advanced Parallelism with DTensor** - PyTorch FSDP2, TP, CP, and SP for efficient training (through NeMo AutoModel).
- **Larger Model Support with Longer Sequences** - Performant parallelism with Megatron Core (TP/PP/CP/SP/EP/FSDP) through NeMo Megatron Bridge.
- **Sequence Packing** - Sequence packing in both DTensor and Megatron Core for large training performance gains.
- **Fast Generation** - vLLM backend for optimized inference.
- **Hugging Face Integration** - Out-of-box support in the DTensor path, with checkpoint conversion available for the Megatron path through Megatron Bridge middleware.
- **End-to-End FP8 Low-Precision Training** - Support for Megatron Core FP8 training and FP8 vLLM generation.
- **Vision Language Models (VLM)** - Support SFT and GRPO on VLMs.
- **Megatron Inference** - Megatron Inference for fast day-0 support for new Megatron models without weight conversion.
- **Async RL** - Support for asynchronous rollouts and replay buffers for off-policy training, and enable a fully asynchronous GRPO.
- **NeMo-Gym Integration** - RL environment integration.
- **GB200** - Container support for GB200.

## Planned / In Progress

- **Muon Optimizer** - Emerging optimizer support for SFT/RL.
- **SGLang Inference** - SGLang rollout support for optimized inference.
- **Improved Native Performance** - Improved training time for native PyTorch models.
- **Improved Large MoE Performance** - Improved Megatron Core training and generation performance.
- **New Models** - Qwen3-Next and Nemotron-Super.
- **Expanded Algorithms** - GDPO and broader LoRA coverage for GRPO and DPO.
- **Resiliency** - Fault tolerance and auto-scaling support.
- **Speculative Decoding** - Speculative decoding support for rollout acceleration.
