# Features and Roadmap

_Available now_ | _Coming in v0.6_

## Coming in v0.6

- **Muon Optimizer** - Emerging Optimizer support for SFT/RL
- **Megatron Inference** - Improved performance for Megatron Inference (avoid weight conversion).
- **SGLang Inference** - SGLang rollout support for optimized inference.
- **Improved Native Performance** - Improve training time for native PyTorch models.
- **Improved Large MoE Performance** - Improve Megatron Core training performance and generation performance.
- **New Models** - Qwen3-Next, Nemotron-Super.
- **Expand Algorithms** - GDPO, LoRA support for RL(GRPO) and DPO
- **Resiliency** - Fault tolerance and auto-scaling support
- **On-Policy Distillation** - Multi-teacher and cross tokenizer distillation support
- **Speculative Decoding** - Speculative Decoding support for rollout acceleration

## Available Now

- **Distributed Training** - Ray-based infrastructure.
- **Environment Support and Isolation** - Support for multi-environment training and dependency isolation between components.
- **Worker Isolation** - Process isolation between RL Actors (no worries about global state).
- **Learning Algorithms** - GRPO/GSPO/DAPO, SFT(with LoRA), DPO, and On-policy distillation.
- **Multi-Turn RL** - Multi-turn generation and training for RL with tool use, games, etc.
- **Advanced Parallelism with DTensor** - PyTorch FSDP2, TP, CP, and SP for efficient training (through NeMo AutoModel).
- **Larger Model Support with Longer Sequences** - Performant parallelisms with Megatron Core (TP/PP/CP/SP/EP/FSDP) (through NeMo Megatron Bridge).
- **Sequence Packing** - Sequence packing in both DTensor and Megatron Core for huge training performance gains.
- **Fast Generation** - vLLM backend for optimized inference.
- **Hugging Face Integration** - OOB support in the DTensor path, CKPT conversion available for Megatron path through Megatron Bridge middleware.
- **End-to-End FP8 Low-Precision Training** - Support for Megatron Core FP8 training and FP8 vLLM generation.
- **Vision Language Models (VLM)** - Support SFT and GRPO on VLMs.
- **Megatron Inference** - Megatron Inference for fast Day-0 support for new Megatron models (avoid weight conversion).
- **Async RL** - Support for asynchronous rollouts and replay buffers for off-policy training, and enable a fully asynchronous GRPO.
- **NeMo-Gym Integration** - RL Environment Integration.
- **GB200** - Container support for GB200.
