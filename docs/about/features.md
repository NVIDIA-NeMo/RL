# Features

_Available now_ | _Coming in v0.4_

## Coming in v0.4

- **Megatron Inference** - Megatron Inference for fast Day-0 support for new Megatron models (avoid weight conversion)
- **Async RL** - Support for asynchronous rollouts and replay buffers for off-policy training, and enable a fully asynchronous GPRO
- **Vision Language Models (VLM)** - Support SFT and GRPO on VLMs through the DTensor path
- **Improved Native Performance** - Improve training time for native PyTorch models
- **Improved Large MoE Performance** - Improve Megatron Core training performance and generation performance
- **End-to-End FP8 Low-Precision Training** - Support for Megatron Core FP8 training and FP8 vLLM generation
- **Megatron Bridge Integration** - Integrate Megatron Bridge to enable training features from Megatron Core
- **NeMo Automodel Integration** - Integrate NeMo Automodel to power the DTensor path
- **New Models** - `gpt-oss`
- **Expand Algorithms** - DAPO, GSPO, On-policy Distillation
- **GB200** - Add container support for GB200

## Available Now

- **Distributed Training** - Ray-based infrastructure
- **Environment Support and Isolation** - Support for multi-environment training and dependency isolation between components
- **Worker Isolation** - Process isolation between RL Actors (no worries about global state)
- **Learning Algorithms** - GRPO/GSPO, SFT, and DPO
- **Multi-Turn RL** - Multi-turn generation and training for RL with tool use, games, etc
- **Advanced Parallelism with DTensor** - PyTorch FSDP2, TP, CP, and SP for efficient training
- **Larger Model Support with Longer Sequences** - Performant parallelisms with Megatron Core (TP/PP/CP/SP/EP/FSDP)
- **MoE Models** - Support for DeepSeekV3 and Qwen-3 MoE models (Megatron)
- **Sequence Packing** - Sequence packing in both DTensor and Megatron Core for huge training performance gains
- **Fast Generation** - vLLM backend for optimized inference
- **Hugging Face Integration** - Works with 1B–70B models (Qwen, Llama)

