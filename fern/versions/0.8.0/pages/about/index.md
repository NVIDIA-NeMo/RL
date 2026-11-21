---
title: "NeMo RL Documentation"
---
Welcome to the NeMo RL documentation. NeMo RL is an open-source post-training library developed by NVIDIA, designed to streamline and scale reinforcement learning methods for multimodal models (LLMs, VLMs, etc.).

This documentation provides comprehensive guides, examples, and references to help you get started with NeMo RL and build powerful post-training pipelines for your models.

## Getting Started

<Cards>

<Card title="Overview" href="/about/overview">

Learn about NeMo RL's architecture, design philosophy, and key features that make it ideal for scalable reinforcement learning.
</Card>

<Card title="Quick Start" href="/about/quick-start">

Get up and running quickly with examples for both Automodel and Megatron Core training backends.
</Card>

<Card title="Installation" href="/about/installation">

Step-by-step instructions for installing NeMo RL, including prerequisites, system dependencies, and environment setup.
</Card>

<Card title="Features" href="/about/features">

Explore the current features and upcoming enhancements in NeMo RL, including distributed training, advanced parallelism, and more.
</Card>

<Card title="Tips and Tricks" href="/about/tips-and-tricks">

Troubleshooting common issues including missing submodules and memory fragmentation.
</Card>

</Cards>

## Training and Generation

<Cards>

<Card title="Training Backends" href="/about/backends">

Learn about Automodel and Megatron Core training backends, their capabilities, and how to choose the right one for your use case.
</Card>

<Card title="Algorithms" href="/about/algorithms">

Discover supported algorithms including GRPO, PPO, SFT, DPO, RM, on-policy distillation, and multi-teacher on-policy distillation (MOPD) with detailed guides and examples.
</Card>

<Card title="Evaluation" href="/about/evaluation">

Measure model and agent capabilities with repeatable benchmarks and custom evaluations.
</Card>

<Card title="Cluster Setup" href="/environment-start/cluster">

Configure and launch NeMo RL on multi-node Slurm or Kubernetes clusters for distributed computing.
</Card>

<Card title="Managed Dynamo Generation" href="/guides/dynamo-generation">

Run a fixed Dynamo vLLM fleet with NCCL refit and W&B telemetry inside a Slurm Ray allocation.
</Card>

</Cards>

## Guides and Examples

<Cards>

<Card title="GRPO DeepscaleR" href="/guides/grpo-deepscaler">

Reproduce DeepscaleR results with NeMo RL using GRPO on mathematical reasoning tasks.
</Card>

<Card title="SFT on OpenMathInstruct2" href="/e2e-examples/sft-openmathinstruct2">

Step-by-step guide for supervised fine-tuning on the OpenMathInstruct2 dataset.
</Card>

<Card title="Nemotron 3 Ultra" href="/guides/model-guides/nemotron/nemotron-3-ultra">

Post-train Nemotron 3 Ultra with RLVR, teacher training, and MOPD stages on GB200 NVL72 hardware.
</Card>

<Card title="Environments" href="/guides/environments">

Create custom reward environments and integrate them with NeMo RL training pipelines.
</Card>

<Card title="Eagle3 Speculative Decoding" href="/guides/eagle3-speculative-decoding">

Configure offline and online Eagle3 draft-model workflows to accelerate rollout generation with vLLM.
</Card>

<Card title="Audio GRPO on AVQA" href="/guides/grpo-audio">

Train Qwen2.5-Omni-3B with GRPO on AVQA and evaluate on MMAU, following the R1-AQA approach.
</Card>

<Card title="Audio-Visual Intent GRPO" href="/guides/grpo-audio-visual">

Train Qwen2.5-Omni-7B with GRPO on PhilipC/IntentTrain (audio-visual intent recognition) and evaluate on Daily-Omni, following HumanOmniV2's joint audio-visual setup.
</Card>

<Card title="Two-Stage SWE RL (Qwen3 Thinking)" href="/guides/swe-rl-qwen3">

Train Qwen3-30B-A3B-Thinking into a SWE agent with a pivot stage plus end-to-end agentic RL on SWE-bench.
</Card>

<Card title="Adding New Models" href="/guides/adding-new-models">

Learn how to add support for new model architectures in NeMo RL.
</Card>

<Card title="LoRA" href="/guides/lora">

Parameter-efficient fine-tuning with LoRA: backend support, Automodel vs Megatron schema comparison, config examples, and recipes.
</Card>

<Card title="YaRN Long-Context Training" href="/guides/yarn-long-context">

Extend a model's context window with YaRN RoPE scaling on the Megatron backend for SFT, GRPO, and other workflows.
</Card>

<Card title="Cross-Tokenizer Distillation" href="/guides/xtoken-off-policy-distillation">

Off-policy distillation across mismatched tokenizers — build a (student, teacher) projection matrix and run x-token KD via CUDA-IPC teacher logits.
</Card>

<Card title="Weight Refit" href="/guides/refit">

Choose among colocated IPC, NCCL, sparse delta, and NIXL refit transports.
</Card>

<Card title="Checkpoint-Engine Refit" href="/guides/checkpoint-engine-refit">

Use NIXL checkpoint-engine refit to update non-colocated vLLM generation workers from policy workers.
</Card>

<Card title="Single-Controller (Async GRPO and PPO)" href="/guides/single-controller">

Run async GRPO or PPO via the SingleController path: TransferQueue data plane, pluggable staleness samplers, and streaming trainer.
</Card>

</Cards>

## Advanced Topics

<Cards>

<Card title="Design and Philosophy" href="/design-docs/design-and-philosophy">

Deep dive into NeMo RL's architecture, APIs, and design decisions for scalable RL.
</Card>

<Card title="Debugging" href="/development/debugging">

Tools and techniques for debugging distributed Ray applications and RL training runs.
</Card>

<Card title="Observability" href="/observability">

OpenTelemetry traces and `rl.*` metrics via nemo-lens: span groups, configuration, vLLM tracing, and an OTLP export stack.
</Card>

<Card title="FP8 Quantization" href="/development/fp8">

Optimize large language models with FP8 quantization for faster training and inference.
</Card>

<Card title="Quantization-Aware RL" href="/guides/quantization-aware-rl">

Run quantized GRPO and distillation with NVIDIA ModelOpt.
</Card>

<Card title="End-to-End NVFP4 Training" href="/design-docs/te-nvfp4-per-token-rollout">

Run W4A4 policy training and per-token vLLM rollout on NVIDIA Blackwell GPUs.
</Card>

<Card title="Docker Containers" href="/containers/docker">

Build and use Docker containers for reproducible NeMo RL environments.
</Card>

</Cards>

## API Reference

<Cards>

<Card title="Complete API Documentation" href="/api-reference">

Comprehensive reference for all NeMo RL modules, classes, functions, and methods. Browse the complete Python API with detailed docstrings and usage examples.
</Card>

</Cards>
