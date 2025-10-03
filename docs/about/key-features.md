---
description: "Key features and capabilities of NeMo RL including scalable training, supported algorithms, and developer experience"
categories: ["concepts-architecture"]
tags: ["features", "scalability", "algorithms", "distributed", "developer-experience", "reinforcement-learning"]
personas: ["mle-focused", "researcher-focused", "admin-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

(about-key-features)=
# Key Features

NeMo RL provides a robust, modular toolkit for scalable reinforcement learning and supervised fine-tuning of large language models. This summary highlights the core features most relevant to AI developers.

## 1. Scalable and Efficient Training

- **Single-GPU to Multi-Node**: Seamless scaling from 1 GPU to 64+ GPUs and multi-node clusters.
- **Model Size Flexibility**: Supports models from small (0.6B) to 70B+ parameters.
- **Advanced Parallelism**: FSDP2, Tensor Parallelism, Pipeline Parallelism, Context Parallelism.
- **Distributed with Ray**: Ray-based orchestration for distributed rollouts and training.
- **Memory Optimization**: Gradient checkpointing, mixed precision, and efficient batching.

## 2. Supported Algorithms

| Algorithm | Purpose                    | Highlights                                |
|----------|-----------------------------|-------------------------------------------|
| **GRPO** | Reasoning-focused RL        | No critic needed, stable updates, memory-efficient |
| **DPO**  | Human preference alignment  | Uses pairwise comparisons, no reward model |
| **SFT**  | Initial supervised alignment| Prepares models for RL fine-tuning         |

NeMo RL also supports **multi-turn RL**, enabling training for agentic tasks like games or tool use.

**Benchmark Performance**: GRPO provides stable and efficient training for large language models with advanced optimization techniques.

## 3. Backend and Model Support

| Backend         | Best Use Case              | Key Features                                   |
|----------------|----------------------------|------------------------------------------------|
| **PyTorch Native** | Broad model support         | FSDP2, TP, SP, activation checkpointing        |
| **Megatron Core**  | Extremely large models      | Supports long context, MoE, deep parallelism   |
| **vLLM**           | High-speed generation       | Optimized inference, tensor parallelism        |

Backends are configurable and interchangeable without altering core algorithm logic.

- **Model Integration**: Hugging Face, Megatron, vLLM, and custom architectures.
- **Conversion Tools**: Utilities for converting between Hugging Face and Torch-native formats.

## 4. Integration Ecosystem

- **Hugging Face**: For model loading, datasets, tokenizer management
- **Ray**: Distributed rollout generation, process isolation, fault tolerance
- **uv**: Python environment management for reproducibility
- **Weights and Biases**: Logging and training visualization support

## 5. Data and Environment Integration

- **Dataset Support**: Direct integration with Hugging Face datasets and custom data loaders.
- **Environment Abstractions**: Standardized RL environment interfaces for custom or built-in tasks.
- **Multi-Environment**: Support for complex, multi-environment RL scenarios.

## 6. Configuration and Extensibility

- **YAML Configs**: Human-readable configuration files for all training and rollout parameters.
- **CLI Overrides**: Command-line parameter overrides for rapid experimentation.
- **Unified APIs**: Standardized interfaces for training, rollout, and generation backends.
- **Modular Components**: Plug-and-play actors, environments, reward models, and policies.

## 7. Developer Experience

- **Comprehensive Documentation**: Detailed guides, API docs, and examples.
- **Type Hints and Testing**: Full type annotation and extensive test coverage.
- **Debugging and Profiling**: Integrated tools for debugging distributed training and profiling performance.
- **Logging and Monitoring**: Weights and Biases integration, advanced logging, and experiment tracking.

## 8. Scalability and Performance
- **Rollout Generation**: Distributed across Ray actors using either Hugging Face or vLLM
- **Training Execution**: Supports various forms of parallelism including FSDP2 and DeepSpeed-style optimization
- **Multi-node Compatibility**: Works seamlessly with Slurm, Kubernetes, and bare-metal clusters
- **Checkpoint Flexibility**: Converts between Hugging Face and Torch-native formats

## 9. Deployment and Reproducibility
- **Local and Cluster Deployment**: Easy setup for local development, Slurm, Kubernetes, and cloud.
- **Container Support**: Docker-based workflows for reproducibility.
- **Python Environment Management**: uv and conda integration for isolated, reproducible environments.
- **Checkpointing**: Robust checkpointing and recovery for long-running jobs.

---

For more details, see the [API Reference](../apidocs/index), [Guides](../guides/index), and [Core Design](../core-design/index).

