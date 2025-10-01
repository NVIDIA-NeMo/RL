---
description: "Learn about NeMo RL's core concepts, key features, and fundamental architecture for reinforcement learning with large language models"
categories: ["concepts-architecture"]
tags: ["overview", "concepts", "architecture", "features", "reinforcement-learning", "distributed", "large-language-models"]
personas: ["mle-focused", "researcher-focused", "admin-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

(about-overview)=
# About NeMo RL

NeMo RL is an open-source, comprehensive framework for reinforcement learning and supervised fine-tuning of large language models. Built for scalability and efficiency, NeMo RL enables researchers and practitioners to train and evaluate RL-enhanced language models at scale.

## What is NeMo RL

NeMo RL is a training-focused framework that combines the power of reinforcement learning with large language models. It provides a unified platform for training and fine-tuning language models using state-of-the-art RL algorithms.

The framework is designed to handle the complexities of distributed training across multiple GPUs and nodes, making it suitable for both research and production training environments. NeMo RL supports multiple training backends, including Hugging Face Transformers and Megatron-LM, providing flexibility for different model architectures and training requirements. NeMo RL is focused on training and evaluation - it does not include production deployment or serving infrastructure.

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quick Start
:link: ../get-started/quickstart
:link-type: doc

Get up and running with your first RL training job in minutes.

+++
{bdg-primary}`Beginner`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Installation
:link: ../get-started/installation
:link-type: doc

Complete setup instructions for all environments and platforms.

+++
{bdg-success}`Essential`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Core Architecture
:link: architecture-overview
:link-type: doc

Learn about NeMo RL's core architecture and design principles.

+++
{bdg-info}`Foundation`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Key Features
:link: key-features
:link-type: doc

Explore NeMo RL's key capabilities and technical highlights.

+++
{bdg-secondary}`Overview`
:::

::::

## Key Technical Capabilities

NeMo RL provides comprehensive support for modern RL workflows with large language models:

- **Model Scale**: Support for models ranging from 0.6B to 70B+ parameters
- **Advanced Parallelism**: FSDP2, Tensor Parallelism, Pipeline Parallelism, and Context Parallelism
- **State-of-the-Art Algorithms**: GRPO, DPO, and SFT with mathematical foundations and optimized implementations
- **Framework Integration**: Seamless integration with Hugging Face Transformers, Megatron-LM, and vLLM
- **Memory Optimization**: Gradient checkpointing, mixed precision, and efficient batching strategies

## Target Users

- **Researchers**: Explore state-of-the-art RL algorithms (GRPO, DPO, SFT) with large language models
- **Machine Learning Engineers**: Deploy scalable RL training pipelines with distributed computing
- **DevOps Engineers**: Manage multi-node training clusters and distributed training infrastructure
- **Data Scientists**: Fine-tune language models for specific domains and applications

## Why Use NeMo RL?

NeMo RL offers several key advantages for reinforcement learning with language models:

- **Scalability**: Built on Ray for distributed computing, enabling training across multiple nodes and GPUs
- **Flexibility**: Support for multiple model backends and RL algorithms
- **Training Focused**: Comprehensive tooling for distributed training and evaluation
- **Research Friendly**: Easy experimentation with different algorithms and environments
- **Performance**: Optimized for high-throughput training with advanced parallelization strategies
- **Reproducibility**: Comprehensive logging and experiment tracking for reliable research
- **Extensibility**: Easy implementation of custom algorithms and environments
- **Benchmarking**: Built-in evaluation frameworks and standardized metrics
- **Collaboration**: Standardized workflows for multi-institution research projects
- **Performance Highlights**: GRPO provides stable training across diverse model sizes and architectures with advanced optimization techniques.

## NeMo RL Framework

NeMo RL is a standalone framework designed specifically for reinforcement learning and post-training alignment of large language models. It provides comprehensive capabilities for distributed training, algorithm implementation, and model evaluation.

### Framework Design

NeMo RL is built as a complete, self-contained framework that provides all necessary components for RL training:

- **Distributed Training**: Ray-based orchestration for scalable training
- **Algorithm Implementation**: State-of-the-art RL algorithms (DPO, GRPO, SFT)
- **Model Support**: Integration with Hugging Face Transformers and Megatron-LM
- **Evaluation Tools**: Comprehensive evaluation and benchmarking capabilities

### Key Capabilities

NeMo RL provides a complete toolkit for RL training:

- **Model Training**: Distributed training across multiple GPUs and nodes
- **Algorithm Development**: Easy implementation of custom RL algorithms
- **Evaluation**: Comprehensive evaluation frameworks and standardized metrics
- **Reproducibility**: Robust logging and experiment tracking

## NeMo RL and Real-World Applications

NeMo RL enables reinforcement learning with large language models across diverse domains including code generation, mathematical reasoning, human preference alignment, and multi-turn agentic behavior. Each application includes architectural patterns, implementation details, and production considerations for building robust RL systems.

For comprehensive guides with step-by-step implementation, architectural patterns, and production deployment strategies, see the detailed [Use Cases](../learning-resources/use-cases/index) documentation.

## Architecture

NeMo RL's architecture is designed for distributed reinforcement learning with modular components, Ray-based coordination, and support for multiple training backends. The framework provides advanced scalability from single-GPU setups to multi-node clusters with thousands of GPUs.

For comprehensive details on system design, components, scalability, and technical specifications, see the [Core Architecture](architecture-overview) documentation.

## API and Development

NeMo RL provides a comprehensive Python API for implementing custom reinforcement learning algorithms, environments, and training workflows. The framework offers clean interfaces for algorithm development, environment creation, and distributed training orchestration.

For complete API documentation, code examples, and development guides, see the [API Documentation](../api-docs/index) section.

## Get Started

Ready to begin your NeMo RL journey? Start with the [Quick Start Guide](../get-started/quickstart) for your first training job, or explore the [Installation Guide](../get-started/installation) for complete setup instructions.



