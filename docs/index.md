# NeMo RL Documentation

Welcome to the NeMo RL documentation. NeMo RL is an open-source post-training library developed by NVIDIA, designed to streamline and scale reinforcement learning methods for multimodal models (LLMs, VLMs, etc.).

This documentation provides comprehensive guides, examples, and references to help you get started with NeMo RL and build powerful post-training pipelines for your models.

## Getting Started

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`book` Overview
:link: about/overview
:link-type: doc

Learn about NeMo RL's architecture, design philosophy, and key features that make it ideal for scalable reinforcement learning.
:::

:::{grid-item-card} {octicon}`rocket` Quick Start
:link: about/quick-start
:link-type: doc

Get up and running quickly with examples for both DTensor and Megatron Core training backends.
:::

:::{grid-item-card} {octicon}`download` Installation
:link: about/installation
:link-type: doc

Step-by-step instructions for installing NeMo RL, including prerequisites, system dependencies, and environment setup.
:::

:::{grid-item-card} {octicon}`star` Features
:link: about/features
:link-type: doc

Explore the current features and upcoming enhancements in NeMo RL, including distributed training, advanced parallelism, and more.
:::

:::{grid-item-card} {octicon}`light-bulb` Tips and Tricks
:link: about/tips-and-tricks
:link-type: doc

Troubleshooting common issues including missing submodules, Ray dashboard access, and debugging techniques.
:::

::::

## Training and Generation

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`cpu` Training Backends
:link: about/backends
:link-type: doc

Learn about DTensor and Megatron Core training backends, their capabilities, and how to choose the right one for your use case.
:::

:::{grid-item-card} {octicon}`workflow` Algorithms
:link: about/algorithms/index
:link-type: doc

Discover supported algorithms including GRPO, SFT, DPO, RM, and on-policy distillation with detailed guides and examples.
:::

:::{grid-item-card} {octicon}`graph` Evaluation
:link: about/evaluation
:link-type: doc

Learn how to evaluate your models using built-in evaluation datasets and custom evaluation pipelines.
:::

:::{grid-item-card} {octicon}`server` Cluster Setup
:link: about/clusters
:link-type: doc

Configure and deploy NeMo RL on multi-node Slurm or Kubernetes clusters for distributed computing.
:::

::::

## Guides and Examples

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`mortar-board` GRPO DeepscaleR
:link: guides/grpo-deepscaler
:link-type: doc

Reproduce DeepscaleR results with NeMo RL using GRPO on mathematical reasoning tasks.
:::

:::{grid-item-card} {octicon}`number` SFT on OpenMathInstruct2
:link: guides/sft-openmathinstruct2
:link-type: doc

Step-by-step guide for supervised fine-tuning on the OpenMathInstruct2 dataset.
:::

:::{grid-item-card} {octicon}`stack` Environments
:link: guides/environments
:link-type: doc

Create custom reward environments and integrate them with NeMo RL training pipelines.
:::

:::{grid-item-card} {octicon}`plus-circle` Adding New Models
:link: adding-new-models
:link-type: doc

Learn how to add support for new model architectures in NeMo RL.
:::

::::

## Advanced Topics

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`telescope` Design and Philosophy
:link: design-docs/design-and-philosophy
:link-type: doc

Deep dive into NeMo RL's architecture, APIs, and design decisions for scalable RL.
:::

:::{grid-item-card} {octicon}`bug` Debugging
:link: debugging
:link-type: doc

Tools and techniques for debugging distributed Ray applications and RL training runs.
:::

:::{grid-item-card} {octicon}`zap` FP8 Quantization
:link: fp8
:link-type: doc

Optimize large language models with FP8 quantization for faster training and inference.
:::

:::{grid-item-card} {octicon}`container` Docker Containers
:link: docker
:link-type: doc

Build and use Docker containers for reproducible NeMo RL environments.
:::

::::

## API Reference

::::{grid} 1 1 1 1
:gutter: 3

:::{grid-item-card} {octicon}`code` Complete API Documentation
:link: apidocs/index
:link-type: doc

Comprehensive reference for all NeMo RL modules, classes, functions, and methods. Browse the complete Python API with detailed docstrings and usage examples.
:::

::::

```{toctree}
:caption: About
:hidden:

about/overview
about/performance-summary
about/model-support
about/features
about/backends
about/quick-start
about/installation
about/algorithms/index
about/evaluation
about/clusters
about/tips-and-tricks
```



```{toctree}
:caption: Environment Start
:hidden:

local-workstation.md
cluster.md

```

```{toctree}
:caption: E2E Examples
:hidden:

guides/sft-openmathinstruct2.md
```

```{toctree}
:caption: Guides
:hidden:

guides/nemotron-3-nano.md
adding-new-models.md
guides/sft.md
guides/dpo.md
guides/dapo.md
guides/grpo.md
guides/grpo-deepscaler.md
guides/grpo-sliding-puzzle.md
guides/rm.md
guides/environments.md
guides/eval.md
guides/deepseek.md
model-quirks.md
guides/async-grpo.md
guides/dtensor-tp-accuracy.md
```

```{toctree}
:caption: Containers
:hidden:

docker.md
```

```{toctree}
:caption: Development
:hidden:

testing.md
documentation.md
debugging.md
nsys-profiling.md
fp8.md
guides/use-custom-vllm.md
```

```{toctree}
:caption: Design Docs
:hidden:

design-docs/design-and-philosophy.md
design-docs/padding.md
design-docs/logger.md
design-docs/uv.md
design-docs/dependency-management.md
design-docs/chat-datasets.md
design-docs/generation.md
design-docs/checkpointing.md
design-docs/loss-functions.md
design-docs/fsdp2-parallel-plan.md
design-docs/training-backends.md
design-docs/sequence-packing-and-dynamic-batching.md
design-docs/env-vars.md
```

```{toctree}
:caption: API Reference
:hidden:

apidocs/index
```
