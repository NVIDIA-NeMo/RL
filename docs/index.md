---
description: "NeMo RL is an open-source post-training library for scaling reinforcement learning methods for multimodal models (LLMs, VLMs, etc.)"
categories:
  - documentation
  - home
tags:
  - reinforcement-learning
  - post-training
  - scalable
  - distributed
  - llm-training
personas:
  - Data Scientists
  - Machine Learning Engineers
  - Cluster Administrators
difficulty: beginner
content_type: index
---

(rl-home)=

# NeMo RL Documentation

**NeMo RL** is an open-source post-training library within the [NeMo Framework](https://github.com/NVIDIA-NeMo), designed to streamline and scale reinforcement learning methods for multimodal models (LLMs, VLMs, etc.). Designed for flexibility, reproducibility, and scale, NeMo RL enables both small-scale experiments and massive multi-GPU, multi-node deployments for fast experimentation in research and production environments.

## Introduction to NeMo RL

Learn about NeMo RL, how it works at a high-level, and the key features.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` About NeMo RL
:link: about/overview
:link-type: doc
Overview of NeMo RL and its capabilities.
+++
{bdg-secondary}`architecture` {bdg-secondary}`design-philosophy` {bdg-secondary}`scalable-rl`
:::

:::{grid-item-card} {octicon}`star;1.5em;sd-mr-1` Key Features
:link: about/features
:link-type: doc
Discover the main features of NeMo RL for post-training.
+++
{bdg-secondary}`algorithms` {bdg-secondary}`backends` {bdg-secondary}`distributed-training`
:::

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` Training Backends
:link: about/backends
:link-type: doc
Explore PyTorch DTensor and Megatron Core training backends and how to choose the right one.
+++
{bdg-secondary}`dtensor` {bdg-secondary}`megatron-core` {bdg-secondary}`parallelism`
:::

::::

## Quickstarts

Start here to install NeMo RL and run your first training job.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Installation & Quickstart
:link: get-started/index
:link-type: doc
**Start Here**: Install NeMo RL and run your first local training job in minutes.
+++
{bdg-primary}`installation` {bdg-secondary}`first-run`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` GRPO Quickstart
:link: get-started/grpo
:link-type: doc
Run Group Relative Policy Optimization (GRPO) training.
+++
{bdg-secondary}`on-policy` {bdg-secondary}`reinforcement-learning`
:::

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` SFT Quickstart
:link: get-started/sft
:link-type: doc
Run supervised fine-tuning (SFT) on instruction datasets.
+++
{bdg-secondary}`fine-tuning` {bdg-secondary}`instruction-following`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` DPO Quickstart
:link: get-started/dpo
:link-type: doc
Run Direct Preference Optimization (DPO) training.
+++
{bdg-secondary}`preference-learning` {bdg-secondary}`alignment`
:::

::::

## Training Algorithms

Explore how you can use NeMo RL with different training algorithms.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` GRPO
:link: guides/grpo
:link-type: doc
Group Relative Policy Optimization for efficient on-policy reinforcement learning.
+++
{bdg-secondary}`on-policy-rl` {bdg-secondary}`reward-optimization` {bdg-secondary}`multi-turn`
:::

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Supervised Fine-Tuning
:link: guides/sft
:link-type: doc
Fine-tune models on instruction-following datasets with supervised learning.
+++
{bdg-secondary}`instruction-tuning` {bdg-secondary}`supervised-learning`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` DPO
:link: guides/dpo
:link-type: doc
Direct Preference Optimization for preference-based training without reward models.
+++
{bdg-secondary}`preference-learning` {bdg-secondary}`alignment`
:::

:::{grid-item-card} {octicon}`trophy;1.5em;sd-mr-1` Reward Modeling
:link: guides/rm
:link-type: doc
Train reward models for preference learning and evaluation.
+++
{bdg-secondary}`reward-models` {bdg-secondary}`preference-learning`
:::

::::

## Tutorial Highlights

Check out tutorials to get a quick start on using NeMo RL.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` GRPO DeepscaleR
:link: guides/grpo-deepscaler
:link-type: doc
Reproduce DeepscaleR results with NeMo RL using GRPO on mathematical reasoning tasks.
+++
{bdg-secondary}`mathematical-reasoning` {bdg-secondary}`reproduction`
:::

:::{grid-item-card} {octicon}`number;1.5em;sd-mr-1` SFT on OpenMathInstruct2
:link: guides/sft-openmathinstruct2
:link-type: doc
Step-by-step guide for supervised fine-tuning on the OpenMathInstruct2 dataset.
+++
{bdg-secondary}`math-datasets` {bdg-secondary}`instruction-tuning`
:::

:::{grid-item-card} {octicon}`stack;1.5em;sd-mr-1` Custom Environments
:link: guides/environments
:link-type: doc
Create custom reward environments and integrate them with NeMo RL training pipelines.
+++
{bdg-secondary}`custom-rewards` {bdg-secondary}`environment-integration`
:::

:::{grid-item-card} {octicon}`plus-circle;1.5em;sd-mr-1` Adding New Models
:link: adding-new-models
:link-type: doc
Learn how to add support for new model architectures in NeMo RL.
+++
{bdg-secondary}`model-integration` {bdg-secondary}`custom-models`
:::

::::

---

::::{toctree}
:hidden:
Home <self>
::::

::::{toctree}
:hidden:
:caption: About NeMo RL
:maxdepth: 1
about/overview.md
about/features.md
about/backends.md
about/installation.md
about/algorithms/index.md
about/evaluation.md
about/tips-and-tricks.md
::::

::::{toctree}
:hidden:
:caption: Get Started
:maxdepth: 2

Quickstart <get-started/index.md>
SFT <get-started/sft.md>
DPO <get-started/dpo.md>
GRPO <get-started/grpo.md>
Cluster Setup <get-started/cluster.md>
::::

::::{toctree}
:hidden:
:caption: Training Algorithms
:maxdepth: 2

guides/grpo.md
guides/dapo.md
guides/sft.md
guides/dpo.md
guides/rm.md
guides/environments.md
guides/eval.md
::::

::::{toctree}
:hidden:
:caption: Examples & Tutorials
:maxdepth: 2

guides/grpo-deepscaler.md
guides/sft-openmathinstruct2.md
guides/grpo-sliding-puzzle.md
guides/deepseek.md
guides/async-grpo.md
adding-new-models.md
model-quirks.md
::::

::::{toctree}
:hidden:
:caption: Setup & Deployment
:maxdepth: 2

get-started/cluster.md
docker.md
::::

::::{toctree}
:hidden:
:caption: Advanced Topics
:maxdepth: 2

design-docs/design-and-philosophy.md
design-docs/training-backends.md
design-docs/generation.md
design-docs/checkpointing.md
design-docs/loss-functions.md
design-docs/sequence-packing-and-dynamic-batching.md
design-docs/fsdp2-parallel-plan.md
design-docs/padding.md
design-docs/logger.md
design-docs/chat-datasets.md
design-docs/uv.md
design-docs/env-vars.md
debugging.md
fp8.md
nsys-profiling.md
testing.md
documentation.md
guides/use-custom-vllm.md
::::

::::{toctree}
:hidden:
:caption: Reference
:maxdepth: 2

apidocs/index.rst
::::
