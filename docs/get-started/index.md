---
description: "Get started quickly with NeMo RL by following these essential setup steps and choosing the right training approach for your reinforcement learning needs"
categories: ["getting-started"]
tags: ["setup", "beginner", "onboarding", "reinforcement-learning", "distributed", "training-algorithms"]
personas: ["mle-focused", "researcher-focused", "admin-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

(gs-overview)=
# About Getting Started with NeMo RL

Welcome to NeMo RL! This guide will help you set up your environment and run your first reinforcement learning training job with large language models.

## Before You Start

- **System Requirements**: Ensure you have CUDA-compatible GPUs and sufficient memory
- **Python Environment**: Python 3.9+ with PyTorch 2.0+ and Ray
- **Model Access**: Access to Hugging Face models (Qwen, Llama, etc.)
- **Cluster Setup**: For distributed training, prepare your Ray cluster

---

## Essential Setup

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Installation
:link: installation
:link-type: doc

Complete setup instructions for all environments and platforms.

+++
{bdg-success}`Essential`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quickstart
:link: quickstart
:link-type: doc

Run your first RL training job in minutes with our step-by-step guide.

+++
{bdg-primary}`Beginner`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Choose Your Model
:link: model-selection
:link-type: doc

Choose the right model size and type for your use case and hardware.

+++
{bdg-info}`Planning`
:::

:::{grid-item-card} {octicon}`device-desktop;1.5em;sd-mr-1` Local Workstation
:link: local-workstation
:link-type: doc

Configure your development environment for optimal performance.

+++
{bdg-secondary}`Setup`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Docker Setup
:link: docker
:link-type: doc

Use NeMo RL with Docker containers for consistent environments.

+++
{bdg-secondary}`Container`
:::

::::

## Choose Your Training Approach

Select the training method that best fits your needs and experience level:

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` SFT Training
:link: ../guides/training-algorithms/sft
:link-alt: Supervised Fine-Tuning guide

Get started with supervised fine-tuning for language models. Perfect for beginners and domain adaptation tasks.

+++
{bdg-primary}`Beginner`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` GRPO Training
:link: ../guides/training-algorithms/grpo
:link-alt: Group Relative Policy Optimization guide

Learn Group Relative Policy Optimization for reinforcement learning with language models. Advanced RL training.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`heart;1.5em;sd-mr-1` DPO Training
:link: ../guides/training-algorithms/dpo
:link-alt: Direct Preference Optimization guide

Explore Direct Preference Optimization for preference learning and alignment. Preference-based training.

+++
{bdg-info}`Intermediate`
:::

::::

## Distributed Training

For details on scaling NeMo RL across multiple GPUs and nodes, see the distributed training documentation.

## Next Steps

After completing your first training run:

1. **Explore Algorithms**: Try [GRPO](../guides/training-algorithms/grpo) and [DPO](../guides/training-algorithms/dpo) training
2. **Scale Up**: Set up [distributed training](cluster.md) for larger models
3. **Optimize Performance**: Use [profiling tools](../development/nsys-profiling) for training optimization
4. **Customize**: Learn to [add new models](../guides/model-development/add-new-models) and environments
5. **Deploy**: Follow the [troubleshooting guide](../guides/troubleshooting) for production deployment

## Get Help

- [Troubleshooting](../guides/troubleshooting) - Common issues and solutions
- [API Reference](../apidocs/index) - Complete API documentation
- [Community Support](https://github.com/NVIDIA/NeMo-RL/issues) - GitHub issues and discussions

