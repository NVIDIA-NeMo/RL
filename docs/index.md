---
description: "Comprehensive documentation for NeMo RL - an open-source framework for reinforcement learning and supervised fine-tuning of large language models"
categories: ["getting-started"]
tags: ["reinforcement-learning", "language-models", "training", "distributed", "documentation", "overview"]
personas: ["mle-focused", "researcher-focused", "admin-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

# NeMo RL Documentation

Welcome to the NeMo RL documentation. This page provides a quick overview of the docs: About, Get Started, Learning Resources, Guides, Development, Core Design, and API Reference.

## Quick Navigation

:::: {grid} 1 2 2 2
:gutter: 1 1 1 2

::: {grid-item-card} {octicon}`gear;1.5em;sd-mr-1` About NeMo RL
:link: about/index
:link-type: doc

Explore NeMo RL's core concepts, architecture, key features, and capabilities for reinforcement learning with large language models.

+++
{bdg-secondary}`Overview` {bdg-secondary}`Concepts`
:::

::: {grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Get Started
:link: get-started/index
:link-type: doc

Set up your environment and run your first RL training job with distributed computing and advanced algorithms.

+++
{bdg-success}`Beginner` {bdg-secondary}`Setup`
:::

::: {grid-item-card} {octicon}`play;1.5em;sd-mr-1` Learning Resources
:link: learning-resources/index
:link-type: doc

Learn NeMo RL with step-by-step tutorials, working examples, and real-world use cases for training language models.

+++
{bdg-primary}`Learning` {bdg-secondary}`Hands-on`
:::

::: {grid-item-card} {octicon}`book;1.5em;sd-mr-1` Guides
:link: guides/index
:link-type: doc

Deep-dive guides for training algorithms, model development, environment setup, and production deployment.

+++
{bdg-primary}`Guides` {bdg-secondary}`How-to`
:::

::: {grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Development
:link: development/index
:link-type: doc

Essential tools for testing, debugging, profiling, and contributing to NeMo RL.

+++
{bdg-info}`Development` {bdg-secondary}`Tools`
:::

::: {grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Core Design
:link: core-design/index
:link-type: doc

Understand NeMo RL's architecture, design principles, computational systems, and development infrastructure.

+++
{bdg-info}`Architecture` {bdg-secondary}`Design`
:::

::: {grid-item-card} {octicon}`code;1.5em;sd-mr-1` API Reference
:link: apidocs/index
:link-type: doc

Technical reference for algorithms, models, data, environments, and distributed training components.

+++
{bdg-info}`API` {bdg-secondary}`Reference`
:::

::::

## Key Resources

:::: {grid} 1 2 2 2
:gutter: 1 1 1 2

::: {grid-item-card} {octicon}`play;1.5em;sd-mr-1` Training Algorithms
:link: guides/training-algorithms/index
:link-type: doc

Master SFT, DPO, and GRPO algorithms for training language models with reinforcement learning.

+++
{bdg-primary}`Algorithms` {bdg-secondary}`Training`
:::

::: {grid-item-card} {octicon}`server;1.5em;sd-mr-1` Distributed Training
:link: learning-resources/tutorials/distributed-training-scaling
:link-type: doc

Scale across multiple GPUs and nodes with Ray, FSDP2, and advanced parallelization strategies.

+++
{bdg-warning}`Distributed` {bdg-secondary}`Scalability`
:::

::: {grid-item-card} {octicon}`package;1.5em;sd-mr-1` Model Development
:link: guides/model-development/index
:link-type: doc

Integrate custom models, handle model quirks, and extend the framework for new architectures.

+++
{bdg-primary}`Models` {bdg-secondary}`Development`
:::

::: {grid-item-card} {octicon}`database;1.5em;sd-mr-1` Data Management
:link: core-design/data-management/index
:link-type: doc

Dataset preparation, processing pipelines, and efficient data handling for RL training.

+++
{bdg-info}`Data` {bdg-secondary}`Pipelines`
:::

::: {grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Performance Optimization
:link: guides/training-optimization/index
:link-type: doc

Profiling, optimization techniques, and performance tuning for high-throughput training.

+++
{bdg-success}`Performance` {bdg-secondary}`Optimization`
:::

::: {grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Troubleshooting
:link: guides/troubleshooting
:link-type: doc

Debugging, common problems, and advanced problem resolution for RL training workflows.

+++
{bdg-danger}`Debugging` {bdg-secondary}`Support`
:::

::::

## Where to Go Next

- **Quick Start**: [Run your first training job](get-started/quickstart)
- **Installation guide**: [Set up your environment](get-started/installation)
- **Tutorials**: [Browse step-by-step lessons](learning-resources/tutorials/index)

## Choose Your Path

- **Researchers**: Start with [About NeMo RL](about/index) and then explore [Training Algorithms](guides/training-algorithms/index)
- **ML Engineers**: Go to [Get Started](get-started/index), [Distributed Training](learning-resources/tutorials/distributed-training-scaling), and [Model Development](guides/model-development/index)
- **DevOps**: See [Core Design](core-design/index), [Performance Optimization](guides/training-optimization/index), and [Troubleshooting](guides/troubleshooting)

## Need Help

- **Troubleshooting**: [Fix common issues](guides/troubleshooting)
- **Community**: [GitHub Issues](https://github.com/NVIDIA/NeMo-RL/issues)

---

```{toctree}
:hidden:
:maxdepth: 1
Home <self>
```

```{toctree}
:hidden:
:caption: About
:maxdepth: 2
about/index
about/purpose
about/key-features
about/architecture-overview
```

```{toctree}
:hidden:
:caption: Get Started
:maxdepth: 2
get-started/index
get-started/installation
get-started/quickstart
get-started/model-selection
get-started/local-workstation
get-started/cluster
get-started/docker
```

```{toctree}
:hidden:
:caption: Guides
:maxdepth: 2
guides/index
guides/training-algorithms/index
guides/model-development/index
guides/environment-data/index
guides/training-optimization/index
guides/troubleshooting
```

```{toctree}
:hidden:
:caption: Development
:maxdepth: 2
development/index
development/debugging
development/testing
development/nsys-profiling
development/fp8
development/use-custom-vllm
```

```{toctree}
:hidden:
:caption: Core Design
:maxdepth: 2
core-design/index
core-design/design-principles/index
core-design/computational-systems/index
core-design/data-management/index
core-design/development-infrastructure/index
```

```{toctree}
:hidden:
:caption: Learning Resources
:maxdepth: 2
learning-resources/index
learning-resources/tutorials/index
learning-resources/examples/index
learning-resources/use-cases/index
```

```{toctree}
:hidden:
:caption: Reference
:maxdepth: 2
apidocs/index
```
