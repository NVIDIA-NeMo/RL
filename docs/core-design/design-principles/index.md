---
description: "Core architectural decisions and design principles that shape NeMo RL."
tags: ["architecture", "design", "core", "foundation"]
categories: ["design-principles"]
---

# Design Principles

This section contains the fundamental architectural decisions and design principles that shape NeMo RL. These documents provide insights into the core abstractions, training backends, and parallelization strategies that enable scalable reinforcement learning.

## What You'll Find Here

- **Design Philosophy**: Understand the core design principles and philosophical approach behind NeMo RL's architecture
- **FSDP2 Parallel Plan**: Learn about the FSDP2 parallelization strategy and implementation details  
- **Generation**: Explore the text generation pipeline and inference mechanisms
- **Loss Functions**: Understand loss function design for gradient accumulation equivalence

## Design Principles

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Design Philosophy
:link: design-and-philosophy
:link-type: doc

Understand the core design principles and philosophical approach behind NeMo RL's architecture.

+++
{bdg-primary}`Foundation`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` FSDP2 Parallel Plan
:link: fsdp2-parallel-plan
:link-type: doc

Learn about the FSDP2 parallelization strategy and implementation details.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Generation
:link: generation
:link-type: doc

Explore the text generation pipeline and inference mechanisms.

+++
{bdg-info}`Computation`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Loss Functions
:link: loss-functions
:link-type: doc

Understand loss function design for gradient accumulation equivalence.

+++
{bdg-warning}`Advanced`
:::

::::

---

::::{toctree}
:hidden:
:caption: Design Principles
:maxdepth: 2
design-and-philosophy
fsdp2-parallel-plan
generation
loss-functions
:::: 