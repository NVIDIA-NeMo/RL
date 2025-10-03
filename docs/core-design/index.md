---
description: "Explore NeMo RL's design documents, architectural decisions, and technical specifications for understanding the framework's internals"
categories: ["concepts-architecture"]
tags: ["design", "architecture", "technical", "specifications", "internals", "design-principles"]
personas: ["mle-focused", "researcher-focused"]
difficulty: "advanced"
content_type: "concept"
modality: "universal"
---

(design-overview)=
# About NeMo RL Core Design

This section contains detailed design documents, architectural decisions, and technical specifications for NeMo RL. These documents provide insights into the framework's internals and design philosophy for advanced users who want to understand, extend, or contribute to the framework.

## What You'll Find Here

Our design documents are organized into four main categories to help you understand the framework's architecture and implementation. These documents are essential for:

- **Understanding the Framework**: Deep dive into how NeMo RL works internally
- **Extending Functionality**: Learn how to add custom algorithms, backends, or environments  
- **Contributing Code**: Understand the codebase structure and design patterns
- **Debugging Issues**: Trace problems through the system architecture
- **Performance Optimization**: Learn about bottlenecks and optimization strategies

### **Design Principles** 
Explore the fundamental architectural decisions and design principles that shape NeMo RL. Understand the core abstractions, training backends, and parallelization strategies that enable scalable reinforcement learning.

### **Data Management**
Learn about data handling, processing, and persistence mechanisms. Discover how NeMo RL manages different data formats, implements efficient padding strategies, and maintains model state through checkpointing.

### **Computational Systems**
Dive into the mathematical foundations and computational components that power NeMo RL. Understand loss function implementations, text generation systems, and observability features.

### **Development Infrastructure**
Master the development tools and infrastructure that support NeMo RL development. Learn about package management, logging systems, and other development utilities.



## Design Principles

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Design Philosophy
:link: design-principles/design-and-philosophy
:link-type: doc

Understand the core design principles and philosophical approach behind NeMo RL's architecture.

+++
{bdg-primary}`Foundation`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` FSDP2 Parallel Plan
:link: design-principles/fsdp2-parallel-plan
:link-type: doc

Learn about the FSDP2 parallelization strategy and implementation details.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Generation
:link: design-principles/generation
:link-type: doc

Explore the text generation pipeline and inference mechanisms.

+++
{bdg-info}`Computation`
:::

::::

## Data Management

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Chat Datasets
:link: data-management/chat-datasets
:link-type: doc

Explore the design of chat dataset processing and conversation handling.

+++
{bdg-info}`Data`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Padding
:link: data-management/padding
:link-type: doc

Learn about padding strategies and their impact on training efficiency.

+++
{bdg-secondary}`Optimization`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Sequence Packing and Dynamic Batching
:link: data-management/sequence-packing-and-dynamic-batching
:link-type: doc

Optimize training efficiency with sequence packing and dynamic batching techniques.

+++
{bdg-warning}`Performance`
:::

::::

## Computational Systems

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Training Backends
:link: computational-systems/training-backends
:link-type: doc

Explore the different training backends supported by NeMo RL and their capabilities.

+++
{bdg-info}`Implementation`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Logger
:link: computational-systems/logger
:link-type: doc

Understand the logging system and observability features in NeMo RL.

+++
{bdg-secondary}`Monitoring`
:::

::::

## Development Infrastructure

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Checkpointing
:link: development-infrastructure/checkpointing
:link-type: doc

Understand the checkpointing mechanisms and strategies for model state persistence.

+++
{bdg-success}`Reliability`
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` Environment Variables
:link: development-infrastructure/env-vars
:link-type: doc

Configure NeMo RL behavior with environment variables for debugging, performance, and customization.

+++
{bdg-info}`Configuration`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` UV Package Management
:link: development-infrastructure/uv
:link-type: doc

Learn about UV-based dependency and environment management for NeMo RL development.

+++
{bdg-secondary}`Tools`
:::

::::



