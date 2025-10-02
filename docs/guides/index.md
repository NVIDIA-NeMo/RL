---
description: "Comprehensive collection of guides for mastering reinforcement learning with large language models using NeMo RL"
categories: ["training-algorithms"]
tags: ["guides", "training-algorithms", "model-development", "environment-data", "reference"]
personas: ["mle-focused", "researcher-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "universal"
---

# About NeMo RL Guides

Welcome to the NeMo RL Guides! This comprehensive collection provides everything you need to master reinforcement learning with large language models using NeMo RL.

## What You'll Find Here

Our guides are organized into five core areas that cover the essential needs of NeMo RL practitioners:

### **Training Algorithms**
Master the fundamental training techniques for reinforcement learning with language models. Learn supervised fine-tuning, preference optimization, and advanced RL algorithms like DPO and GRPO. This section provides the foundation for training high-quality language models with human feedback.

### **Model Development**
Integrate custom models and architectures into NeMo RL training pipelines. Handle model-specific behaviors, special cases, and learn how to extend the framework for new model types. This section is essential for researchers and developers working with custom architectures.

### **Development Workflows**
Essential workflows for contributing to NeMo RL. Learn debugging techniques, testing strategies, and how to build documentation. This section helps contributors and maintainers develop and improve the framework.

### **Environment and Data**
Set up robust development environments and optimize your training infrastructure. Learn performance profiling and data management strategies. This section helps you build reliable, efficient training workflows.

### **Production and Support**
Deploy and maintain NeMo RL models in production environments. Learn testing strategies, packaging, and deployment best practices. This section ensures your models are ready for real-world applications.





## Training Algorithms

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` SFT Training
:link: training-algorithms/sft
:link-type: doc

Supervised Fine-Tuning for language models - the foundation of RL training.

+++
{bdg-primary}`Beginner`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` GRPO Training
:link: training-algorithms/grpo
:link-type: doc

Group Relative Policy Optimization for advanced reinforcement learning training.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`heart;1.5em;sd-mr-1` DPO Training
:link: training-algorithms/dpo
:link-type: doc

Direct Preference Optimization for preference learning and model alignment.

+++
{bdg-info}`Intermediate`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Evaluation
:link: training-algorithms/eval
:link-type: doc

Model evaluation and benchmarking strategies for RL-trained models.

+++
{bdg-secondary}`Analysis`
:::

::::

## Model Development

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`plus;1.5em;sd-mr-1` Add New Models
:link: model-development/adding-new-models
:link-type: doc

Learn how to integrate custom models and architectures into NeMo RL training pipelines.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`alert;1.5em;sd-mr-1` Model Quirks and Special Cases
:link: model-development/model-quirks
:link-type: doc

Handle model-specific behaviors and special cases in NeMo RL.

+++
{bdg-warning}`Advanced`
:::

::::

## Development Workflows

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Debugging
:link: development-workflows/debugging
:link-type: doc

Debug NeMo RL applications using Ray distributed debugger for worker/actor processes.

+++
{bdg-success}`Quality`
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Testing
:link: development-workflows/testing
:link-type: doc

Run unit tests, functional tests, and track metrics for quality assurance.

+++
{bdg-primary}`Quality Assurance`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Documentation
:link: development-workflows/documentation
:link-type: doc

Build and contribute to NeMo RL documentation with Sphinx.

+++
{bdg-info}`Documentation`
:::

::::

## Environment and Data

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` NSYS Profiling
:link: environment-data/nsys-profiling
:link-type: doc

NSYS-specific profiling for RL training performance.

+++
{bdg-secondary}`Performance`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Advanced Performance
:link: ../advanced/performance/index
:link-type: doc

Comprehensive performance optimization and profiling techniques.

+++
{bdg-warning}`Advanced`
:::

::::

## Production and Support

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Troubleshooting
:link: troubleshooting/index
:link-type: doc

Resolve common problems and errors in production environments.

+++
{bdg-warning}`Support`
:::

::::

For additional learning resources, visit the main [Guides](../index) page.

```{toctree}
:hidden:
:maxdepth: 1

training-algorithms/index
model-development/index
development-workflows/index
environment-data/index
training-optimization/index
troubleshooting/index
```