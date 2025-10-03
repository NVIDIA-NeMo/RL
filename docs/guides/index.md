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

These guides cover four core areas that address the essential needs of NeMo RL practitioners:

### **Training Algorithms**
Master the fundamental training techniques for reinforcement learning with language models. Learn supervised fine-tuning, preference optimization, and advanced RL algorithms like DPO and GRPO. This section provides the foundation for training high-quality language models with human feedback.

### **Model Development**
Integrate custom models and architectures into NeMo RL training pipelines. Handle model-specific behaviors, special cases, and learn how to extend the framework for new model types. This section is essential for researchers and developers working with custom architectures.

### **Environment and Data**
Set up robust development environments and optimize your training infrastructure. Learn data management strategies. This section helps you build reliable, efficient training workflows.

### **Training Optimization**
Learn optimization techniques including learning rate scheduling, training stability, and hyperparameter optimization to improve your training results.

### **Production and Support**
Troubleshoot common issues and deploy NeMo RL models in production environments. This section ensures your models are ready for real-world applications.





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
:link: model-development/add-new-models
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

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Use DeepSeek Models
:link: model-development/deepseek
:link-type: doc

Convert and configure DeepSeek models for training in NeMo RL.

+++
{bdg-primary}`Practical`
:::

::::

## Environment and Data

This section covers environment setup and data management strategies for NeMo RL development. See [Development](../development/index) for GPU profiling and optimization tools.

## Production and Support

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Troubleshooting
:link: troubleshooting
:link-type: doc

Resolve common problems and errors in production environments.

+++
{bdg-warning}`Support`
:::

::::

For additional learning resources, visit the main [Guides](../index) page.
