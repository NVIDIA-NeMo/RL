---
description: "Comprehensive collection of learning resources for mastering reinforcement learning with large language models using NeMo RL"
categories: ["training-algorithms"]
tags: ["learning-resources", "tutorials", "examples", "sft", "dpo", "grpo", "reinforcement-learning", "training-execution"]
personas: ["mle-focused", "researcher-focused", "admin-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

# About NeMo RL Learning Resources

Welcome to the NeMo RL Learning Resources! This comprehensive collection provides everything you need to learn and master reinforcement learning with large language models using NeMo RL.

## What You'll Find Here

These learning resources cover three main categories to help you progress from basic concepts to real-world applications:

### **Tutorials** 
Step-by-step learning guides that teach you NeMo RL concepts and techniques. Start with intermediate tutorials to build your foundation, then progress to advanced topics like distributed training and performance optimization.

### **Examples**
Complete, working examples that demonstrate real-world applications. These examples walk you through entire training workflows using specific datasets and configurations, providing hands-on experience with NeMo RL.

### **Use Cases**
Real-world applications and production patterns for NeMo RL. Learn how to apply reinforcement learning to solve practical problems across different domains including code generation, mathematical reasoning, and conversational AI.

## Learning Path

Follow this structured learning path to master NeMo RL:

### **Intermediate Path** (2-4 weeks)
1. **Installation and Setup** → [Installation Guide](../get-started/installation)
2. **First Training Run** → [Quickstart Guide](../get-started/quickstart)
3. **Custom Environments** → [Build Custom Environments](tutorials/custom-environments)
4. **Design Custom Loss Functions** → [Master Custom Loss Functions](tutorials/custom-loss-functions)
5. **Basic Examples** → [SFT on OpenMathInstruct-2](examples/sft-openmathinstruct2)

### **Advanced Path** (4+ weeks)

2. **Distributed Training and Scaling** → [Scale Training Across Multiple GPUs](tutorials/distributed-training-scaling)
3. **Advanced Examples** → [GRPO on DeepScaleR](examples/grpo-deepscaler)
4. **Use Cases** → [Code Generation](use-cases/code-generation) and [Mathematical Reasoning](use-cases/mathematical-reasoning)

## Tutorials

::::{grid} 1 1 1 2
:gutter: 2 2 2 2



:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Custom Environments
:link: tutorials/custom-environments
:link-type: doc

Build custom environments for reinforcement learning with domain-specific tasks.

+++
{bdg-info}`Intermediate`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Design Custom Loss Functions
:link: tutorials/custom-loss-functions
:link-type: doc

Implement custom loss functions for specialized training objectives and domain-specific requirements.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Distributed Training and Scaling
:link: tutorials/distributed-training-scaling
:link-type: doc

Scale your training across multiple GPUs and nodes with advanced distributed training techniques.

+++
{bdg-warning}`Advanced`
:::

::::

## Examples

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` SFT on OpenMathInstruct-2
:link: examples/sft-openmathinstruct2
:link-type: doc

Complete example of supervised fine-tuning on math instruction dataset.

+++
{bdg-primary}`Example`
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` GRPO on DeepScaleR
:link: examples/grpo-deepscaler
:link-type: doc

Large-scale distributed training example with DeepScaleR integration.

+++
{bdg-secondary}`Cloud`
:::





::::

## Use Cases

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Code Generation
:link: use-cases/code-generation
:link-type: doc

Train models to generate, debug, and optimize code across multiple programming languages.

+++
{bdg-primary}`Development`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Mathematical Reasoning
:link: use-cases/mathematical-reasoning
:link-type: doc

Build models that can solve complex mathematical problems with step-by-step reasoning.

+++
{bdg-warning}`Advanced`
:::

::::





## Next Steps

After completing the learning resources:

- **Explore Features**: Dive into distributed computing and performance optimization
- **Build Your Workflows**: Create custom experiment pipelines
- **Optimize Performance**: Learn best practices for performance optimization
- **Contribute**: Share your experiences and contribute to the community

For additional learning resources and community support, visit the NeMo RL GitHub repository and documentation.

:::{note}
The tutorial files referenced in this guide are available in the NeMo RL examples repository. Clone the repository to access the complete tutorial notebooks and scripts.
:::

Start with the learning path that matches your experience level, and gradually work your way through the learning resource series. The advanced tutorials will help you master complex scenarios and performance optimization.
