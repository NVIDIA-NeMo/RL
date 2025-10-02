---
description: "Step-by-step learning guides to master reinforcement learning with large language models using NeMo RL"
categories: ["training-algorithms"]
tags: ["tutorials", "sft", "dpo", "grpo", "evaluation", "reinforcement-learning", "training-execution"]
personas: ["mle-focused", "researcher-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

# Tutorials

Welcome to the NeMo RL Tutorials! These step-by-step learning guides will help you master reinforcement learning with large language models.

## Tutorial Categories

Our tutorials are organized by difficulty level to help you progress from basic concepts to advanced techniques:

### **Beginner Tutorials**
Perfect for newcomers to NeMo RL. Start here to build your foundation with supervised fine-tuning and basic concepts.

### **Intermediate Tutorials**
For users familiar with the basics. Learn preference-based training, evaluation techniques, and custom environment development.

### **Advanced Tutorials**
For experienced practitioners. Master advanced reinforcement learning, distributed training, and performance optimization.

## Tutorials by Level

::::{grid} 1 2 2 2
:gutter: 2 2 2 2



:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Custom Environments
:link: custom-environments
:link-type: doc

Build custom environments for reinforcement learning with domain-specific tasks.

+++
{bdg-info}`Intermediate`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Design Custom Loss Functions
:link: custom-loss-functions
:link-type: doc

Implement custom loss functions for specialized training objectives and domain-specific requirements.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Distributed Training and Scaling
:link: distributed-training-scaling
:link-type: doc

Scale your training across multiple GPUs and nodes with advanced distributed training techniques.

+++
{bdg-warning}`Advanced`
:::

::::

## Learning Progression

### **Intermediate Path** (2-4 weeks)
1. **Custom Environments** → Build custom environments for RL
2. **Design Custom Loss Functions** → Master custom loss function design
3. **Basic Examples** → Apply concepts to real datasets

### **Advanced Path** (4+ weeks)

2. **Distributed Training and Scaling** → Scale training across multiple GPUs and nodes

## Tutorial Descriptions



### **Custom Environments**
- **Domain-Specific Tasks**: Build custom environments for reinforcement learning
- **Environment Design**: Create specialized environments for your use cases
- **Integration**: Learn to integrate custom environments with NeMo RL
- **Testing**: Implement comprehensive testing for custom environments

### **Design Custom Loss Functions**
- **Specialized Objectives**: Implement custom loss functions for domain-specific requirements
- **Advanced Training**: Master sophisticated loss function design for RL algorithms
- **Performance Optimization**: Optimize loss functions for better training outcomes
- **Debugging**: Learn to debug and validate custom loss functions

### **Distributed Training and Scaling**
- **Multi-GPU Training**: Scale your training across multiple GPUs efficiently
- **Multi-Node Training**: Implement distributed training across multiple nodes
- **Advanced Techniques**: Master advanced distributed training techniques
- **Performance Optimization**: Optimize for maximum training throughput

## Prerequisites

### **For Intermediate Tutorials**
- Basic Python knowledge
- Familiarity with PyTorch (helpful but not required)
- Understanding of machine learning concepts
- Basic knowledge of reinforcement learning concepts

### **For Advanced Tutorials**
- Completion of intermediate tutorials
- Experience with distributed computing (for distributed training)
- Understanding of performance optimization concepts


## Getting Started

1. **Choose Your Level**: Start with intermediate tutorials if you're new to NeMo RL
2. **Follow the Progression**: Complete tutorials in order for best results
3. **Practice with Examples**: Apply what you learn to the example projects
4. **Explore Advanced Topics**: Move to advanced tutorials as you gain experience

## Next Steps

After completing the tutorials:

- **Try the Examples**: Apply your knowledge to real-world examples
- **Explore Use Cases**: Learn about specific applications and domains
- **Contribute**: Share your experiences and help improve the tutorials
- **Advanced Topics**: Dive into research and experimentation

For additional learning resources, visit the main [Learning Resources](../index) page.

---

::::{toctree}
:hidden:
:caption: Tutorials
:maxdepth: 2

custom-environments
custom-loss-functions
distributed-training-scaling
:::: 