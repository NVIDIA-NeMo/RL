---
description: "Comprehensive API documentation for NeMo RL framework including distributed computing, models, algorithms, and data interfaces"
categories: ["reference"]
tags: ["api", "reference", "distributed", "models", "algorithms", "data-interfaces", "python-api"]
personas: ["mle-focused", "researcher-focused"]
difficulty: "reference"
content_type: "reference"
modality: "universal"
---

# About NeMo RL API

Welcome to the NeMo RL API reference documentation. This section provides detailed technical documentation for all modules, classes, functions, and interfaces in the NeMo RL codebase.

## Overview

This API reference is auto-generated from the NeMo RL source code and provides:

- **Class and function signatures** with parameter types and return values
- **Docstring documentation** explaining usage and behavior
- **Module organization** showing how components relate to each other
- **Source code links** for diving deeper into implementation details

For conceptual guides and tutorials, see the [Guides](../guides/index) section. For architectural overview, see [About NeMo RL](../about/index).

## API Modules

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Distributed Computing
:link: nemo_rl/nemo_rl.distributed
:link-type: doc

Core distributed computing abstractions including `VirtualCluster`, `WorkerGroup`, and communication primitives.

+++
{bdg-primary}`Core`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Models and Policies
:link: nemo_rl/nemo_rl.models
:link-type: doc

Model interfaces, policy implementations, and generation backends for Hugging Face, Megatron, and vLLM.

+++
{bdg-info}`Models`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Algorithms
:link: nemo_rl/nemo_rl.algorithms
:link-type: doc

RL training algorithms including DPO, GRPO, SFT, loss functions, and training utilities.

+++
{bdg-warning}`Training`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Data and Environments
:link: nemo_rl/nemo_rl.data
:link-type: doc

Dataset interfaces, data processing utilities, and RL environment abstractions.

+++
{bdg-secondary}`Data`
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Environments
:link: nemo_rl/nemo_rl.environments
:link-type: doc

Environment interfaces and implementations for math problems, games, and custom tasks.

+++
{bdg-info}`Environments`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Converters
:link: nemo_rl/nemo_rl.converters
:link-type: doc

Model conversion and export utilities for vLLM deployment and format transformations.

+++
{bdg-success}`Deployment`
:::

:::{grid-item-card} {octicon}`meter;1.5em;sd-mr-1` Evaluation
:link: nemo_rl/nemo_rl.evals
:link-type: doc

Evaluation frameworks, metrics computation, and model assessment utilities.

+++
{bdg-warning}`Evaluation`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Utilities
:link: nemo_rl/nemo_rl.utils
:link-type: doc

Logging, configuration, checkpointing, profiling, and other helper utilities.

+++
{bdg-secondary}`Utilities`
:::

::::

## Key Interfaces

### PolicyInterface

The core interface for RL policies. Implement this to create custom policy implementations:

- `nemo_rl.models.policy.interfaces.PolicyInterface` - Abstract base class for policies
- `nemo_rl.models.policy.lm_policy.Policy` - Main policy implementation

### GenerationInterface

Unified interface for text generation across backends:

- `nemo_rl.models.generation.interfaces.GenerationInterface` - Base generation interface
- `nemo_rl.models.generation.vllm.VllmGeneration` - vLLM-based generation

### EnvironmentInterface

Interface for RL environments:

- `nemo_rl.environments.interfaces.EnvironmentInterface` - Base environment interface
- `nemo_rl.environments.math_environment.MathEnvironment` - Math problem environment

## Common Patterns

### Creating a Virtual Cluster

```python
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster

# Single node, 4 GPUs
cluster = RayVirtualCluster(
    bundle_ct_per_node_list=[4],
    use_gpus=True,
    num_gpus_per_node=4
)
```

### Setting Up a Worker Group

```python
from nemo_rl.distributed.worker_groups import RayWorkerGroup

worker_group = RayWorkerGroup(
    virtual_cluster=cluster,
    actor_class=PolicyWorker,
    num_workers=4
)
```

### Implementing Custom Loss Functions

```python
from nemo_rl.algorithms.interfaces import LossFunctionInterface

class CustomLoss(LossFunctionInterface):
    def compute_loss(self, batch, model_output):
        # Your loss computation
        pass
```

## Module Index

Browse the complete API by module:

- [nemo_rl.algorithms](nemo_rl/nemo_rl.algorithms) - Training algorithms
- [nemo_rl.converters](nemo_rl/nemo_rl.converters) - Model converters
- [nemo_rl.data](nemo_rl/nemo_rl.data) - Data utilities
- [nemo_rl.distributed](nemo_rl/nemo_rl.distributed) - Distributed computing
- [nemo_rl.environments](nemo_rl/nemo_rl.environments) - RL environments
- [nemo_rl.evals](nemo_rl/nemo_rl.evals) - Evaluation tools
- [nemo_rl.experience](nemo_rl/nemo_rl.experience) - Experience management
- [nemo_rl.metrics](nemo_rl/nemo_rl.metrics) - Metrics and logging
- [nemo_rl.models](nemo_rl/nemo_rl.models) - Models and policies
- [nemo_rl.utils](nemo_rl/nemo_rl.utils) - Utility functions

## Additional Resources

- [Core Architecture](../about/architecture-overview) - System design and components
- [Core Design](../core-design/index) - Design principles and patterns
- [Guides](../guides/index) - Task-focused how-to guides
- [Examples](../learning-resources/examples/index) - Working code examples
