# API Reference

This page contains API reference documentation for NeMo RL components generated from source code.

```{toctree}
:titlesonly:
:maxdepth: 3

nemo_rl/nemo_rl
```

## Package Structure

The NeMo RL package is organized into the following modules:

### Core Algorithms
- **DPO**: Direct Preference Optimization implementation
- **GRPO**: Group Relative Policy Optimization  
- **SFT**: Supervised Fine-Tuning
- **Loss Functions**: Custom loss function framework

### Distributed Computing
- **VirtualCluster**: Resource allocation and management
- **WorkerGroup**: Distributed worker coordination
- **BatchedDataDict**: Efficient distributed data structures

### Models and Policies
- **Policy Interfaces**: Abstract policy implementations
- **Generation Backends**: Text generation with VLLM, Hugging Face, Megatron
- **Model Converters**: Model format conversions

### Data and Environments
- **Dataset Interfaces**: Standardized data loading
- **Environment Abstractions**: RL environment implementations
- **Data Processing**: Tokenization and preprocessing utilities

### Utilities
- **Logging**: Comprehensive logging system
- **Configuration**: Configuration management
- **Checkpointing**: Model checkpoint utilities
- **Profiling**: Performance profiling tools

## Quick Navigation

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Algorithms
:link: nemo_rl/nemo_rl.algorithms
:link-type: doc

DPO, GRPO, SFT, and custom loss functions.

+++
{bdg-warning}`Training`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Distributed
:link: nemo_rl/nemo_rl.distributed
:link-type: doc

VirtualCluster, WorkerGroup, and distributed utilities.

+++
{bdg-primary}`Core`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Models
:link: nemo_rl/nemo_rl.models
:link-type: doc

Policy interfaces, generation backends, and model implementations.

+++
{bdg-info}`Models`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Data
:link: nemo_rl/nemo_rl.data
:link-type: doc

Dataset interfaces, environment abstractions, and data processing.

+++
{bdg-secondary}`Data`
:::

::::

## Detailed API Reference

For complete API documentation with all functions, classes, and parameters, see the full reference: [nemo_rl](nemo_rl/nemo_rl).

This documentation is generated from the source code using [sphinx-autodoc2](https://github.com/chrisjsewell/sphinx-autodoc2) and includes:

- **Function signatures** with parameter types and return values
- **Class hierarchies** and inheritance relationships  
- **Module documentation** and package structure
- **Cross-references** between related components
- **Type annotations** and docstrings 