---
description: "Essential tools and workflows for testing, debugging, profiling, and contributing to NeMo RL"
categories: ["development"]
tags: ["development", "debugging", "testing", "documentation", "contributing", "profiling", "fp8", "vllm"]
personas: ["contributor-focused", "devops-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "guide"
modality: "universal"
---

# About NeMo RL Development

Welcome to the NeMo RL Development guide! This section covers essential tools and workflows for testing, debugging, profiling, and contributing to NeMo RL.

## Overview

Whether you're contributing to NeMo RL, debugging issues, running tests, or building documentation, this guide provides the workflows and tools you need to be productive.

## What You'll Find Here

- **Testing**: Run unit tests, functional tests, and track performance metrics
- **Debugging**: Debug distributed Ray applications with the Ray Distributed Debugger
- **Building Documentation**: Build, develop, and contribute to NeMo RL documentation
- **GPU Profiling**: Profile GPU performance with Nsys
- **FP8 Quantization**: Enable FP8 quantization for efficient training
- **Custom vLLM**: Experiment with custom vLLM implementations

## Quick Navigation

::::{grid} 1 2 2 3
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Testing
:link: testing
:link-type: doc

Run unit tests, functional tests, Docker-based tests, and track metrics.

+++
{bdg-primary}`Quality Assurance`
:::

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Debugging
:link: debugging
:link-type: doc

Debug NeMo RL applications using Ray distributed debugger for worker/actor processes and driver scripts.

+++
{bdg-success}`Quality`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Build Documentation
:link: BUILD_INSTRUCTIONS
:link-type: doc

Build and contribute to NeMo RL documentation with Sphinx.

+++
{bdg-info}`Documentation`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` GPU Profiling
:link: nsys-profiling
:link-type: doc

Profile GPU performance and optimize training with Nsys.

+++
{bdg-secondary}`Performance`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` FP8 Quantization
:link: fp8
:link-type: doc

Enable FP8 quantization for efficient model training and inference.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Custom vLLM
:link: use-custom-vllm
:link-type: doc

Experiment with custom vLLM implementations and modifications.

+++
{bdg-warning}`Advanced`
:::

::::

## Development Best Practices

### Set Up Your Environment

Before starting development work:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/NVIDIA-NeMo/RL.git
   cd RL
   ```

2. **Install dependencies**:

   ```bash
   uv sync --all-extras --all-groups
   ```

3. **Run tests to verify setup**:

   ```bash
   uv run --group test bash tests/run_unit.sh
   ```

### Development Cycle

A typical development workflow:

1. **Write code** with appropriate tests
2. **Run tests locally** to verify functionality
3. **Debug issues** using the Ray debugger if needed
4. **Build documentation** if adding new features
5. **Submit PR** with tests and docs

### Get Help

- **Issues**: Report bugs on [GitHub Issues](https://github.com/NVIDIA-NeMo/RL/issues)
- **Discussions**: Post questions in [GitHub Discussions](https://github.com/NVIDIA-NeMo/RL/discussions)
- **Contributing**: See the [Contributing Guide](https://github.com/NVIDIA-NeMo/RL/blob/main/CONTRIBUTING.md)

