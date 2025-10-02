---
description: "Essential workflows for developing, testing, and contributing to the NeMo RL framework"
categories: ["guides"]
tags: ["development", "debugging", "testing", "documentation", "contributing", "workflows"]
personas: ["contributor-focused", "devops-focused"]
difficulty: "intermediate"
content_type: "guide"
modality: "universal"
---

# Development Workflows

Welcome to the NeMo RL Development Workflows guide! This section covers essential workflows and tools for developing, testing, debugging, and contributing to the NeMo RL framework.

## Overview

Whether you're contributing to NeMo RL, debugging issues, running tests, or building documentation, this guide provides the workflows and tools you need to be productive.

## What You'll Find Here

- **Debugging**: Debug distributed Ray applications with the Ray Distributed Debugger
- **Testing**: Run unit tests, functional tests, and track performance metrics
- **Building Documentation**: Build, develop, and contribute to NeMo RL documentation

## Quick Navigation

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Debugging
:link: debugging
:link-type: doc

Debug NeMo RL applications using Ray distributed debugger for worker/actor processes and driver scripts.

+++
{bdg-success}`Quality`
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Testing
:link: testing
:link-type: doc

Run unit tests, functional tests, Docker-based tests, and track metrics.

+++
{bdg-primary}`Quality Assurance`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Documentation
:link: documentation
:link-type: doc

Build and contribute to NeMo RL documentation with Sphinx.

+++
{bdg-info}`Documentation`
:::

::::

## Development Best Practices

### Setting Up Your Environment

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

### Getting Help

- **Issues**: Report bugs on [GitHub Issues](https://github.com/NVIDIA-NeMo/RL/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/NVIDIA-NeMo/RL/discussions)
- **Contributing**: See our [Contributing Guide](https://github.com/NVIDIA-NeMo/RL/blob/main/CONTRIBUTING.md)

---

::::{toctree}
:hidden:
:caption: Development Workflows
:maxdepth: 2
debugging
testing
documentation
::::

