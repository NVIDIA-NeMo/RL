---
description: "Quickstart guide to installing and running your first NeMo RL training job"
categories: ["getting-started"]
tags: ["quickstart", "installation", "tutorial"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
---

(gs-overview)=
# Quickstart Guide

Welcome to NeMo RL! This guide gets you up and running in less than 15 minutes. You will install the library, set up your environment, and run a sample training job on your local machine.

## Prerequisites

* **OS**: Linux (Ubuntu 22.04/20.04 recommended)
* **Hardware**: NVIDIA GPU (Volta/Compute Capability 7.0+ required)
* **Software**: 
  * Python 3.10+
  * CUDA 12+
  * Git

## 1. Installation

We use `uv` for fast, reliable package management.

1. **Install `uv`** (if not installed):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   ```

2. **Clone NeMo RL**:
   Clone the repository with submodules to include all dependencies.

   ```bash
   git clone git@github.com:NVIDIA-NeMo/RL.git nemo-rl --recursive
   cd nemo-rl
    
   # If you cloned without recursive, run:
   # git submodule update --init --recursive
   ```

3. **Initialize Environment**:
   Create the virtual environment.

   ```bash
   uv venv
   ```

   ```{note}
   Do not use `-p/--python`. `uv` will automatically read the correct Python version from `.python-version`.
   ```

## 2. Run Your First Job (Local)

Let's verify your installation by running a **Group Relative Policy Optimization (GRPO)** training job. This example fine-tunes a small model on a math dataset.

1. **Set Environment Variables**:

   ```bash
   export HF_HOME=/path/to/your/hf_cache
   export HF_DATASETS_CACHE=/path/to/your/hf_datasets_cache
   # Optional: For logging
   # export WANDB_API_KEY=your_key
   ```

2. **Run the Training Script**:
   Use `uv run` to execute the script. This automatically handles dependencies.

   ```bash
   uv run python examples/run_grpo_math.py
   ```

   **What to expect**:
   * NeMo RL will automatically start a local Ray cluster on your machine.
   * It will download a small model (`Qwen/Qwen2.5-1.5B-Instruct` or similar) and dataset.
   * You should see training logs indicating "Training started" and loss metrics streaming.

## 3. Choose Your Path

Now that you have a working setup, choose the workflow that matches your goal:

::::{grid} 1 1 1 3
:gutter: 2

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Fine-Tune (SFT)
:link: gs-sft
:link-type: ref
Start here if you have a base model and want to teach it instructions.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Align (DPO)
:link: gs-dpo
:link-type: ref
Start here if you have preference data (A vs B) and want to align your model.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Reinforce (GRPO)
:link: gs-grpo
:link-type: ref
Dive deeper into RL with GRPO, configuring rewards and complex reasoning tasks.
:::

::::

## Advanced Setup

*   **Local Management**: Learn how to control GPUs and run concurrent jobs in the [Local Deployment Guide](local-workstation.md).
*   **Cluster Setup**: Ready to scale? Set up multi-node training on [Slurm or Kubernetes](cluster.md).
