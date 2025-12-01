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

Welcome to NeMo RL! This guide gets you up and running in less than 15 minutes.

:::{card}

**Goal**: Install NeMo RL and run your first local training job.

^^^

**In this tutorial, you will**:

1. Install `uv` and system prerequisites
2. Clone the repository and initialize the environment
3. Run a sample GRPO training job to verify installation

:::

## Prerequisites

* **OS**: Linux (Ubuntu 22.04/20.04 recommended)
* **Hardware**: NVIDIA GPU (Volta/Compute Capability 7.0+ required)
* **Software**: 
  * Python 3.12+
  * CUDA 12+
  * Git

## 1. Installation

:::{seealso}
For detailed system requirements, bare-metal setup (non-container), and troubleshooting, refer to the [Comprehensive Installation Guide](installation.md).
:::

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

   :::{warning}
   If you previously ran without checking out submodules, you may need to rebuild virtual environments:
   `NRL_FORCE_REBUILD_VENVS=true uv sync`
   :::

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

   ::::{tab-set}

   :::{tab-item} Native PyTorch (DTensor)
   ```bash
   uv run python examples/run_grpo_math.py
   ```
   :::

   :::{tab-item} Megatron Core
   ```bash
   uv run examples/run_grpo_math.py \
     --config examples/configs/grpo_math_1B_megatron.yaml
   ```
   :::

   ::::

   **What to expect**:
   * NeMo RL will automatically start a local Ray cluster on your machine.
   * It will download a small model (`Qwen/Qwen2.5-1.5B-Instruct` or similar) and dataset.
   * You should see training logs indicating "Training started" and loss metrics streaming.

### Local Development Tips

Use these tips to manage your local resources and troubleshoot.

:::{dropdown} 💡 How to Control GPU Usage & Run Concurrent Jobs

**Controlling GPU Usage**

By default, Ray detects and uses all available GPUs. To restrict a job to specific GPUs, use `CUDA_VISIBLE_DEVICES`:

```bash
# Only use GPU 0 and 3
CUDA_VISIBLE_DEVICES=0,3 uv run examples/run_grpo_math.py
```

**Running Concurrent Jobs**

You can run multiple independent training jobs on the same machine by isolating them to different GPUs. Each job spins up its own isolated Ray instance.

**Terminal 1 (Job A)**:

```bash
CUDA_VISIBLE_DEVICES=0 uv run examples/run_grpo_math.py
```

**Terminal 2 (Job B)**:

```bash
CUDA_VISIBLE_DEVICES=1 uv run examples/run_sft.py
```

:::

:::{dropdown} 🔍 Monitoring & Logs

**Ray Dashboard**

When a job starts, Ray provides a dashboard URL (usually `http://127.0.0.1:8265`) in the logs. Open this URL in your browser to view actor status, logs, and resource utilization.

**Weights & Biases**

If you set `WANDB_API_KEY`, metrics stream to W&B. This is the recommended way to track training curves (loss, reward, KL divergence).
:::

:::{dropdown} ℹ️ How NeMo RL manages the local cluster

When you execute a training script (for example, `uv run ...`), NeMo RL:

1. Checks for an existing Ray cluster.
2. If none is found, it automatically starts a local Ray instance using your available resources.
3. It shuts down the cluster when the script finishes (unless connected to a persistent Ray server).

You generally do **not** need to start Ray manually.
:::

:::{dropdown} 🛠️ Troubleshooting

* **"Resources not available"**: If a job hangs, check if another Ray instance is holding the GPUs. You may need to manually stop stray Ray processes:

  ```bash
  ray stop
  ```

* **OOM Errors**: If you run out of memory, try reducing the batch size or model size in the configuration YAML.
:::

## 3. Choose Your Path

Now that you have a working setup, choose the workflow that matches your goal.

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

* **Cluster Setup**: Ready to scale? Set up multi-node training on [Slurm or Kubernetes](cluster.md).
