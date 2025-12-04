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

Welcome to NeMo RL!

:::{card}

**Goal**: Install NeMo RL and run your first local training job.

^^^

**Steps**:

1. Install `uv` and system prerequisites
2. Clone the repository and initialize the environment
3. Run a sample GRPO training job to verify installation

:::

## Prerequisites

* **OS**: Linux (Ubuntu 22.04/20.04 recommended)
* **Hardware**: 
  * NVIDIA GPU (Volta/Compute Capability 7.0+ required)
  * Sufficient VRAM for the model and batch sizes configured in the example (memory requirements vary by configuration; reduce batch sizes if you encounter out-of-memory errors)
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
   If you cloned without the `--recursive` flag, you may need to rebuild virtual environments:
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
   * The Ray dashboard URL will appear in the logs (typically `http://127.0.0.1:8265`).

   **Example output**:
   ```
   Initializing Ray cluster...
   Ray dashboard available at http://127.0.0.1:8265
   Loading model: Qwen/Qwen2.5-1.5B-Instruct
   Training started...
   Step 1: reward=0.25, policy_kl_error=0.001
   Step 2: reward=0.31, policy_kl_error=0.002
   ...
   ```

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

You can run independent training jobs on the same machine by isolating them to different GPUs. Each job spins up its own isolated Ray instance.

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

When a job starts, Ray provides a dashboard URL (`http://127.0.0.1:8265`) in the logs. Open this URL in your browser to view actor status, logs, and resource usage.

**Weights & Biases**

If you set `WANDB_API_KEY`, metrics stream to W&B. This is the recommended way to track training curves (loss, reward, KL divergence).
:::

:::{dropdown} ℹ️ How NeMo RL manages the local cluster

When you execute a training script (for example, `uv run ...`), NeMo RL:

1. Checks for an existing Ray cluster.
2. If no cluster exists, it automatically starts a local Ray instance using your available resources.
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

### How It Works

NeMo RL uses a distributed architecture built on **Ray** to coordinate multiple components (RL Actors) during training:

* **Policy Model**: The model being trained (e.g., Qwen, Llama)
* **Generation Backend**: Fast inference engine (vLLM) that generates responses
* **Environment**: Reward evaluator (e.g., Math verifier) that scores outputs
* **Training Backend**: PyTorch DTensor or Megatron Core for efficient distributed training

Ray manages resource allocation, process isolation, and communication between these components, allowing NeMo RL to scale seamlessly from a single GPU to multi-node clusters.

:::{seealso}
For more details on the architecture, design philosophy, and how RL Actors coordinate, refer to:
* [NeMo RL Overview](../about/overview.md) - High-level introduction and capabilities
* [Design and Philosophy](../design-docs/design-and-philosophy.md) - Deep dive into the architecture
* [Training Backends](../about/backends.md) - PyTorch DTensor vs Megatron Core comparison
:::

## 3. Choose Your Path

Now that you have a working setup, choose the workflow that matches your goal.

::::{grid} 1 1 1 3
:gutter: 2

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Fine-Tune (SFT)
:link: gs-sft
:link-type: ref
**Start here** if you have a base model and want to teach it instructions. SFT is the standard first step in aligning language models using supervised learning on instruction-response pairs.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Align (DPO)
:link: gs-dpo
:link-type: ref
**Start here** if you have preference data (chosen vs rejected pairs) and want to align your model to human preferences. DPO learns directly from preference comparisons without needing a separate reward model.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Reinforce (GRPO)
:link: gs-grpo
:link-type: ref
**Start here** for reasoning tasks (math, coding) where you can verify correctness programmatically. GRPO is efficient for on-policy RL without requiring a separate critic model—perfect for tasks with deterministic rewards.
:::

::::

## Advanced Setup

* **Cluster Setup**: Ready to scale? Set up multi-node training on [Slurm or Kubernetes](cluster.md).
