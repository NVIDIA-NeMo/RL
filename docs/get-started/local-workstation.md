---
description: "Guide to managing local NeMo RL training jobs, GPU selection, and concurrent runs"
categories: ["getting-started"]
tags: ["local-workstation", "ray", "single-node", "how-to", "deployment"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "how-to"
---

# Local Deployment Guide

This guide explains how to manage NeMo RL jobs on your local workstation. NeMo RL uses Ray to abstract away infrastructure, meaning code that runs locally works seamlessly on clusters.

## Automatic Cluster Management

When you execute a training script (e.g., `uv run ...`), NeMo RL:

1. Checks for an existing Ray cluster.
2. If none is found, it automatically starts a local Ray instance using your available resources (CPUs/GPUs).
3. It shuts down the cluster when the script finishes (unless connected to a persistent Ray server).

You generally do **not** need to start Ray manually.

## Controlling GPU Usage

By default, Ray will detect and use all available GPUs. To restrict a job to specific GPUs, use the `CUDA_VISIBLE_DEVICES` environment variable.

**Example: Run on specific GPUs**

```bash
# Only use GPU 0 and 3
CUDA_VISIBLE_DEVICES=0,3 uv run examples/run_grpo_math.py
```

## Running Concurrent Jobs

You can run multiple independent training jobs on the same machine by isolating them to different GPUs. Each job will spin up its own isolated Ray "cluster" instance within that GPU scope.

**Terminal 1 (Job A)**:

```bash
CUDA_VISIBLE_DEVICES=0 uv run examples/run_grpo_math.py
```

**Terminal 2 (Job B)**:

```bash
CUDA_VISIBLE_DEVICES=1 uv run examples/run_sft.py
```

## Monitoring

### Ray Dashboard

When a job starts, Ray provides a dashboard URL (usually `http://127.0.0.1:8265`) in the logs.
* Open this URL in your browser to view actor status, logs, and resource utilization.

### Weights & Biases

If you set `WANDB_API_KEY`, metrics are streamed to W&B. This is the recommended way to track training curves (loss, reward, KL divergence).

## Troubleshooting Local Runs

* **"Resources not available"**: If a job hangs, check if another Ray instance is holding the GPUs. You may need to manually stop stray Ray processes:

  ```bash
  ray stop
  ```

* **OOM Errors**: If you run out of memory, try reducing the batch size or model size in the configuration YAML.
