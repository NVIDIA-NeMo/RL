---
description: "Step-by-step guide to running GRPO training with NeMo RL"
categories: ["getting-started"]
tags: ["grpo", "quickstart", "reinforcement-learning", "python-api"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
---

(gs-grpo)=

# Get Started with GRPO

**Group Relative Policy Optimization (GRPO)** is a reinforcement learning algorithm that eliminates the need for a critic model by using group-based baselines. This makes it highly efficient for reasoning tasks.

## Prerequisites

Ensure you have completed the [Quickstart Installation](index.md). You should have:
*   NeMo RL installed via `uv`
*   Environment variables set (`HF_HOME`, etc.)

## Running GRPO

NeMo RL provides a reference GRPO configuration for math benchmarks using the [OpenInstructMath2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) dataset.

### 1. Basic Command
Run the example with the default DTensor backend:

```bash
uv run python examples/run_grpo_math.py
```

This trains a `Qwen/Qwen2.5-1.5B` model on math problems using a single GPU.

### 2. Using Megatron Core Backend
To scale to larger models, switch to the Megatron Core backend:

```bash
uv run python examples/run_grpo_math.py \
  --config examples/configs/grpo_math_1B_megatron.yaml
```

### 3. Customizing Training
You can override parameters directly from the command line:

```bash
# Scale to 8 GPUs
uv run python examples/run_grpo_math.py cluster.gpus_per_node=8

# Change generation parameters
uv run python examples/run_grpo_math.py \
  grpo.num_prompts_per_step=64 \
  grpo.num_generations_per_prompt=16
```

## Understanding the Output

GRPO training produces specific metrics:
*   **`reward_mean`**: Average reward across the group.
*   **`kl_policy`**: Divergence from the reference model.
*   **`advantage`**: Computed relative to the group mean.

Logs are saved to `results/grpo_math_1B/experiment_0/`.

## How it Works
1.  **Sample**: Generate $G$ completions for each prompt.
2.  **Score**: Evaluate all $G$ completions with the reward function (e.g., correctness).
3.  **Update**: Optimize the policy to increase the likelihood of high-scoring completions relative to the group average.

## Next Steps
*   **Deep Dive**: See the [GRPO Guide](../guides/grpo.md) for algorithm details.
*   **Custom Rewards**: Learn to build [Custom Environments](../guides/environments.md).
*   **Reproduction**: Try the [DeepScaleR Tutorial](../guides/grpo-deepscaler.md).
