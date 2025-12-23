---
description: "Step-by-step guide to running Group Relative Policy Optimization (GRPO) with NeMo RL"
categories: ["getting-started"]
tags: ["grpo", "quickstart", "reinforcement-learning", "python-api", "reasoning"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "tutorial"
---

(gs-grpo)=

# Get Started with GRPO

**Group Relative Policy Optimization (GRPO)** is a highly efficient algorithm for reasoning tasks (like Math or Coding). Unlike PPO, it removes the need for a Critic model by using group-based baselines.

:::{card}
**Goal**: Train a model to solve math problems by generating multiple solutions and reinforcing the correct ones.

^^^

**Steps**:

1. **Prepare Data**: Understand the prompt-only format.
2. **Configure**: Set group size and generation parameters.
3. **Train**: Run the `run_grpo_math.py` script.
4. **Verify**: Monitor the group rewards.
:::

:::{button-ref} index
:color: secondary
:outline:
:ref-type: doc

← Previous: Quickstart Guide
:::

---

## Prerequisites

* NeMo RL installed via `uv`.
* A base model capable of some reasoning (e.g., `Qwen/Qwen2.5-1.5B-Instruct`).

---

## 1. Prepare Your Data

GRPO is unique because it primarily needs **Prompts** (questions) and a way to verify the answer (Ground Truth).

NeMo RL's math example uses the **OpenMathInstruct-2** dataset format.

### The Data Format

```json
{
  "problem": "What is 2 + 2?",
  "generated_solution": "...",
  "expected_answer": "4"
}
```

* **problem**: The input prompt given to the model.
* **expected_answer**: Used by the reward function (rule-based verifier) to score the model's output.

You don't need a pre-built preference dataset like DPO; the algorithm generates its own data during training.

---

## 2. Configure the Job

The configuration is located at `examples/configs/grpo_math_1B.yaml`.

Key parameters for GRPO:

* **`grpo.num_generations_per_prompt`**: The group size ($G$). The model generates this many outputs for *each* prompt (e.g., 16).
* **`grpo.num_prompts_per_step`**: How many unique prompts to process in one batch.
* **`policy.model_name`**: The model being trained.

---

## 3. Run the Training

Run the `examples/run_grpo_math.py` script. This example uses a deterministic "Math Verifier" to reward correct answers.

::::{tab-set}

:::{tab-item} Native PyTorch (DTensor)
```bash
uv run python examples/run_grpo_math.py \
  grpo.max_num_steps=100 \
  grpo.num_generations_per_prompt=4
```
:::

:::{tab-item} Megatron Core
```bash
uv run python examples/run_grpo_math.py \
  --config examples/configs/grpo_math_1B_megatron.yaml \
  grpo.max_num_steps=100 \
  grpo.num_generations_per_prompt=4
```
:::

::::

**What's happening?**

1.  **Rollout**: The model generates 4 solutions for each math problem.
2.  **Evaluation**: The system checks each solution against the correct answer. Correct = Reward 1.0, Incorrect = Reward 0.0.
3.  **Update**: The model is updated to make the correct solutions more likely compared to the group average.

---

## 4. Monitor and Verify

Monitor the `reward` metric to see if the model is improving. Note that some models may plateau early if they have already been fine-tuned or RL-trained for the task.

Training can destabilize over time, so also monitor `policy_kl_error` and `token_mult_prob_error` to detect instability. For detailed information about these metrics, see the [GRPO Guide](../guides/grpo.md#metrics).

### Output Artifacts

Results are saved to `results/grpo`:
* **Checkpoints**: The model weights, optimized for reasoning.

---

## Scaling Up

### Multi-GPU Training

GRPO involves heavy generation (inference) and training. For larger models (8B+), you almost certainly need multiple GPUs.

Use the Megatron backend for efficient scaling:

```bash
uv run python examples/run_grpo_math.py \
  --config examples/configs/grpo_math_8B_megatron.yaml \
  cluster.gpus_per_node=8
```

### Async Training

For maximum throughput, you can decouple generation and training. See the [GRPO Guide](../guides/grpo.md) for advanced `async_grpo` configurations.
