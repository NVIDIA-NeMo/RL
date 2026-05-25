# Number Guessing Game - GRPO Training Example

## Overview

Multi-turn environment where the model learns to guess a secret number (1-20) using binary search strategy. The model gets up to 7 turns with "too high"/"too low" feedback after each guess. Uses `<guess>N</guess>` tags for structured output.

**Reward structure**:
- 1.0 for correct guess
- 0.3 if final guess is within 5 of the target (but not exact)
- 0.0 otherwise

## Data Setup

Procedurally generated - no download needed. Each datum generates a random target number between 1 and 20.

## Training Commands

### 1-GPU Training (50 steps)

```bash
python examples/run_grpo_number_guess.py \
  --config examples/configs/grpo_number_guess.yaml \
  cluster.num_nodes=1 \
  cluster.gpus_per_node=1 \
  grpo.max_num_steps=50 \
  policy.dtensor_cfg.cpu_offload=false \
  logger.wandb_enabled=true \
  logger.wandb.project=nemo-rl-native-env \
  logger.wandb.name=grpo-number-guess-v2-1gpu
```

**Note**: `cpu_offload=false` is required for single GPU (CPUOffload does NOT work on single GPU with AutoModel).

### 8-GPU Training (50 steps)

```bash
python examples/run_grpo_number_guess.py \
  --config examples/configs/grpo_number_guess.yaml \
  cluster.num_nodes=1 \
  cluster.gpus_per_node=8 \
  grpo.max_num_steps=50 \
  grpo.num_prompts_per_step=128 \
  +env.number_guess.num_workers=8 \
  checkpointing.checkpoint_dir=results/grpo-number-guess-8gpu \
  logger.wandb_enabled=true \
  logger.wandb.project=nemo-rl-native-env \
  logger.wandb.name=grpo-number-guess-v2-8gpu
```

## Results

| Run | GPUs | Steps | Final Mean Reward | Accuracy (exact) | wandb |
|-----|------|-------|-------------------|-------------------|-------|
| 1-GPU | 1 | 50 | 1.00 | 99.8% | [link](https://wandb.ai/nvidia/nemo-rl-native-env/runs/hkeeyxc4) |
| 8-GPU | 8 | 50 | 1.00 | 100.0% | [link](https://wandb.ai/nvidia/nemo-rl-native-env/runs/7ikba1zj) |

### Reward Trajectory (1-GPU)

| Step | Mean Reward | Accuracy |
|------|------------|----------|
| 1    | 0.55       | 46.1%    |
| 10   | 0.67       | 58.6%    |
| 20   | 0.93       | 90.0%    |
| 30   | 0.99       | 99.2%    |
| 40   | 1.00       | 99.8%    |
| 50   | 1.00       | 99.8%    |

### Reward Trajectory (8-GPU)

| Step | Mean Reward | Accuracy |
|------|------------|----------|
| 1    | 0.52       | 42.3%    |
| 10   | 0.94       | 91.1%    |
| 20   | 1.00       | 100.0%   |
| 30   | 1.00       | 100.0%   |
| 40   | 1.00       | 99.9%    |
| 50   | 1.00       | 100.0%   |

The 0.6B model learns binary search within ~20-30 steps, reaching near-perfect accuracy on numbers 1-20.
