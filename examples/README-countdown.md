# Countdown Game -- GRPO Training Example

## Overview

The Countdown Game is a classic number puzzle: given a list of numbers (e.g. `[2, 5, 7, 11]`) and a target (e.g. `38`), combine the numbers using `+`, `-`, `*`, `/` to reach the target. Each number may be used at most once. This is the famous "Mini-R1" GRPO example.

Puzzles are procedurally generated and guaranteed to be solvable. Numbers are drawn from the classic Countdown pool (1-10, 25, 50, 75, 100), with 3-6 numbers per puzzle and targets in the range 1-999.

## Quick Start

### Train (1 GPU)

```bash
python examples/run_grpo_countdown.py \
  --config examples/configs/grpo_countdown.yaml \
  cluster.num_nodes=1 cluster.gpus_per_node=1 \
  grpo.max_num_steps=50 \
  grpo.num_prompts_per_step=16 \
  grpo.num_generations_per_prompt=8 \
  policy.train_global_batch_size=128 \
  logger.wandb_enabled=true \
  policy.dtensor_cfg.cpu_offload=false
```

### Train (8 GPU)

```bash
python examples/run_grpo_countdown.py \
  --config examples/configs/grpo_countdown.yaml \
  cluster.num_nodes=1 cluster.gpus_per_node=8 \
  grpo.max_num_steps=50 \
  grpo.num_prompts_per_step=64 \
  grpo.num_generations_per_prompt=16 \
  policy.train_global_batch_size=1024 \
  logger.wandb_enabled=true \
  logger.wandb.name=grpo-countdown-8gpu
```

## Results

Model: Qwen/Qwen3-0.6B, 50 training steps. Reward = fraction of correct countdown solutions.

| Run | Steps | Mean Reward (start) | Mean Reward (end) | W&B Link |
|-----|-------|--------------------|--------------------|----------|
| 1-GPU (16x8) | 50 | 0.047 | 0.867 | [wandb](https://wandb.ai/nvidia/nemo-rl-native-env/runs/s64hunlh) |
| 8-GPU (64x16) | 50 | 0.041 | 0.865 | [wandb](https://wandb.ai/nvidia/nemo-rl-native-env/runs/pnti5c3d) |

Both runs show a clear learning curve from ~4% to ~87% accuracy over 50 steps.
