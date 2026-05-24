# 24 Game - GRPO Training Example

## Overview

The 24 Game is a single-turn RL environment where the model is given 4 numbers (1-13) and must find a mathematical expression using +, -, *, / that evaluates to exactly 24. The environment uses AST-based safe evaluation (no exec/eval) and procedurally generates solvable puzzles verified before presentation.

## Quick Start

### Data Setup

No data download needed. Puzzles are procedurally generated on-the-fly. Each puzzle is verified solvable before being presented to the model.

### Train (1 GPU)

```bash
python examples/run_grpo_twenty_four_game.py \
  --config examples/configs/grpo_twenty_four_game.yaml \
  cluster.num_nodes=1 \
  cluster.gpus_per_node=1 \
  grpo.max_num_steps=50 \
  policy.dtensor_cfg.cpu_offload=false \
  logger.wandb_enabled=true \
  logger.wandb.project=nemo-rl-native-env \
  logger.wandb.name=grpo-24-game-v2-1gpu
```

**Note:** `cpu_offload=false` is required for single-GPU runs (CPUOffload does not work on single GPU with AutoModel).

### Train (8 GPU)

```bash
python examples/run_grpo_twenty_four_game.py \
  --config examples/configs/grpo_twenty_four_game.yaml \
  cluster.num_nodes=1 \
  cluster.gpus_per_node=8 \
  grpo.max_num_steps=50 \
  grpo.num_prompts_per_step=128 \
  logger.wandb_enabled=true \
  logger.wandb.project=nemo-rl-native-env \
  logger.wandb.name=grpo-24-game-v2-8gpu
```

## Results

| Run | GPUs | Steps | Final Mean Reward | Validation Accuracy | wandb |
|-----|------|-------|-------------------|---------------------|-------|
| 1-GPU | 1 | 50 | 0.39 | 47.3% | [link](https://wandb.ai/nvidia/nemo-rl-native-env/runs/i8bimrqg) |
| 8-GPU | 8 | 50 | TBD | TBD | TBD |

### Reward Trajectory (1-GPU)

| Step | Mean Reward |
|------|------------|
| 1    | 0.09       |
| 10   | 0.18       |
| 20   | 0.36       |
| 30   | 0.47       |
| 40   | 0.47       |
| 50   | 0.39       |

The model learns to solve ~40-50% of 24 Game puzzles within 50 GRPO steps, starting from ~9% with a 0.6B parameter base model (Qwen3-0.6B).
