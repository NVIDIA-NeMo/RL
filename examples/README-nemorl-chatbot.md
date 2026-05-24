# NeMo-RL Technical Chatbot — GRPO Training with Reward Model

## Overview

Train a technical chatbot that answers NeMo-RL questions using GRPO with a Skywork reward model (RLHF). The policy model (Qwen3-0.6B) learns to generate helpful responses scored by the reward model (Skywork-Reward-V2-Qwen3-0.6B).

## Quick Start

### 1. Generate Data

```bash
uv run python examples/data/build_nemorl_questions.py
```

This creates `examples/data/nemorl_questions.jsonl` with 100 diverse NeMo-RL technical questions.

### 2. Train (2 GPUs: 1 policy + 1 reward model)

```bash
uv run python examples/run_grpo.py \
  --config examples/configs/grpo_nemorl_chatbot.yaml \
  grpo.max_num_steps=30 \
  logger.wandb_enabled=true
```

## Configuration

The config inherits from `grpo_rm_1B.yaml` and overrides:

- **Policy**: Qwen/Qwen3-0.6B (thinking disabled)
- **Reward model**: Skywork/Skywork-Reward-V2-Qwen3-0.6B (Bradley-Terry)
- **Dataset**: `ResponseDataset` with `math_hf_data_processor`
- **Batch**: 20 prompts x 8 generations = 160 samples/step
- **GPUs**: 2 (1 for policy, 1 for reward model)

## Results

Training reward trajectory (wandb): https://wandb.ai/nvidia/nemo-rl-native-env/runs/d63gvjpp

| Step | Mean Reward | KL Penalty |
|------|-------------|------------|
| 6    | 1.19        | 0.0007     |
| 14   | 1.29        | 0.0017     |
| 20   | 2.31        | 0.0064     |
| 25   | 2.55        | 0.0172     |

Mean reward increased from ~1.2 to ~2.5 over 30 steps, showing the policy learned to generate responses preferred by the reward model.
