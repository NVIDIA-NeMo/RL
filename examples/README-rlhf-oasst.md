# RLHF Training: OpenAssistant + Skywork Reward Model

This example demonstrates RLHF training on instruction-following conversations
using the GRPO algorithm with a Bradley-Terry reward model.

## Overview

- **Dataset**: [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) -- multi-turn instruction-following conversations
- **Policy**: Qwen/Qwen3-0.6B (thinking disabled)
- **Reward Model**: Skywork/Skywork-Reward-V2-Qwen3-0.6B (Bradley-Terry)
- **Algorithm**: GRPO with 20 prompts/step, 8 generations/prompt

## Training Command

```bash
uv run python examples/run_grpo.py \
  --config examples/configs/grpo_rlhf_oasst.yaml \
  grpo.max_num_steps=30 \
  logger.wandb_enabled=true
```

## Results

Training run: _TODO: add wandb link after run completes_
