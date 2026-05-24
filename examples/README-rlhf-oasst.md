# RLHF Training: OpenAssistant + Skywork Reward Model

This example demonstrates RLHF training on instruction-following conversations
using the GRPO algorithm with a Bradley-Terry reward model.

## Overview

- **Dataset**: [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) -- first-turn user prompts extracted from instruction-following conversations
- **Policy**: Qwen/Qwen3-0.6B (thinking disabled)
- **Reward Model**: Skywork/Skywork-Reward-V2-Qwen3-0.6B (Bradley-Terry)
- **Algorithm**: GRPO with 20 prompts/step, 8 generations/prompt

## Data Preparation

Extract first-turn user prompts from OASST into a JSONL file:

```bash
python examples/data/create_oasst_prompts.py
```

This creates `examples/data/oasst_prompts.jsonl` with ~14k unique user prompts.

## Training Command

```bash
python examples/run_grpo.py \
  --config examples/configs/grpo_rlhf_oasst.yaml \
  grpo.max_num_steps=30 \
  logger.wandb_enabled=true
```

## Results

Training run (30 steps, 2 GPUs): https://wandb.ai/nvidia/nemo-rl-native-env/runs/5ebm96lu
