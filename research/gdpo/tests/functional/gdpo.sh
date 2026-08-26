#!/bin/bash
set -euo pipefail

uv run research/gdpo/gdpo.py \
    --config research/gdpo/configs/gdpo_llada_8b.yaml \
    grpo.max_num_steps=1 \
    grpo.val_period=0 \
    grpo.val_at_start=false \
    checkpointing.enabled=false \
    logger.wandb_enabled=false \
    logger.tensorboard_enabled=false
