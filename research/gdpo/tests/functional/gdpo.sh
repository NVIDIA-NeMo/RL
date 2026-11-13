#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

set -euo pipefail

cd "$PROJECT_ROOT"

uv run gdpo.py \
    --config configs/gdpo_llada_8b.yaml \
    grpo.max_num_steps=1 \
    grpo.num_prompts_per_step=4 \
    grpo.num_generations_per_prompt=2 \
    grpo.num_iterations=1 \
    grpo.val_period=0 \
    grpo.val_at_start=true \
    grpo.max_val_samples=8 \
    grpo.val_batch_size=8 \
    policy.train_global_batch_size=8 \
    checkpointing.enabled=false \
    logger.wandb_enabled=false \
    logger.tensorboard_enabled=false \
    "$@"
