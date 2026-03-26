#!/bin/bash
# Smoke test: sampled token distillation on develbooster (2 nodes, 2 steps)
# Uses ray.sub for multi-node Ray setup.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="examples/configs/opsd/juelich/std-1.7b-to-1.7b-2n4g-gen8k.yaml"
CKPT="/p/scratch/scifi/kryeziu1/opsd/checkpoints"

export COMMAND="uv run python examples/run_sampled_token_distillation.py --config ${CONFIG} \
    sampled_token_distillation.max_num_steps=2 \
    sampled_token_distillation.val_at_start=false \
    sampled_token_distillation.val_period=0 \
    checkpointing.enabled=false \
    checkpointing.checkpoint_dir=${CKPT}/std-test-1.7b-self \
    logger.log_dir=logs/std-test-1.7b-self \
    logger.wandb.name=std-test-1.7b-self \
    logger.wandb_enabled=false"

sbatch --nodes=2 --time=02:00:00 --partition=develbooster --job-name="std-test" "$SCRIPT_DIR/ray.sub"
