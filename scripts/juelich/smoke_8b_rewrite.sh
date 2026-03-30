#!/bin/bash
# Smoke test: 8B→8B rewrite teacher on develbooster (4 nodes × 4 GPUs).
# Runs 10 steps only, no eval, for quick debugging.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="examples/configs/opsd/juelich/distill-1.7b-to-1.7b-4n4g-gen16k-rewrite.yaml"
CKPT="/p/scratch/scifi/kryeziu1/opsd/checkpoints"

export COMMAND="uv run python examples/run_distillation.py --config ${CONFIG} \
    policy.model_name=Qwen/Qwen3-8B \
    teacher.model_name=Qwen/Qwen3-8B \
    policy.generation.vllm_cfg.tensor_parallel_size=4 \
    distillation.val_at_start=false \
    distillation.val_at_end=false \
    distillation.max_num_steps=10 \
    distillation.val_period=100 \
    checkpointing.enabled=false \
    checkpointing.checkpoint_dir=${CKPT}/smoke-8b-rewrite \
    logger.log_dir=logs/smoke-8b-rewrite \
    logger.wandb.name=smoke-8b-rewrite"

sbatch \
    --nodes=4 \
    --partition=develbooster \
    --time=02:00:00 \
    --job-name="smoke-8b-rw" \
    "$SCRIPT_DIR/ray.sub"

echo "Smoke test submitted."
