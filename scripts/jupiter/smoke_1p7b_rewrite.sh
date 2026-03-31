#!/bin/bash
# Smoke test: 1.7B -> 1.7B rewrite distillation on JUPITER.
# Keeps the same distributed topology as production (2 nodes x 4 GPUs),
# but runs only a few steps with validation/checkpointing disabled.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-examples/configs/opsd/juelich/distill-1.7b-to-1.7b-4n4g-gen16k-rewrite.yaml}"
RAY_SUB="${RAY_SUB:-$SCRIPT_DIR/ray.sub}"

if [[ -d "/e/scratch/scifi/yll/checkpoints" ]]; then
  DEFAULT_CKPT_ROOT="/e/scratch/scifi/yll/checkpoints"
else
  DEFAULT_CKPT_ROOT="${REPO_ROOT}/checkpoints"
fi
CKPT_ROOT="${CKPT_ROOT:-$DEFAULT_CKPT_ROOT}"

mkdir -p "$CKPT_ROOT"

export COMMAND="uv run python examples/run_distillation.py --config ${CONFIG} \
    cluster.num_nodes=2 \
    +data.default.column_mapping.qwen3_1b7_answer=trace \
    distillation.num_prompts_per_step=8 \
    policy.train_global_batch_size=8 \
    policy.generation_batch_size=8 \
    distillation.val_at_start=false \
    distillation.val_at_end=false \
    distillation.val_period=1000 \
    distillation.max_num_steps=5 \
    checkpointing.enabled=false \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/smoke-1p7b-rewrite \
    logger.log_dir=logs/smoke-1p7b-rewrite \
    logger.wandb.name=smoke-1p7b-rewrite"

sbatch \
  --nodes=2 \
  --time=02:00:00 \
  --job-name="smoke-1p7b-rw" \
  "$RAY_SUB"
