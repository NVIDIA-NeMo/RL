#!/bin/bash
# Submit 2 Qwen3-8B self-distill runs on 8 nodes (Jülich Booster):
#   v14 teacher prompt, with rewrite warmup over 50 steps
#
#   Run 1: fraction 1.0 → 0.75 (rewrite warms from 0% to 25%)
#   Run 2: fraction 1.0 → 0.50 (rewrite warms from 0% to 50%)
#
# Config from Jupiter prompt sweep (no LR/scheduler overrides — config defaults).
# Cluster/paths adapted for Jülich.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="examples/configs/opsd/juelich/distill-8b-2n4g-gen16k-rewrite-bs64.yaml"
CKPT="/p/scratch/scifi/kryeziu1/opsd/checkpoints"
RAY_SUB="${RAY_SUB:-$SCRIPT_DIR/ray.sub}"

NODES=8
VLLM_UTIL=0.85

if [[ ! -f "$RAY_SUB" ]]; then
  echo "ERROR: ray launcher not found: $RAY_SUB"
  exit 1
fi

mkdir -p "$CKPT" "${REPO_ROOT}/logs"

submit_job() {
  local job_name="$1"
  local total_nodes="$2"
  shift 2

  local cmd=(
    uv run python examples/run_distillation.py
    --config "$CONFIG"
    "$@"
  )

  printf -v COMMAND '%q ' "${cmd[@]}"
  export COMMAND
  sbatch --nodes="$total_nodes" --job-name="$job_name" "$RAY_SUB"
}

BASE_8B=(
  policy.model_name=Qwen/Qwen3-8B
  teacher.model_name=Qwen/Qwen3-8B
  cluster.num_nodes="${NODES}"
  policy.generation.vllm_cfg.gpu_memory_utilization="${VLLM_UTIL}"
  policy.dynamic_batching.train_mb_tokens=20480
  distillation.val_at_start=false
)

V14_PROMPT="examples/prompts/teacher_rewrite_v14_independent_check_improved.txt"
WARMUP_STEPS=50

submit_job "d-8b-v14-mix75-8n-wu50" "${NODES}" \
  "${BASE_8B[@]}" \
  data.default.teacher_prompt_file="${V14_PROMPT}" \
  distillation.teacher_student_prefix_fraction=0.75 \
  distillation.teacher_student_prefix_fraction_start=1.0 \
  distillation.teacher_student_prefix_fraction_warmup_steps="${WARMUP_STEPS}" \
  checkpointing.checkpoint_dir="${CKPT}/distill-8b-v14-mix75-8n-wu50" \
  logger.log_dir=logs/distill-8b-v14-mix75-8n-wu50 \
  logger.wandb.name=distill-8b-v14-mix75-8n-wu50

submit_job "d-8b-v14-mix50-8n-wu50" "${NODES}" \
  "${BASE_8B[@]}" \
  data.default.teacher_prompt_file="${V14_PROMPT}" \
  distillation.teacher_student_prefix_fraction=0.50 \
  distillation.teacher_student_prefix_fraction_start=1.0 \
  distillation.teacher_student_prefix_fraction_warmup_steps="${WARMUP_STEPS}" \
  checkpointing.checkpoint_dir="${CKPT}/distill-8b-v14-mix50-8n-wu50" \
  logger.log_dir=logs/distill-8b-v14-mix50-8n-wu50 \
  logger.wandb.name=distill-8b-v14-mix50-8n-wu50

echo "Both v14 warmup runs (mix75 + mix50, 8 nodes, warmup 50 steps) submitted."
