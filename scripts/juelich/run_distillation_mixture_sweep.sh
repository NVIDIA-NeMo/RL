#!/bin/bash
# Submit 8B v14 teacher-context mix sweep via JUPITER Ray launcher.
#
# 4 runs, all Qwen3-8B self-distill on 2 nodes:
#   v14-mix00: 0% of teacher-scored samples reuse the student prefix
#   v14-mix10: 10% of teacher-scored samples reuse the student prefix
#   v14-mix25: 25% of teacher-scored samples reuse the student prefix
#   v14-mix50: 50% of teacher-scored samples reuse the student prefix

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-examples/configs/opsd/juelich/distill-1.7b-to-1.7b-4n4g-gen16k-rewrite.yaml}"
RAY_SUB="${RAY_SUB:-$SCRIPT_DIR/ray.sub}"
SBATCH_TIME="${SBATCH_TIME:-10:00:00}"

if [[ -d "/e/scratch/scifi/yll/checkpoints" ]]; then
  DEFAULT_CKPT_ROOT="/e/scratch/scifi/yll/checkpoints"
else
  DEFAULT_CKPT_ROOT="${REPO_ROOT}/checkpoints"
fi
CKPT_ROOT="${CKPT_ROOT:-$DEFAULT_CKPT_ROOT}"
LOG_ROOT="${LOG_ROOT:-/e/scratch/scifi/kryeziu1/RL/logs}"

if [[ ! -f "$RAY_SUB" ]]; then
  echo "ERROR: ray launcher not found: $RAY_SUB"
  exit 1
fi

mkdir -p "$CKPT_ROOT" "$LOG_ROOT"

submit_job() {
  local job_name="$1"
  local nodes="$2"
  local command="$3"
  export COMMAND="$command"
  sbatch --nodes="$nodes" --time="$SBATCH_TIME" --job-name="$job_name" "$RAY_SUB"
}

BASE_8B="\
    cluster.num_nodes=2 \
    policy.model_name=Qwen/Qwen3-8B \
    teacher.model_name=Qwen/Qwen3-8B \
    policy.generation.vllm_cfg.tensor_parallel_size=4 \
    +data.default.column_mapping.qwen3_8b_answer=trace \
    distillation.num_prompts_per_step=32 \
    policy.train_global_batch_size=32 \
    policy.generation_batch_size=32 \
    distillation.val_at_start=true \
    'distillation.val_steps=[10]' \
    distillation.val_period=25 \
    data.default.teacher_prompt_file=examples/prompts/teacher_rewrite_v14_independent_check_improved.txt"

submit_job "d-8b-v14-mix00" 2 "uv run python examples/run_distillation.py --config ${CONFIG} \
    ${BASE_8B} \
    distillation.teacher_student_prefix_fraction=0.00 \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-8b-v14-independent-check-improved-mix00 \
    logger.log_dir=${LOG_ROOT}/distill-8b-v14-independent-check-improved-mix00 \
    logger.wandb.name=distill-8b-v14-independent-check-improved-mix00"

submit_job "d-8b-v14-mix10" 2 "uv run python examples/run_distillation.py --config ${CONFIG} \
    ${BASE_8B} \
    distillation.teacher_student_prefix_fraction=0.10 \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-8b-v14-independent-check-improved-mix10 \
    logger.log_dir=${LOG_ROOT}/distill-8b-v14-independent-check-improved-mix10 \
    logger.wandb.name=distill-8b-v14-independent-check-improved-mix10"

submit_job "d-8b-v14-mix25" 2 "uv run python examples/run_distillation.py --config ${CONFIG} \
    ${BASE_8B} \
    distillation.teacher_student_prefix_fraction=0.25 \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-8b-v14-independent-check-improved-mix25 \
    logger.log_dir=${LOG_ROOT}/distill-8b-v14-independent-check-improved-mix25 \
    logger.wandb.name=distill-8b-v14-independent-check-improved-mix25"

submit_job "d-8b-v14-mix50" 2 "uv run python examples/run_distillation.py --config ${CONFIG} \
    ${BASE_8B} \
    distillation.teacher_student_prefix_fraction=0.50 \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-8b-v14-independent-check-improved-mix50 \
    logger.log_dir=${LOG_ROOT}/distill-8b-v14-independent-check-improved-mix50 \
    logger.wandb.name=distill-8b-v14-independent-check-improved-mix50"

echo "All 4 v14 teacher-context mix runs submitted through ${RAY_SUB}."
