#!/bin/bash
# Submit 8B mix sweep via JUPITER Ray launcher (val_temperature=0.6).
#
# 8 runs, all Qwen3-8B self-distill on 2 nodes, train_global_batch_size=64:
#   v14 teacher prompt (mix 25/50/75/95)
#   v1  teacher prompt (mix 25/50/75/95)
#
# Evals: math500, aime2024, aime2025, aime2026 (each avg@4, temperature=0.6)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-examples/configs/opsd/juelich/distill-8b-2n4g-gen16k-rewrite-bs64.yaml}"
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

# --- v14 teacher prompt runs ---
RUN_TAG="distill-8b-v14-independent-check-improved"

for mix in 25 50 75 95; do
  fraction=$(printf "0.%02d" "$mix")
  name="${RUN_TAG}-mix${mix}-bs64"

  submit_job "d-8b-v14-mix${mix}-bs64" 2 \
    "uv run python examples/run_distillation.py --config ${CONFIG} \
    distillation.teacher_student_prefix_fraction=${fraction} \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/${name} \
    logger.log_dir=${LOG_ROOT}/${name} \
    logger.wandb.name=${name}"
done

# --- v1 teacher prompt runs ---
V1_TAG="distill-8b-v1-rewrite"

for mix in 25 50 75 95; do
  fraction=$(printf "0.%02d" "$mix")
  name="${V1_TAG}-mix${mix}-bs64"

  submit_job "d-8b-v1-mix${mix}-bs64" 2 \
    "uv run python examples/run_distillation.py --config ${CONFIG} \
    distillation.teacher_student_prefix_fraction=${fraction} \
    data.default.teacher_prompt_file=examples/prompts/teacher_rewrite.txt \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/${name} \
    logger.log_dir=${LOG_ROOT}/${name} \
    logger.wandb.name=${name}"
done

echo "All 8 runs (4 v14 + 4 v1) submitted through ${RAY_SUB}."
