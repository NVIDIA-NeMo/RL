#!/bin/bash
# Submit 3 Qwen3-8B self-distill runs on Jupiter (2 nodes × 4 GPUs).
#
# TP=2, CP=1, DP=4 — GBS=128, MBS=2, train_mb_tokens=20480
#
#   Run 1: LR 1e-6, warmup 50,  mix10
#   Run 2: LR 1e-6, warmup 50,  mix25
#   Run 3: LR 5e-7, warmup 100, mix10
#
# Student/eval prompt: cot.txt (config default)
# Teacher prompt: teacher_rewrite_v14_independent_check_improved.txt (config default)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="examples/configs/opsd/juelich/distill-8b-2n4g-gen16k-rewrite-bs64.yaml"
CKPT="/e/scratch/scifi/kryeziu1/RL/checkpoints"
LOG_ROOT="/e/scratch/scifi/kryeziu1/RL/logs"
RAY_SUB="${RAY_SUB:-$SCRIPT_DIR/ray.sub}"
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"

NODES=2

if [[ ! -f "$RAY_SUB" ]]; then
  echo "ERROR: ray launcher not found: $RAY_SUB"
  exit 1
fi

mkdir -p "$CKPT" "$LOG_ROOT"

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
  sbatch --nodes="$total_nodes" --time="$SBATCH_TIME" --job-name="$job_name" "$RAY_SUB"
}

BASE_8B=(
  cluster.num_nodes="${NODES}"
  policy.model_name=Qwen/Qwen3-8B
  teacher.model_name=Qwen/Qwen3-8B
  policy.dtensor_cfg.tensor_parallel_size=2
  policy.dtensor_cfg.context_parallel_size=1
  teacher.dtensor_cfg.tensor_parallel_size=2
  teacher.dtensor_cfg.context_parallel_size=1
  policy.generation.vllm_cfg.tensor_parallel_size=2
  policy.generation.vllm_cfg.gpu_memory_utilization=0.5
  policy.generation.vllm_cfg.enforce_eager=False
  policy.train_global_batch_size=128
  policy.generation_batch_size=128
  policy.train_micro_batch_size=2
  policy.logprob_batch_size=2
  policy.dynamic_batching.train_mb_tokens=20480
  policy.dynamic_batching.logprob_mb_tokens=20480
  distillation.num_prompts_per_step=128
  distillation.val_at_start=true
  data.default.column_mapping.qwen3_8b_answer=trace
  'data.validation=[{dataset_name: MATH500, repeat: 1, teacher_prompt_file: null}, {dataset_name: AIME2024, repeat: 4, teacher_prompt_file: null}, {dataset_name: AIME2025, repeat: 4, teacher_prompt_file: null}, {dataset_name: AIME2026, repeat: 4, teacher_prompt_file: null}]'
)

# --- Run 1: LR 1e-6, warmup 50, mix10 ---

submit_job "d-8b-v14-mix10-lr1e6" "${NODES}" \
  "${BASE_8B[@]}" \
  distillation.teacher_student_prefix_fraction=0.10 \
  policy.optimizer.kwargs.lr=1e-6 \
  policy.scheduler.0.kwargs.total_iters=50 \
  'policy.scheduler.2.milestones=[50]' \
  checkpointing.checkpoint_dir="${CKPT}/distill-8b-v14-mix10-lr1e6" \
  logger.log_dir="${LOG_ROOT}/distill-8b-v14-mix10-lr1e6" \
  logger.wandb.name=distill-8b-v14-mix10-lr1e6

# --- Run 2: LR 1e-6, warmup 50, mix25 ---

submit_job "d-8b-v14-mix25-lr1e6" "${NODES}" \
  "${BASE_8B[@]}" \
  distillation.teacher_student_prefix_fraction=0.25 \
  policy.optimizer.kwargs.lr=1e-6 \
  policy.scheduler.0.kwargs.total_iters=50 \
  'policy.scheduler.2.milestones=[50]' \
  checkpointing.checkpoint_dir="${CKPT}/distill-8b-v14-mix25-lr1e6" \
  logger.log_dir="${LOG_ROOT}/distill-8b-v14-mix25-lr1e6" \
  logger.wandb.name=distill-8b-v14-mix25-lr1e6

# --- Run 3: LR 5e-7, warmup 100, mix10 ---

submit_job "d-8b-v14-mix10-lr5e7" "${NODES}" \
  "${BASE_8B[@]}" \
  distillation.teacher_student_prefix_fraction=0.10 \
  policy.optimizer.kwargs.lr=5e-7 \
  policy.scheduler.0.kwargs.total_iters=100 \
  'policy.scheduler.2.milestones=[100]' \
  checkpointing.checkpoint_dir="${CKPT}/distill-8b-v14-mix10-lr5e7" \
  logger.log_dir="${LOG_ROOT}/distill-8b-v14-mix10-lr5e7" \
  logger.wandb.name=distill-8b-v14-mix10-lr5e7

echo "All 3 runs submitted through ${RAY_SUB}."
