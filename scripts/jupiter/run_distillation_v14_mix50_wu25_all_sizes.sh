#!/bin/bash
# Submit v14-mix50-wu25 self-distillation for Qwen3 1.7B–32B on Jupiter.
#
# Training dynamics (matching 8B Fleet 1 wu25 run):
#   LR = 5e-6, warmup 25 steps (linear → constant)
#   Teacher prompt: teacher_rewrite_v14_independent_check_improved.txt
#   Mix: teacher_student_prefix_fraction = 0.50
#   GBS = 64, topk = 512, reverse KL, 1000 steps
#
# Cluster: Jupiter — 4×GH200 96GB VRAM per node
#
#   Model  Nodes  GPUs  dtensor TP/CP  vLLM TP  DP
#   1.7B   1      4     1 / 1          1        4
#   4B     1      4     1 / 1          1        4
#   8B     2      8     2 / 1          2        4
#   14B    2      8     2 / 1          2        4
#   32B    8      32    4 / 1          4        8
#
# Evals: MATH500 (×1), AIME2024 (×4), AIME2025 (×4), AIME2026 (×4)
# Validation at start enabled.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="examples/configs/opsd/juelich/distill-8b-2n4g-gen16k-rewrite-bs64.yaml"
CKPT="${CKPT:-/e/scratch/scifi/kryeziu1/RL/checkpoints}"
LOG_ROOT="${LOG_ROOT:-/e/scratch/scifi/kryeziu1/RL/logs}"
RAY_SUB="${RAY_SUB:-$SCRIPT_DIR/ray.sub}"
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"
QWEN3_TP_PLAN="${QWEN3_TP_PLAN:-examples.custom_parallel.custom_parallel.qwen_model_tp_plan_stable}"

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

V14_PROMPT="examples/prompts/teacher_rewrite_v14_independent_check_improved.txt"

SHARED=(
  data.default.teacher_prompt_file="${V14_PROMPT}"
  distillation.teacher_student_prefix_fraction=0.50
  distillation.val_at_start=true
  policy.scheduler.0.kwargs.total_iters=25
  'policy.scheduler.2.milestones=[25]'
)

# ── 1.7B ── 1 node, TP=1, CP=1, vLLM TP=1, DP=4 ────────────────────────────
NAME="d-1.7b-v14-mix50-wu25"
submit_job "${NAME}" 1 \
  cluster.num_nodes=1 \
  policy.model_name=Qwen/Qwen3-1.7B \
  teacher.model_name=Qwen/Qwen3-1.7B \
  policy.dtensor_cfg.tensor_parallel_size=1 \
  policy.dtensor_cfg.context_parallel_size=1 \
  teacher.dtensor_cfg.tensor_parallel_size=1 \
  teacher.dtensor_cfg.context_parallel_size=1 \
  policy.dtensor_cfg.custom_parallel_plan="${QWEN3_TP_PLAN}" \
  teacher.dtensor_cfg.custom_parallel_plan="${QWEN3_TP_PLAN}" \
  policy.generation.vllm_cfg.tensor_parallel_size=1 \
  policy.generation.vllm_cfg.gpu_memory_utilization=0.4 \
  '~data.default.column_mapping.qwen3_8b_answer' \
  '++data.default.column_mapping.qwen3_1b7_answer=trace' \
  "${SHARED[@]}" \
  checkpointing.checkpoint_dir="${CKPT}/${NAME}" \
  logger.log_dir="${LOG_ROOT}/${NAME}" \
  logger.wandb.name="${NAME}"

# ── 4B ── 1 node, TP=1, CP=1, vLLM TP=1, DP=4 ──────────────────────────────
NAME="d-4b-v14-mix50-wu25"
submit_job "${NAME}" 1 \
  cluster.num_nodes=1 \
  policy.model_name=Qwen/Qwen3-4B \
  teacher.model_name=Qwen/Qwen3-4B \
  policy.dtensor_cfg.tensor_parallel_size=1 \
  policy.dtensor_cfg.context_parallel_size=1 \
  teacher.dtensor_cfg.tensor_parallel_size=1 \
  teacher.dtensor_cfg.context_parallel_size=1 \
  policy.dtensor_cfg.custom_parallel_plan="${QWEN3_TP_PLAN}" \
  teacher.dtensor_cfg.custom_parallel_plan="${QWEN3_TP_PLAN}" \
  policy.generation.vllm_cfg.tensor_parallel_size=1 \
  policy.generation.vllm_cfg.gpu_memory_utilization=0.5 \
  '~data.default.column_mapping.qwen3_8b_answer' \
  '++data.default.column_mapping.qwen3_4b_answer=trace' \
  "${SHARED[@]}" \
  checkpointing.checkpoint_dir="${CKPT}/${NAME}" \
  logger.log_dir="${LOG_ROOT}/${NAME}" \
  logger.wandb.name="${NAME}"

# ── 8B ── 2 nodes, TP=2, CP=1, vLLM TP=2, DP=4 ─────────────────────────────
NAME="d-8b-v14-mix50-wu25"
submit_job "${NAME}" 2 \
  cluster.num_nodes=2 \
  policy.model_name=Qwen/Qwen3-8B \
  teacher.model_name=Qwen/Qwen3-8B \
  policy.dtensor_cfg.tensor_parallel_size=2 \
  policy.dtensor_cfg.context_parallel_size=1 \
  teacher.dtensor_cfg.tensor_parallel_size=2 \
  teacher.dtensor_cfg.context_parallel_size=1 \
  policy.dtensor_cfg.custom_parallel_plan="${QWEN3_TP_PLAN}" \
  teacher.dtensor_cfg.custom_parallel_plan="${QWEN3_TP_PLAN}" \
  policy.generation.vllm_cfg.tensor_parallel_size=2 \
  policy.generation.vllm_cfg.gpu_memory_utilization=0.5 \
  data.default.column_mapping.qwen3_8b_answer=trace \
  "${SHARED[@]}" \
  checkpointing.checkpoint_dir="${CKPT}/${NAME}" \
  logger.log_dir="${LOG_ROOT}/${NAME}" \
  logger.wandb.name="${NAME}"

# ── 14B ── 2 nodes, TP=2, CP=1, vLLM TP=2, DP=4 ────────────────────────────
NAME="d-14b-v14-mix50-wu25"
submit_job "${NAME}" 2 \
  cluster.num_nodes=2 \
  policy.model_name=Qwen/Qwen3-14B \
  teacher.model_name=Qwen/Qwen3-14B \
  policy.dtensor_cfg.tensor_parallel_size=2 \
  policy.dtensor_cfg.context_parallel_size=1 \
  teacher.dtensor_cfg.tensor_parallel_size=2 \
  teacher.dtensor_cfg.context_parallel_size=1 \
  policy.dtensor_cfg.custom_parallel_plan="${QWEN3_TP_PLAN}" \
  teacher.dtensor_cfg.custom_parallel_plan="${QWEN3_TP_PLAN}" \
  policy.generation.vllm_cfg.tensor_parallel_size=2 \
  policy.generation.vllm_cfg.gpu_memory_utilization=0.4 \
  policy.train_micro_batch_size=1 \
  policy.logprob_batch_size=1 \
  policy.dynamic_batching.train_mb_tokens=20480 \
  policy.dynamic_batching.logprob_mb_tokens=20480 \
  '~data.default.column_mapping.qwen3_8b_answer' \
  '++data.default.column_mapping.qwen3_14b_answer=trace' \
  "${SHARED[@]}" \
  checkpointing.checkpoint_dir="${CKPT}/${NAME}" \
  logger.log_dir="${LOG_ROOT}/${NAME}" \
  logger.wandb.name="${NAME}"

# ── 32B ── 8 nodes, TP=4, CP=1, vLLM TP=4, DP=8 ────────────────────────────
NAME="d-32b-v14-mix50-wu25"
submit_job "${NAME}" 8 \
  cluster.num_nodes=8 \
  policy.model_name=Qwen/Qwen3-32B \
  teacher.model_name=Qwen/Qwen3-32B \
  policy.dtensor_cfg.tensor_parallel_size=4 \
  policy.dtensor_cfg.context_parallel_size=1 \
  teacher.dtensor_cfg.tensor_parallel_size=4 \
  teacher.dtensor_cfg.context_parallel_size=1 \
  policy.dtensor_cfg.custom_parallel_plan="${QWEN3_TP_PLAN}" \
  teacher.dtensor_cfg.custom_parallel_plan="${QWEN3_TP_PLAN}" \
  policy.generation.vllm_cfg.tensor_parallel_size=4 \
  policy.generation.vllm_cfg.gpu_memory_utilization=0.5 \
  '~data.default.column_mapping.qwen3_8b_answer' \
  '++data.default.column_mapping.qwen3_14b_answer=trace' \
  policy.train_micro_batch_size=1 \
  policy.logprob_batch_size=1 \
  policy.dynamic_batching.train_mb_tokens=20480 \
  policy.dynamic_batching.logprob_mb_tokens=20480 \
  "${SHARED[@]}" \
  checkpointing.checkpoint_dir="${CKPT}/${NAME}" \
  logger.log_dir="${LOG_ROOT}/${NAME}" \
  logger.wandb.name="${NAME}"

echo "All 5 model-size runs submitted (1.7B, 4B, 8B, 14B, 32B)."
