#!/bin/bash
# Submit 5 self-distillation runs (be-concise prompts) on Juelich Booster (4xA100 40GB).
# Config: distill-be-concise-self-bs64.yaml - student-concise (train), cot (eval), teacher-concise.
#
# Models: 1.7B, 4B, 8B, 14B, 32B (same teacher/student per run).
# Large-model notes:
# - 8B/14B/32B use colocated vLLM so ALL nodes participate in both training and
#   inference. gpu_memory_utilization is tuned per model to avoid OOM during refit.
#   cpu_offload + activation_checkpointing + sleep_mode keep memory safe.
# - 14B/32B disable dynamic batching so train/logprob microbatch overrides actually
#   take effect. With dynamic batching enabled, train_micro_batch_size=1 is ignored.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="examples/configs/opsd/juelich/distill-be-concise-self-bs64.yaml"
CKPT="/p/scratch/scifi/kryeziu1/opsd/checkpoints"
RAY_SUB="${RAY_SUB:-$SCRIPT_DIR/ray.sub}"
QWEN3_TP_PLAN="${QWEN3_TP_PLAN:-examples.custom_parallel.custom_parallel.qwen_model_tp_plan_stable}"

# Colocated mode: all nodes do both training and inference. Tuned gpu_memory_utilization
# per model size to leave room for refit buffer (~25 GB peak for 32B).
BC_8B_NODES="${BC_8B_NODES:-16}"
BC_14B_NODES="${BC_14B_NODES:-16}"
BC_32B_NODES="${BC_32B_NODES:-16}"

BC_8B_VLLM_UTIL="${BC_8B_VLLM_UTIL:-0.80}"
BC_14B_VLLM_UTIL="${BC_14B_VLLM_UTIL:-0.75}"
BC_32B_VLLM_UTIL="${BC_32B_VLLM_UTIL:-0.75}"

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

# -- 1.7B self-distill, 4 colocated nodes -------------------------------------
submit_job "d-bc-1.7b" 4 \
  policy.model_name=Qwen/Qwen3-1.7B \
  teacher.model_name=Qwen/Qwen3-1.7B \
  cluster.num_nodes=4 \
  checkpointing.checkpoint_dir="${CKPT}/distill-be-concise-1.7b-self" \
  logger.log_dir=logs/distill-be-concise-1.7b-self \
  logger.wandb.name=distill-be-concise-1.7b-self

# -- 4B self-distill, 4 colocated nodes ---------------------------------------
submit_job "d-bc-4b" 4 \
  policy.model_name=Qwen/Qwen3-4B \
  teacher.model_name=Qwen/Qwen3-4B \
  cluster.num_nodes=4 \
  checkpointing.checkpoint_dir="${CKPT}/distill-be-concise-4b-self" \
  logger.log_dir=logs/distill-be-concise-4b-self \
  logger.wandb.name=distill-be-concise-4b-self

# -- 8B self-distill, colocated on all nodes ----------------------------------
submit_job "d-bc-8b" "${BC_8B_NODES}" \
  policy.model_name=Qwen/Qwen3-8B \
  teacher.model_name=Qwen/Qwen3-8B \
  cluster.num_nodes="${BC_8B_NODES}" \
  policy.dtensor_cfg.custom_parallel_plan="${QWEN3_TP_PLAN}" \
  teacher.dtensor_cfg.custom_parallel_plan="${QWEN3_TP_PLAN}" \
  policy.generation.vllm_cfg.gpu_memory_utilization="${BC_8B_VLLM_UTIL}" \
  checkpointing.checkpoint_dir="${CKPT}/distill-be-concise-8b-self" \
  logger.log_dir=logs/distill-be-concise-8b-self \
  logger.wandb.name=distill-be-concise-8b-self

# -- 14B self-distill, colocated on all nodes ---------------------------------
submit_job "d-bc-14b" "${BC_14B_NODES}" \
  policy.model_name=Qwen/Qwen3-14B \
  teacher.model_name=Qwen/Qwen3-14B \
  cluster.num_nodes="${BC_14B_NODES}" \
  policy.dtensor_cfg.custom_parallel_plan="${QWEN3_TP_PLAN}" \
  teacher.dtensor_cfg.custom_parallel_plan="${QWEN3_TP_PLAN}" \
  policy.dynamic_batching.enabled=false \
  teacher.dynamic_batching.enabled=false \
  policy.train_micro_batch_size=1 \
  policy.logprob_batch_size=1 \
  teacher.train_micro_batch_size=1 \
  teacher.logprob_batch_size=1 \
  policy.generation.vllm_cfg.gpu_memory_utilization="${BC_14B_VLLM_UTIL}" \
  checkpointing.checkpoint_dir="${CKPT}/distill-be-concise-14b-self" \
  logger.log_dir=logs/distill-be-concise-14b-self \
  logger.wandb.name=distill-be-concise-14b-self

# -- 32B self-distill, colocated on all nodes ---------------------------------
submit_job "d-bc-32b" "${BC_32B_NODES}" \
  policy.model_name=Qwen/Qwen3-32B \
  teacher.model_name=Qwen/Qwen3-32B \
  cluster.num_nodes="${BC_32B_NODES}" \
  policy.dtensor_cfg.custom_parallel_plan="${QWEN3_TP_PLAN}" \
  teacher.dtensor_cfg.custom_parallel_plan="${QWEN3_TP_PLAN}" \
  policy.dynamic_batching.enabled=false \
  teacher.dynamic_batching.enabled=false \
  policy.train_micro_batch_size=1 \
  policy.logprob_batch_size=1 \
  teacher.train_micro_batch_size=1 \
  teacher.logprob_batch_size=1 \
  policy.generation.vllm_cfg.gpu_memory_utilization="${BC_32B_VLLM_UTIL}" \
  checkpointing.checkpoint_dir="${CKPT}/distill-be-concise-32b-self" \
  logger.log_dir=logs/distill-be-concise-32b-self \
  logger.wandb.name=distill-be-concise-32b-self

echo "All 5 be-concise self-distillation runs submitted."
