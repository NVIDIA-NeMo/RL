#!/bin/bash
# Submit 12 Qwen3-8B self-distill runs: rewrite prompt × mix × LR/scheduler sweep.
#
# Two prompt variants:
#   v14  = teacher_rewrite_v14_independent_check_improved.txt
#   rw   = teacher_rewrite.txt
#
# Two mix fractions: 0.50, 0.75
#
# Three LR/scheduler fleets:
#   Fleet 1 (wu25):  lr=5e-6, warmup 25 steps, constant after
#   Fleet 2 (lr1e6): lr=1e-6, warmup 10 steps, constant after
#   Fleet 3 (cos):   lr=3e-6, warmup 80 steps, cosine decay over 920 steps
#
# Cluster: Jülich Booster — 4 nodes × 4 A100 40GB (colocated, gpu_mem_util=0.85)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="examples/configs/opsd/juelich/distill-8b-2n4g-gen16k-rewrite-bs64.yaml"
CKPT="/p/scratch/scifi/kryeziu1/opsd/checkpoints"
RAY_SUB="${RAY_SUB:-$SCRIPT_DIR/ray.sub}"
QWEN3_TP_PLAN="${QWEN3_TP_PLAN:-examples.custom_parallel.custom_parallel.qwen_model_tp_plan_stable}"

NODES="${NODES:-4}"
VLLM_UTIL="${VLLM_UTIL:-0.85}"

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

# -- Shared overrides for all 12 runs ----------------------------------------
BASE_8B=(
  policy.model_name=Qwen/Qwen3-8B
  teacher.model_name=Qwen/Qwen3-8B
  cluster.num_nodes="${NODES}"
  policy.dtensor_cfg.custom_parallel_plan="${QWEN3_TP_PLAN}"
  teacher.dtensor_cfg.custom_parallel_plan="${QWEN3_TP_PLAN}"
  policy.generation.vllm_cfg.gpu_memory_utilization="${VLLM_UTIL}"
)

V14_PROMPT="examples/prompts/teacher_rewrite_v14_independent_check_improved.txt"
RW_PROMPT="examples/prompts/teacher_rewrite.txt"

# -- Fleet 1: longer warmup (25 steps, lr=5e-6, linear+constant) -------------
FLEET1_SCHED=(
  policy.scheduler.0.kwargs.total_iters=25
  'policy.scheduler.2.milestones=[25]'
)

submit_job "d-8b-v14-mix50-wu25" "${NODES}" \
  "${BASE_8B[@]}" "${FLEET1_SCHED[@]}" \
  data.default.teacher_prompt_file="${V14_PROMPT}" \
  distillation.teacher_student_prefix_fraction=0.50 \
  checkpointing.checkpoint_dir="${CKPT}/distill-8b-v14-mix50-wu25" \
  logger.log_dir=logs/distill-8b-v14-mix50-wu25 \
  logger.wandb.name=distill-8b-v14-mix50-wu25

submit_job "d-8b-v14-mix75-wu25" "${NODES}" \
  "${BASE_8B[@]}" "${FLEET1_SCHED[@]}" \
  data.default.teacher_prompt_file="${V14_PROMPT}" \
  distillation.teacher_student_prefix_fraction=0.75 \
  checkpointing.checkpoint_dir="${CKPT}/distill-8b-v14-mix75-wu25" \
  logger.log_dir=logs/distill-8b-v14-mix75-wu25 \
  logger.wandb.name=distill-8b-v14-mix75-wu25

submit_job "d-8b-rw-mix50-wu25" "${NODES}" \
  "${BASE_8B[@]}" "${FLEET1_SCHED[@]}" \
  data.default.teacher_prompt_file="${RW_PROMPT}" \
  distillation.teacher_student_prefix_fraction=0.50 \
  checkpointing.checkpoint_dir="${CKPT}/distill-8b-rw-mix50-wu25" \
  logger.log_dir=logs/distill-8b-rw-mix50-wu25 \
  logger.wandb.name=distill-8b-rw-mix50-wu25

submit_job "d-8b-rw-mix75-wu25" "${NODES}" \
  "${BASE_8B[@]}" "${FLEET1_SCHED[@]}" \
  data.default.teacher_prompt_file="${RW_PROMPT}" \
  distillation.teacher_student_prefix_fraction=0.75 \
  checkpointing.checkpoint_dir="${CKPT}/distill-8b-rw-mix75-wu25" \
  logger.log_dir=logs/distill-8b-rw-mix75-wu25 \
  logger.wandb.name=distill-8b-rw-mix75-wu25

# -- Fleet 2: lower peak LR (lr=1e-6, warmup 10 steps, linear+constant) ------
FLEET2_SCHED=(
  policy.optimizer.kwargs.lr=1e-6
)

submit_job "d-8b-v14-mix50-lr1e6" "${NODES}" \
  "${BASE_8B[@]}" "${FLEET2_SCHED[@]}" \
  data.default.teacher_prompt_file="${V14_PROMPT}" \
  distillation.teacher_student_prefix_fraction=0.50 \
  checkpointing.checkpoint_dir="${CKPT}/distill-8b-v14-mix50-lr1e6" \
  logger.log_dir=logs/distill-8b-v14-mix50-lr1e6 \
  logger.wandb.name=distill-8b-v14-mix50-lr1e6

submit_job "d-8b-v14-mix75-lr1e6" "${NODES}" \
  "${BASE_8B[@]}" "${FLEET2_SCHED[@]}" \
  data.default.teacher_prompt_file="${V14_PROMPT}" \
  distillation.teacher_student_prefix_fraction=0.75 \
  checkpointing.checkpoint_dir="${CKPT}/distill-8b-v14-mix75-lr1e6" \
  logger.log_dir=logs/distill-8b-v14-mix75-lr1e6 \
  logger.wandb.name=distill-8b-v14-mix75-lr1e6

submit_job "d-8b-rw-mix50-lr1e6" "${NODES}" \
  "${BASE_8B[@]}" "${FLEET2_SCHED[@]}" \
  data.default.teacher_prompt_file="${RW_PROMPT}" \
  distillation.teacher_student_prefix_fraction=0.50 \
  checkpointing.checkpoint_dir="${CKPT}/distill-8b-rw-mix50-lr1e6" \
  logger.log_dir=logs/distill-8b-rw-mix50-lr1e6 \
  logger.wandb.name=distill-8b-rw-mix50-lr1e6

submit_job "d-8b-rw-mix75-lr1e6" "${NODES}" \
  "${BASE_8B[@]}" "${FLEET2_SCHED[@]}" \
  data.default.teacher_prompt_file="${RW_PROMPT}" \
  distillation.teacher_student_prefix_fraction=0.75 \
  checkpointing.checkpoint_dir="${CKPT}/distill-8b-rw-mix75-lr1e6" \
  logger.log_dir=logs/distill-8b-rw-mix75-lr1e6 \
  logger.wandb.name=distill-8b-rw-mix75-lr1e6

# -- Fleet 3: cosine (lr=3e-6, warmup 80 steps, cosine decay 920 steps) ------
FLEET3_SCHED=(
  policy.optimizer.kwargs.lr=3e-6
  'policy.scheduler=[{name: "torch.optim.lr_scheduler.LinearLR", kwargs: {start_factor: 0.1, end_factor: 1.0, total_iters: 80}}, {name: "torch.optim.lr_scheduler.CosineAnnealingLR", kwargs: {T_max: 920, eta_min: 0}}, {milestones: [80]}]'
)

submit_job "d-8b-v14-mix50-cos" "${NODES}" \
  "${BASE_8B[@]}" "${FLEET3_SCHED[@]}" \
  data.default.teacher_prompt_file="${V14_PROMPT}" \
  distillation.teacher_student_prefix_fraction=0.50 \
  checkpointing.checkpoint_dir="${CKPT}/distill-8b-v14-mix50-cos" \
  logger.log_dir=logs/distill-8b-v14-mix50-cos \
  logger.wandb.name=distill-8b-v14-mix50-cos

submit_job "d-8b-v14-mix75-cos" "${NODES}" \
  "${BASE_8B[@]}" "${FLEET3_SCHED[@]}" \
  data.default.teacher_prompt_file="${V14_PROMPT}" \
  distillation.teacher_student_prefix_fraction=0.75 \
  checkpointing.checkpoint_dir="${CKPT}/distill-8b-v14-mix75-cos" \
  logger.log_dir=logs/distill-8b-v14-mix75-cos \
  logger.wandb.name=distill-8b-v14-mix75-cos

submit_job "d-8b-rw-mix50-cos" "${NODES}" \
  "${BASE_8B[@]}" "${FLEET3_SCHED[@]}" \
  data.default.teacher_prompt_file="${RW_PROMPT}" \
  distillation.teacher_student_prefix_fraction=0.50 \
  checkpointing.checkpoint_dir="${CKPT}/distill-8b-rw-mix50-cos" \
  logger.log_dir=logs/distill-8b-rw-mix50-cos \
  logger.wandb.name=distill-8b-rw-mix50-cos

submit_job "d-8b-rw-mix75-cos" "${NODES}" \
  "${BASE_8B[@]}" "${FLEET3_SCHED[@]}" \
  data.default.teacher_prompt_file="${RW_PROMPT}" \
  distillation.teacher_student_prefix_fraction=0.75 \
  checkpointing.checkpoint_dir="${CKPT}/distill-8b-rw-mix75-cos" \
  logger.log_dir=logs/distill-8b-rw-mix75-cos \
  logger.wandb.name=distill-8b-rw-mix75-cos

echo "All 12 runs submitted (4 variants × 3 LR/scheduler fleets)."
