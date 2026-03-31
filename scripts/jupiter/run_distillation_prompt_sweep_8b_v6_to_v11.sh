#!/bin/bash
# Submit 8B prompt-variant sweep via JUPITER Ray launcher.
#
# 6 runs, all Qwen3-8B self-distill on 2 nodes:
#   v6: problem-only calibrated solver
#   v7: skeptical auditor of candidate trace
#   v8: independent-first, compare-later
#   v9: trace as pacing/pitfall signal only
#   v10: subgoal and invariant planner
#   v11: epistemic trace densifier

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
  sbatch --nodes="$nodes" --job-name="$job_name" "$RAY_SUB"
}

# Common overrides for all 8B runs.
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
    distillation.val_period=25"

submit_job "d-8b-v6-problem" 2 "uv run python examples/run_distillation.py --config ${CONFIG} \
    ${BASE_8B} \
    data.default.teacher_prompt_file=examples/prompts/teacher_rewrite_v6_problem_only_calibrated_solver.txt \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-8b-v6-problem-only \
    logger.log_dir=${LOG_ROOT}/distill-8b-v6-problem-only \
    logger.wandb.name=distill-8b-v6-problem-only"

submit_job "d-8b-v7-auditor" 2 "uv run python examples/run_distillation.py --config ${CONFIG} \
    ${BASE_8B} \
    data.default.teacher_prompt_file=examples/prompts/teacher_rewrite_v7_skeptical_auditor.txt \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-8b-v7-skeptical-auditor \
    logger.log_dir=${LOG_ROOT}/distill-8b-v7-skeptical-auditor \
    logger.wandb.name=distill-8b-v7-skeptical-auditor"

submit_job "d-8b-v8-compare" 2 "uv run python examples/run_distillation.py --config ${CONFIG} \
    ${BASE_8B} \
    data.default.teacher_prompt_file=examples/prompts/teacher_rewrite_v8_independent_first_compare_later.txt \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-8b-v8-independent-first-compare-later \
    logger.log_dir=${LOG_ROOT}/distill-8b-v8-independent-first-compare-later \
    logger.wandb.name=distill-8b-v8-independent-first-compare-later"

submit_job "d-8b-v9-pacing" 2 "uv run python examples/run_distillation.py --config ${CONFIG} \
    ${BASE_8B} \
    data.default.teacher_prompt_file=examples/prompts/teacher_rewrite_v9_trace_as_pacing_signal.txt \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-8b-v9-trace-as-pacing-signal \
    logger.log_dir=${LOG_ROOT}/distill-8b-v9-trace-as-pacing-signal \
    logger.wandb.name=distill-8b-v9-trace-as-pacing-signal"

submit_job "d-8b-v10-subgoal" 2 "uv run python examples/run_distillation.py --config ${CONFIG} \
    ${BASE_8B} \
    data.default.teacher_prompt_file=examples/prompts/teacher_rewrite_v10_subgoal_invariant_planner.txt \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-8b-v10-subgoal-invariant-planner \
    logger.log_dir=${LOG_ROOT}/distill-8b-v10-subgoal-invariant-planner \
    logger.wandb.name=distill-8b-v10-subgoal-invariant-planner"

submit_job "d-8b-v11-epistemic" 2 "uv run python examples/run_distillation.py --config ${CONFIG} \
    ${BASE_8B} \
    data.default.teacher_prompt_file=examples/prompts/teacher_rewrite_v11_epistemic_trace_densifier.txt \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-8b-v11-epistemic-trace-densifier \
    logger.log_dir=${LOG_ROOT}/distill-8b-v11-epistemic-trace-densifier \
    logger.wandb.name=distill-8b-v11-epistemic-trace-densifier"

echo "All 6 prompt-variant 8B runs (v6-v11) submitted through ${RAY_SUB}."
