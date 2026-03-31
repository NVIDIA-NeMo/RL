#!/bin/bash
# Submit 8B prompt-variant sweep via JUPITER Ray launcher.
#
# 6 runs, all Qwen3-8B self-distill on 2 nodes:
#   v1: anti-compression prompt (hard length floor)
#   v2: explicit longer reasoning prompt
#   v3: trace as verification only (independent solve)
#   v4: difficulty-adaptive with length floors
#   v5: no trace at all (ablation control)
#   v6: teacher_improve_adaptive.txt with 30% trace truncation

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

# Common overrides for all 8B runs
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

# ── V1: Anti-compression (hard length floor) ────────────────────────────────
submit_job "d-8b-v1-anticompress" 2 "uv run python examples/run_distillation.py --config ${CONFIG} \
    ${BASE_8B} \
    data.default.teacher_prompt_file=examples/prompts/teacher_rewrite_v1_anticompress.txt \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-8b-v1-anticompress \
    logger.log_dir=${LOG_ROOT}/distill-8b-v1-anticompress \
    logger.wandb.name=distill-8b-v1-anticompress"

# ── V2: Explicit longer reasoning ───────────────────────────────────────────
submit_job "d-8b-v2-longer" 2 "uv run python examples/run_distillation.py --config ${CONFIG} \
    ${BASE_8B} \
    data.default.teacher_prompt_file=examples/prompts/teacher_rewrite_v2_longer.txt \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-8b-v2-longer \
    logger.log_dir=${LOG_ROOT}/distill-8b-v2-longer \
    logger.wandb.name=distill-8b-v2-longer"

# ── V3: Trace as verification only (solve independently) ────────────────────
submit_job "d-8b-v3-verify" 2 "uv run python examples/run_distillation.py --config ${CONFIG} \
    ${BASE_8B} \
    data.default.teacher_prompt_file=examples/prompts/teacher_rewrite_v3_verify_only.txt \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-8b-v3-verify \
    logger.log_dir=${LOG_ROOT}/distill-8b-v3-verify \
    logger.wandb.name=distill-8b-v3-verify"

# ── V4: Difficulty-adaptive with length floors ───────────────────────────────
submit_job "d-8b-v4-adaptive" 2 "uv run python examples/run_distillation.py --config ${CONFIG} \
    ${BASE_8B} \
    data.default.teacher_prompt_file=examples/prompts/teacher_rewrite_v4_adaptive_floor.txt \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-8b-v4-adaptive \
    logger.log_dir=${LOG_ROOT}/distill-8b-v4-adaptive \
    logger.wandb.name=distill-8b-v4-adaptive"

# ── V5: No trace (ablation — teacher sees only the problem) ─────────────────
submit_job "d-8b-v5-notrace" 2 "uv run python examples/run_distillation.py --config ${CONFIG} \
    ${BASE_8B} \
    data.default.teacher_prompt_file=examples/prompts/teacher_rewrite_v5_no_trace.txt \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-8b-v5-notrace \
    logger.log_dir=${LOG_ROOT}/distill-8b-v5-notrace \
    logger.wandb.name=distill-8b-v5-notrace"

# ── V6: teacher_improve_adaptive + 30% trace truncation ─────────────────────
submit_job "d-8b-improve-trunc30" 2 "uv run python examples/run_distillation.py --config ${CONFIG} \
    ${BASE_8B} \
    data.default.teacher_prompt_file=examples/prompts/teacher_improve_adaptive.txt \
    +data.default.trace_mode=truncate \
    +data.default.trace_truncate_fraction=0.3 \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-8b-improve-trunc30 \
    logger.log_dir=${LOG_ROOT}/distill-8b-improve-trunc30 \
    logger.wandb.name=distill-8b-improve-trunc30"

echo "All 6 prompt-variant 8B runs submitted through ${RAY_SUB}."
