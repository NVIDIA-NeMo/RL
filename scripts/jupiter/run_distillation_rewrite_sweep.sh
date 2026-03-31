#!/bin/bash
# Submit rewrite-teacher distillation sweeps via JUPITER Ray launcher.
#
# Reduced-node matrix for 4x96GB nodes:
#  - 1.7B self-distill (1 node)
#  - 4B self-distill   (1 node)
#  - 8B self-distill   (2 nodes, vLLM TP=4)
#  - 14B self-distill  (2 nodes, vLLM TP=4)
#  - 32B self-distill  (4 nodes, TP=8)

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

if [[ ! -f "$RAY_SUB" ]]; then
  echo "ERROR: ray launcher not found: $RAY_SUB"
  exit 1
fi

mkdir -p "$CKPT_ROOT"

submit_job() {
  local job_name="$1"
  local nodes="$2"
  local command="$3"
  export COMMAND="$command"
  sbatch --nodes="$nodes" --job-name="$job_name" "$RAY_SUB"
}

submit_job "d-1.7b-rewrite" 1 "uv run python examples/run_distillation.py --config ${CONFIG} \
    cluster.num_nodes=1 \
    +data.default.column_mapping.qwen3_1b7_answer=trace \
    distillation.num_prompts_per_step=32 \
    policy.train_global_batch_size=32 \
    policy.generation_batch_size=32 \
    distillation.val_at_start=true \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-1.7b-self-rewrite \
    logger.log_dir=logs/distill-1.7b-self-rewrite \
    logger.wandb.name=distill-1.7b-self-rewrite"

submit_job "d-4b-rewrite" 1 "uv run python examples/run_distillation.py --config ${CONFIG} \
    cluster.num_nodes=1 \
    policy.model_name=Qwen/Qwen3-4B \
    teacher.model_name=Qwen/Qwen3-4B \
    +data.default.column_mapping.qwen3_4b_answer=trace \
    distillation.num_prompts_per_step=32 \
    policy.train_global_batch_size=32 \
    policy.generation_batch_size=32 \
    distillation.val_at_start=true \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-4b-self-rewrite \
    logger.log_dir=logs/distill-4b-self-rewrite \
    logger.wandb.name=distill-4b-self-rewrite"

submit_job "d-8b-rewrite" 2 "uv run python examples/run_distillation.py --config ${CONFIG} \
    cluster.num_nodes=2 \
    policy.model_name=Qwen/Qwen3-8B \
    teacher.model_name=Qwen/Qwen3-8B \
    policy.generation.vllm_cfg.tensor_parallel_size=4 \
    +data.default.column_mapping.qwen3_8b_answer=trace \
    distillation.num_prompts_per_step=32 \
    policy.train_global_batch_size=32 \
    policy.generation_batch_size=32 \
    distillation.val_at_start=true \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-8b-self-rewrite \
    logger.log_dir=logs/distill-8b-self-rewrite \
    logger.wandb.name=distill-8b-self-rewrite"

submit_job "d-14b-rewrite" 2 "uv run python examples/run_distillation.py --config ${CONFIG} \
    cluster.num_nodes=2 \
    policy.model_name=Qwen/Qwen3-14B \
    teacher.model_name=Qwen/Qwen3-14B \
    policy.generation.vllm_cfg.tensor_parallel_size=4 \
    +data.default.column_mapping.qwen3_14b_answer=trace \
    distillation.num_prompts_per_step=32 \
    policy.train_global_batch_size=32 \
    policy.generation_batch_size=32 \
    distillation.val_at_start=true \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-14b-self-rewrite \
    logger.log_dir=logs/distill-14b-self-rewrite \
    logger.wandb.name=distill-14b-self-rewrite"

submit_job "d-32b-rewrite" 4 "uv run python examples/run_distillation.py --config ${CONFIG} \
    policy.model_name=Qwen/Qwen3-32B \
    teacher.model_name=Qwen/Qwen3-32B \
    policy.dtensor_cfg.tensor_parallel_size=8 \
    teacher.dtensor_cfg.tensor_parallel_size=8 \
    policy.generation.vllm_cfg.tensor_parallel_size=8 \
    cluster.num_nodes=4 \
    +data.default.column_mapping.qwen3_32b_answer=trace \
    distillation.num_prompts_per_step=32 \
    policy.train_global_batch_size=32 \
    policy.generation_batch_size=32 \
    distillation.val_at_start=true \
    checkpointing.checkpoint_dir=${CKPT_ROOT}/distill-32b-self-rewrite \
    logger.log_dir=logs/distill-32b-self-rewrite \
    logger.wandb.name=distill-32b-self-rewrite"

echo "All 5 rewrite runs submitted through ${RAY_SUB}."
