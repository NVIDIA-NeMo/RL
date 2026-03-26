#!/bin/bash
# Submit 5 distillation runs on booster with unique logging/checkpoint paths.
# All runs: reverse KL, topk=512, val_at_start=true, booster partition.
#
# Run 0: 1.7B self-distill, prefix prompts (re-run of develbooster job)
# Run 1: 4B teacher → 1.7B student, default prompts (cot.txt)
# Run 2: 4B teacher → 1.7B student, prefix prompts
# Run 3: 4B self-distill, prefix prompts
# Run 4: 8B self-distill, prefix prompts (4 nodes)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="examples/configs/opsd/juelich/distill-1.7b-to-1.7b-2n4g-gen8k.yaml"
CKPT="/p/scratch/scifi/kryeziu1/opsd/checkpoints"

# ── Run 0: 1.7B→1.7B, prefix+teacher-concise prompts, 2 nodes ──────────────
export COMMAND="uv run python examples/run_distillation.py --config ${CONFIG} \
    distillation.val_at_start=true \
    data.default.prompt_file=examples/prompts/prefix.txt \
    data.default.teacher_prompt_file=examples/prompts/teacher-concise.txt \
    checkpointing.checkpoint_dir=${CKPT}/distill-1.7b-self-prefix \
    logger.log_dir=logs/distill-1.7b-self-prefix \
    logger.wandb.name=distill-1.7b-self-prefix"
sbatch --nodes=2 --job-name="d-1.7b-self-pfx" "$SCRIPT_DIR/ray.sub"

# ── Run 1: 4B→1.7B, default prompts (cot.txt, no teacher prompt), 2 nodes ──
export COMMAND="uv run python examples/run_distillation.py --config ${CONFIG} \
    teacher.model_name=Qwen/Qwen3-4B \
    distillation.val_at_start=true \
    checkpointing.checkpoint_dir=${CKPT}/distill-4b-to-1.7b-cot \
    logger.log_dir=logs/distill-4b-to-1.7b-cot \
    logger.wandb.name=distill-4b-to-1.7b-cot"
sbatch --nodes=2 --job-name="d-4b-1.7b-cot" "$SCRIPT_DIR/ray.sub"

# ── Run 2: 4B→1.7B, prefix+teacher-concise prompts, 2 nodes ────────────────
export COMMAND="uv run python examples/run_distillation.py --config ${CONFIG} \
    teacher.model_name=Qwen/Qwen3-4B \
    distillation.val_at_start=true \
    data.default.prompt_file=examples/prompts/prefix.txt \
    data.default.teacher_prompt_file=examples/prompts/teacher-concise.txt \
    checkpointing.checkpoint_dir=${CKPT}/distill-4b-to-1.7b-prefix \
    logger.log_dir=logs/distill-4b-to-1.7b-prefix \
    logger.wandb.name=distill-4b-to-1.7b-prefix"
sbatch --nodes=2 --job-name="d-4b-1.7b-pfx" "$SCRIPT_DIR/ray.sub"

# ── Run 3: 4B→4B, prefix+teacher-concise prompts, 2 nodes ──────────────────
# 4B fits in 2 nodes with TP=4 CP=2 (same as 1.7B config).
# If OOM, bump to 4 nodes and add: cluster.num_nodes=4
export COMMAND="uv run python examples/run_distillation.py --config ${CONFIG} \
    policy.model_name=Qwen/Qwen3-4B \
    teacher.model_name=Qwen/Qwen3-4B \
    distillation.val_at_start=true \
    data.default.prompt_file=examples/prompts/prefix.txt \
    data.default.teacher_prompt_file=examples/prompts/teacher-concise.txt \
    checkpointing.checkpoint_dir=${CKPT}/distill-4b-self-prefix \
    logger.log_dir=logs/distill-4b-self-prefix \
    logger.wandb.name=distill-4b-self-prefix"
sbatch --nodes=2 --job-name="d-4b-self-pfx" "$SCRIPT_DIR/ray.sub"

# ── Run 4: 8B→8B, prefix+teacher-concise prompts, 4 nodes ──────────────────
# 8B needs more headroom; 4 nodes gives DP=2 for faster training.
# vLLM TP=4 required: 8B model (16GB) fills the entire vLLM budget at TP=1
# on 40GB A100s (0.4×40=16GB), leaving no room for KV cache.
export COMMAND="uv run python examples/run_distillation.py --config ${CONFIG} \
    policy.model_name=Qwen/Qwen3-8B \
    teacher.model_name=Qwen/Qwen3-8B \
    cluster.num_nodes=4 \
    policy.generation.vllm_cfg.tensor_parallel_size=4 \
    distillation.val_at_start=true \
    data.default.prompt_file=examples/prompts/prefix.txt \
    data.default.teacher_prompt_file=examples/prompts/teacher-concise.txt \
    checkpointing.checkpoint_dir=${CKPT}/distill-8b-self-prefix \
    logger.log_dir=logs/distill-8b-self-prefix \
    logger.wandb.name=distill-8b-self-prefix"
sbatch --nodes=4 --job-name="d-8b-self-pfx" "$SCRIPT_DIR/ray.sub"

echo "All 5 runs submitted."
