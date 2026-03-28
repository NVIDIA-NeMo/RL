#!/bin/bash
# Submit 5 sampled token distillation runs on booster.
# All runs: kl_penalty_coef=1.0, val_at_start=true, booster partition.
#
# Run 0: 1.7B self-distill, default prompts (cot.txt)
# Run 1: 4B teacher → 1.7B student, default prompts (cot.txt)
# Run 2: 4B teacher → 1.7B student, prefix + teacher-concise prompts
# Run 3: 4B self-distill, prefix + teacher-concise prompts
# Run 4: 8B self-distill, prefix + teacher-concise prompts (4 nodes)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="examples/configs/opsd/juelich/std-1.7b-to-1.7b-2n4g-gen8k.yaml"
CKPT="/p/scratch/scifi/kryeziu1/opsd/checkpoints"

# ── Run 0: 1.7B→1.7B, default prompts, 2 nodes ─────────────────────────────
export COMMAND="uv run python examples/run_sampled_token_distillation.py --config ${CONFIG} \
    sampled_token_distillation.val_at_start=true \
    policy.optimizer.kwargs.lr=2e-4 \
    checkpointing.checkpoint_dir=${CKPT}/std-1.7b-self-cot \
    logger.log_dir=logs/std-1.7b-self-cot \
    logger.wandb.name=std-1.7b-self-cot"
sbatch --nodes=2 --job-name="std-1.7b-self" "$SCRIPT_DIR/ray.sub"

# ── Run 1: 4B→1.7B, default prompts, 2 nodes ───────────────────────────────
export COMMAND="uv run python examples/run_sampled_token_distillation.py --config ${CONFIG} \
    teacher.model_name=Qwen/Qwen3-4B \
    sampled_token_distillation.val_at_start=true \
    policy.optimizer.kwargs.lr=2e-4 \
    checkpointing.checkpoint_dir=${CKPT}/std-4b-to-1.7b-cot \
    logger.log_dir=logs/std-4b-to-1.7b-cot \
    logger.wandb.name=std-4b-to-1.7b-cot"
sbatch --nodes=2 --job-name="std-4b-1.7b-cot" "$SCRIPT_DIR/ray.sub"

# ── Run 2: 4B→1.7B, prefix + teacher-concise prompts, 2 nodes ──────────────
export COMMAND="uv run python examples/run_sampled_token_distillation.py --config ${CONFIG} \
    teacher.model_name=Qwen/Qwen3-4B \
    sampled_token_distillation.val_at_start=true \
    policy.optimizer.kwargs.lr=2e-4 \
    data.default.prompt_file=examples/prompts/prefix.txt \
    data.default.teacher_prompt_file=examples/prompts/teacher-concise.txt \
    checkpointing.checkpoint_dir=${CKPT}/std-4b-to-1.7b-prefix \
    logger.log_dir=logs/std-4b-to-1.7b-prefix \
    logger.wandb.name=std-4b-to-1.7b-prefix"
sbatch --nodes=2 --job-name="std-4b-1.7b-pfx" "$SCRIPT_DIR/ray.sub"

# ── Run 3: 4B→4B, prefix + teacher-concise prompts, 2 nodes ────────────────
export COMMAND="uv run python examples/run_sampled_token_distillation.py --config ${CONFIG} \
    policy.model_name=Qwen/Qwen3-4B \
    teacher.model_name=Qwen/Qwen3-4B \
    sampled_token_distillation.val_at_start=true \
    policy.optimizer.kwargs.lr=1e-4 \
    data.default.prompt_file=examples/prompts/prefix.txt \
    data.default.teacher_prompt_file=examples/prompts/teacher-concise.txt \
    checkpointing.checkpoint_dir=${CKPT}/std-4b-self-prefix \
    logger.log_dir=logs/std-4b-self-prefix \
    logger.wandb.name=std-4b-self-prefix"
sbatch --nodes=2 --job-name="std-4b-self-pfx" "$SCRIPT_DIR/ray.sub"

# ── Run 4: 8B→8B, prefix + teacher-concise prompts, 4 nodes ────────────────
export COMMAND="uv run python examples/run_sampled_token_distillation.py --config ${CONFIG} \
    policy.model_name=Qwen/Qwen3-8B \
    teacher.model_name=Qwen/Qwen3-8B \
    cluster.num_nodes=4 \
    policy.generation.vllm_cfg.tensor_parallel_size=4 \
    sampled_token_distillation.val_at_start=true \
    policy.optimizer.kwargs.lr=5e-5 \
    data.default.prompt_file=examples/prompts/prefix.txt \
    data.default.teacher_prompt_file=examples/prompts/teacher-concise.txt \
    checkpointing.checkpoint_dir=${CKPT}/std-8b-self-prefix \
    logger.log_dir=logs/std-8b-self-prefix \
    logger.wandb.name=std-8b-self-prefix"
sbatch --nodes=4 --job-name="std-8b-self-pfx" "$SCRIPT_DIR/ray.sub"

echo "All 5 runs submitted."
