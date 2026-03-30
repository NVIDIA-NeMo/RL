#!/bin/bash
# Submit 3 rewrite-teacher distillation runs on booster.
# Dataset: OpenThoughtsMath (yllkryeziu/openthoughts114k-math-qwen3)
# Teacher prompt: examples/prompts/teacher_rewrite.txt
# All runs: reverse KL, topk=512, 16k generation, 4 nodes × 4 GPUs
# Batch: num_prompts_per_step=32, train/generation_batch_size=32
#   (128 prompts × 33k teacher tokens exhausts head-node CPU RAM; 32 is safe)
#
# Run 0: 1.7B self-distill  — base config, no model overrides
# Run 1: 4B  self-distill  — override policy + teacher model
# Run 2: 8B  self-distill  — override policy + teacher model, vLLM TP=4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="examples/configs/opsd/juelich/distill-1.7b-to-1.7b-4n4g-gen16k-rewrite.yaml"
CKPT="/p/scratch/scifi/kryeziu1/opsd/checkpoints"

# ── Run 0: 1.7B→1.7B, rewrite teacher, openthoughts, 4 nodes ────────────────
export COMMAND="uv run python examples/run_distillation.py --config ${CONFIG} \
    +data.default.column_mapping.qwen3_1b7_answer=trace \
    distillation.num_prompts_per_step=32 \
    policy.train_global_batch_size=32 \
    policy.generation_batch_size=32 \
    distillation.val_at_start=true \
    checkpointing.checkpoint_dir=${CKPT}/distill-1.7b-self-rewrite \
    logger.log_dir=logs/distill-1.7b-self-rewrite \
    logger.wandb.name=distill-1.7b-self-rewrite"
sbatch --nodes=4 --job-name="d-1.7b-rewrite" "$SCRIPT_DIR/ray.sub"

# ── Run 1: 4B→4B, rewrite teacher, openthoughts, 4 nodes ────────────────────
export COMMAND="uv run python examples/run_distillation.py --config ${CONFIG} \
    policy.model_name=Qwen/Qwen3-4B \
    teacher.model_name=Qwen/Qwen3-4B \
    +data.default.column_mapping.qwen3_4b_answer=trace \
    distillation.num_prompts_per_step=32 \
    policy.train_global_batch_size=32 \
    policy.generation_batch_size=32 \
    distillation.val_at_start=true \
    checkpointing.checkpoint_dir=${CKPT}/distill-4b-self-rewrite \
    logger.log_dir=logs/distill-4b-self-rewrite \
    logger.wandb.name=distill-4b-self-rewrite"
sbatch --nodes=4 --job-name="d-4b-rewrite" "$SCRIPT_DIR/ray.sub"

# ── Run 2: 8B→8B, rewrite teacher, openthoughts, 4 nodes ────────────────────
# vLLM TP=4 required: 8B model fills the entire vLLM budget at TP=1
# on 40GB A100s (0.4×40=16GB), leaving no room for KV cache.
export COMMAND="uv run python examples/run_distillation.py --config ${CONFIG} \
    policy.model_name=Qwen/Qwen3-8B \
    teacher.model_name=Qwen/Qwen3-8B \
    policy.generation.vllm_cfg.tensor_parallel_size=4 \
    +data.default.column_mapping.qwen3_8b_answer=trace \
    distillation.num_prompts_per_step=32 \
    policy.train_global_batch_size=32 \
    policy.generation_batch_size=32 \
    distillation.val_at_start=true \
    checkpointing.checkpoint_dir=${CKPT}/distill-8b-self-rewrite \
    logger.log_dir=logs/distill-8b-self-rewrite \
    logger.wandb.name=distill-8b-self-rewrite"
sbatch --nodes=4 --job-name="d-8b-rewrite" "$SCRIPT_DIR/ray.sub"

# ── Run 3: 14B→14B, rewrite teacher, openthoughts, 4 nodes ──────────────────
# Same TP=4/CP=2 as 8B — 14B weights fit with cpu_offload.
# vLLM TP=4: 28GB/4=7GB model, ~9GB KV cache on 40GB A100s.
export COMMAND="uv run python examples/run_distillation.py --config ${CONFIG} \
    policy.model_name=Qwen/Qwen3-14B \
    teacher.model_name=Qwen/Qwen3-14B \
    policy.generation.vllm_cfg.tensor_parallel_size=4 \
    +data.default.column_mapping.qwen3_14b_answer=trace \
    distillation.num_prompts_per_step=32 \
    policy.train_global_batch_size=32 \
    policy.generation_batch_size=32 \
    distillation.val_at_start=true \
    checkpointing.checkpoint_dir=${CKPT}/distill-14b-self-rewrite \
    logger.log_dir=logs/distill-14b-self-rewrite \
    logger.wandb.name=distill-14b-self-rewrite"
sbatch --nodes=4 --job-name="d-14b-rewrite" "$SCRIPT_DIR/ray.sub"

# ── Run 4: 32B→32B, rewrite teacher, openthoughts, 8 nodes ──────────────────
# TP=4 vLLM OOMs (64GB/4=16GB fills entire 40GB×0.4 budget, 0 for KV cache).
# TP=8 cross-node: 64GB/8=8GB model, 8GB KV cache — handled automatically by
# the framework (unified placement group when TP > gpus_per_node=4).
# distributed_executor_backend=ray is forced by code when model_parallel_size>1.
export COMMAND="uv run python examples/run_distillation.py --config ${CONFIG} \
    policy.model_name=Qwen/Qwen3-32B \
    teacher.model_name=Qwen/Qwen3-32B \
    policy.dtensor_cfg.tensor_parallel_size=8 \
    teacher.dtensor_cfg.tensor_parallel_size=8 \
    policy.generation.vllm_cfg.tensor_parallel_size=8 \
    cluster.num_nodes=8 \
    +data.default.column_mapping.qwen3_32b_answer=trace \
    distillation.num_prompts_per_step=32 \
    policy.train_global_batch_size=32 \
    policy.generation_batch_size=32 \
    distillation.val_at_start=true \
    checkpointing.checkpoint_dir=${CKPT}/distill-32b-self-rewrite \
    logger.log_dir=logs/distill-32b-self-rewrite \
    logger.wandb.name=distill-32b-self-rewrite"
sbatch --nodes=8 --job-name="d-32b-rewrite" "$SCRIPT_DIR/ray.sub"

echo "All 5 rewrite runs submitted."
