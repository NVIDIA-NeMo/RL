#!/bin/bash
# Smoke test for the mooncake_cpu TQ backend with the jagged-tensor
# wire (commit d447c3e1). Uses the same mcore-1B + seqpack + CP=1 config
# as the v4 baseline, just flips the backend from "simple" to
# "mooncake_cpu". Goal: verify nested tensors survive the mooncake
# distributed-store serialization path.
set -euo pipefail

cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/zhiyul/data-plane/RL

source /lustre/fsw/portfolios/coreai/users/zhiyul/secrets.sh 2>/dev/null || true
export HF_HOME=${HF_HOME:-/lustre/fsw/portfolios/coreai/users/zhiyul/hf}
export NRL_FORCE_REBUILD_VENVS=true

LOG=grpo-mooncake-cpu-smoke.log
echo "=== mooncake_cpu smoke at $(date) ===" | tee "$LOG"

uv run --extra mcore ./examples/run_grpo.py \
    --config examples/configs/grpo_math_1B_megatron.yaml \
    cluster.num_nodes=1 \
    cluster.gpus_per_node=8 \
    grpo.max_num_steps=5 \
    grpo.num_prompts_per_step=8 \
    grpo.num_generations_per_prompt=4 \
    grpo.use_dynamic_sampling=false \
    grpo.val_at_start=false \
    grpo.val_at_end=false \
    policy.train_global_batch_size=32 \
    policy.megatron_cfg.tensor_model_parallel_size=1 \
    policy.megatron_cfg.force_reconvert_from_hf=True \
    policy.sequence_packing.enabled=true \
    checkpointing.enabled=false \
    logger.wandb_enabled=false \
    logger.tensorboard_enabled=false \
    +data_plane.enabled=true \
    +data_plane.impl=transfer_queue \
    +data_plane.backend=mooncake_cpu 2>&1 | tee -a "$LOG"
