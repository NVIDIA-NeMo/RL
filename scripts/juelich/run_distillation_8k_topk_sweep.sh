#!/bin/bash
# Submits a fleet of distillation jobs sweeping topk_logits_k values.
# Uses the best config from the matrix sweep (run 13581780): reverse KL, zero_outside_topk=false.
# Node counts scale with topk to avoid OOM (DP increases automatically).
#
# DP = total_gpus / (TP * CP) = (num_nodes * 4) / (4 * 2) = num_nodes / 2
#   2 nodes → DP=1, 4 nodes → DP=2, 8 nodes → DP=4
#
# Usage: ./run_distillation_8k_topk_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# topk -> num_nodes mapping (more topk = more memory = more nodes for higher DP)
declare -A TOPK_NODES
TOPK_NODES[512]=2    # DP=1, baseline — same as previous best run
TOPK_NODES[2048]=4   # DP=2, halves per-GPU batch vs baseline
TOPK_NODES[4096]=4   # DP=2, 4x topk but 2x less batch per GPU
TOPK_NODES[8192]=8   # DP=4, 8x topk but 4x less batch per GPU

# topk -> train_global_batch_size (scale down to avoid CPU RAM OOM from large topk tensors)
declare -A TOPK_BS
TOPK_BS[512]=64      # baseline
TOPK_BS[2048]=32     # 2x topk → halve batch
TOPK_BS[4096]=16     # 4x topk → quarter batch
TOPK_BS[8192]=16     # 8x topk but 2x more nodes → quarter batch

for TOPK in 512 2048 4096 8192; do
    NUM_NODES=${TOPK_NODES[$TOPK]}
    BS=${TOPK_BS[$TOPK]}
    echo "Submitting topk=${TOPK} with ${NUM_NODES} nodes (DP=$((NUM_NODES * 4 / 8))), bs=${BS}..."

    export COMMAND="uv run python examples/run_distillation.py \
        cluster.num_nodes=${NUM_NODES} \
        cluster.gpus_per_node=4 \
        policy.model_name=Qwen/Qwen3-1.7B \
        policy.train_global_batch_size=${BS} \
        policy.max_total_sequence_length=8704 \
        policy.generation.max_new_tokens=8192 \
        policy.generation.vllm_cfg.max_model_len=33280 \
        policy.generation.vllm_cfg.gpu_memory_utilization=0.4 \
        policy.dynamic_batching.train_mb_tokens=16384 \
        policy.dynamic_batching.logprob_mb_tokens=16384 \
        teacher.dynamic_batching.logprob_mb_tokens=16384 \
        policy.dtensor_cfg.tensor_parallel_size=4 \
        policy.dtensor_cfg.cpu_offload=true \
        policy.offload_optimizer_for_logprob=true \
        distillation.val_at_start=false \
        distillation.val_max_total_sequence_length=33280 \
        distillation.val_max_new_tokens=32768 \
        distillation.topk_logits_k=${TOPK} \
        policy.optimizer.kwargs.lr=5e-6 \
        loss_fn.kl_type=reverse \
        loss_fn.zero_outside_topk=false \
        'checkpointing.checkpoint_dir=/p/scratch/scifi/kryeziu1/opsd/checkpoints/distillation-gen8k-\${policy.model_name}-reverse-lr5e-6-bs\${policy.train_global_batch_size}-topk${TOPK}' \
        'logger.log_dir=logs/distillation-gen8k-\${policy.model_name}-reverse-lr5e-6-bs\${policy.train_global_batch_size}-topk${TOPK}' \
        'logger.wandb.name=distill-DeepScaler-Qwen3-4B-to-1.7B-reverse-topk${TOPK}-lr5e6-gen8k'"

    sbatch --nodes="${NUM_NODES}" --job-name="distill-topk${TOPK}" "$SCRIPT_DIR/ray.sub"
done

echo "All 4 jobs submitted."
