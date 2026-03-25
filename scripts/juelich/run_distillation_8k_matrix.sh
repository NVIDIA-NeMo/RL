#!/bin/bash
# Submits a distillation job via the Ray multi-node launcher.
# Usage: ./run_distillation_8k_matrix.sh <kl_type> <zero_outside_topk>
# Example: ./run_distillation_8k_matrix.sh mixed true

set -euo pipefail

KL_TYPE=${1:?"Usage: $0 <kl_type: forward|reverse|mixed> <zero_outside_topk: true|false>"}
ZERO_OUTSIDE_TOPK=${2:?"Usage: $0 <kl_type> <zero_outside_topk: true|false>"}
NUM_NODES=${3:-2}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export COMMAND="uv run python examples/run_distillation.py \
    cluster.num_nodes=${NUM_NODES} \
    cluster.gpus_per_node=4 \
    policy.model_name=Qwen/Qwen3-1.7B \
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
    distillation.topk_logits_k=1024 \
    policy.optimizer.kwargs.lr=5e-6 \
    loss_fn.kl_type=${KL_TYPE} \
    loss_fn.zero_outside_topk=${ZERO_OUTSIDE_TOPK} \
    'checkpointing.checkpoint_dir=/p/scratch/scifi/kryeziu1/opsd/checkpoints/distillation-gen8k-\${policy.model_name}-\${loss_fn.kl_type}-lr\${policy.optimizer.kwargs.lr}-bs\${policy.train_global_batch_size}-topk\${distillation.topk_logits_k}-zotk\${loss_fn.zero_outside_topk}' \
    'logger.log_dir=logs/distillation-gen8k-\${policy.model_name}-\${loss_fn.kl_type}-lr\${policy.optimizer.kwargs.lr}-bs\${policy.train_global_batch_size}-topk\${distillation.topk_logits_k}-zotk\${loss_fn.zero_outside_topk}'"

sbatch --nodes="${NUM_NODES}" "$SCRIPT_DIR/ray.sub"
