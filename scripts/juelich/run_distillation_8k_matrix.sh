#!/bin/bash
#SBATCH --job-name=run_distillation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32    # More CPUs = faster parallel compilation of flash-attn/vLLM
#SBATCH --gres=gpu:4          # GPU required to compile flash-attn, deep_ep, deep_gemm
#SBATCH --time=24:00:00       # flash-attn ~1h, vLLM ~1-2h to compile from source
#SBATCH --partition=booster
#SBATCH --account=envcomp
#SBATCH --output=/p/project1/envcomp/yll/RL/scripts/logs/%j-%x.out

# Usage: sbatch run_distillation_8k_matrix.sh <kl_type> <zero_outside_topk>
# Example: sbatch run_distillation_8k_matrix.sh mixed true

KL_TYPE=${1:?"Usage: $0 <kl_type: forward|reverse|mixed> <zero_outside_topk: true|false>"}
ZERO_OUTSIDE_TOPK=${2:?"Usage: $0 <kl_type> <zero_outside_topk: true|false>"}

module purge
module load Stages/2025
module load GCC
module load CUDA
module load git
module load cuDNN
module load NCCL

cd /p/project1/envcomp/yll/RL

export TORCH_CUDA_ARCH_LIST='9.0 10.0 11.0 12.0'
export WANDB_MODE=offline
export HF_HOME=/p/project1/envcomp/yll/.cache/huggingface
export HF_HUB_OFFLINE=1

uv run python examples/run_distillation.py \
    policy.model_name="Qwen/Qwen3-1.7B" \
    policy.max_total_sequence_length=8704 \
    policy.generation.max_new_tokens=8192 \
    policy.generation.vllm_cfg.max_model_len=33280 \
    policy.dynamic_batching.train_mb_tokens=16384 \
    policy.dynamic_batching.logprob_mb_tokens=16384 \
    teacher.dynamic_batching.logprob_mb_tokens=16384 \
    distillation.val_at_start=false \
    distillation.val_max_total_sequence_length=33280 \
    distillation.val_max_new_tokens=32768 \
    distillation.topk_logits_k=2048 \
    policy.optimizer.kwargs.lr=5e-6 \
    loss_fn.kl_type=${KL_TYPE} \
    loss_fn.zero_outside_topk=${ZERO_OUTSIDE_TOPK} \
    'checkpointing.checkpoint_dir=/p/scratch/scifi/kryeziu1/opsd/checkpoints/distillation-gen8k-${policy.model_name}-${loss_fn.kl_type}-lr${policy.optimizer.kwargs.lr}-bs${policy.train_global_batch_size}-topk${distillation.topk_logits_k}-zotk${loss_fn.zero_outside_topk}' \
    'logger.log_dir=logs/distillation-gen8k-${policy.model_name}-${loss_fn.kl_type}-lr${policy.optimizer.kwargs.lr}-bs${policy.train_global_batch_size}-topk${distillation.topk_logits_k}-zotk${loss_fn.zero_outside_topk}'
