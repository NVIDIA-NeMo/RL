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
    policy.model_name="Qwen/Qwen3-1.7B-Base" \
    policy.optimizer.kwargs.lr=2e-5 \
    loss_fn.kl_type=mixed
