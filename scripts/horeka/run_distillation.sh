#!/bin/bash
#SBATCH --job-name=run_distillation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32    # More CPUs = faster parallel compilation of flash-attn/vLLM
#SBATCH --gres=gpu:4          # GPU required to compile flash-attn, deep_ep, deep_gemm
#SBATCH --time=04:00:00       # flash-attn ~1h, vLLM ~1-2h to compile from source
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0023960
#SBATCH --output=/hkfs/work/workspace/scratch/tum_hki2875-myspace/RL/scripts/logs/%j-%x.out

module purge
module load compiler/gnu/13
module load devel/cuda/12.9

# Make CUDA compiler visible to build tools (flash-attn, deep_ep, etc.)
export CUDA_HOME=$CUDA_DIR
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd /hkfs/work/workspace/scratch/tum_hki2875-myspace/RL

export WANDB_MODE=offline
export HF_HOME=/hkfs/work/workspace/scratch/tum_hki2875-myspace/.cache/huggingface
export HF_HUB_OFFLINE=1

uv run python examples/run_distillation.py
