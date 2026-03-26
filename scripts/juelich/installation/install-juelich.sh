#!/bin/bash
#SBATCH --job-name=install-RL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32    # More CPUs = faster parallel compilation of flash-attn/vLLM
#SBATCH --gres=gpu:4          # GPU required to compile flash-attn, deep_ep, deep_gemm
#SBATCH --time=04:00:00       # flash-attn ~1h, vLLM ~1-2h to compile from source
#SBATCH --partition=booster
#SBATCH --account=envcomp
#SBATCH --output=/p/project1/envcomp/yll/RL/scripts/logs/%j-%x.out

module purge
module load Stages/2025
module load CUDA
module load GCC
module load git

# Make CUDA compiler visible to build tools (flash-attn, deep_ep, etc.)

cd /p/project1/envcomp/yll/RL
git submodule update --init --recursive

# Install core + fsdp (flash-attn for FSDP2) + vllm (vLLM for generation) extras
# These are the two extras required for FSDP2 distillation with vLLM-based generation
uv pip install -U pip setuptools wheel ninja packaging cmake
uv sync --extra fsdp --extra vllm

echo "Installation complete. Verify with:"
echo "  uv run python -c 'import flash_attn; print(flash_attn.__version__)'"
echo "  uv run python -c 'import vllm; print(vllm.__version__)'"
