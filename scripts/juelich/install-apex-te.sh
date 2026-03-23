#!/bin/bash
#SBATCH --job-name=install-apex-te
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4          # GPU required for CUDA extension compilation
#SBATCH --time=02:00:00       # Apex ~30min, TE ~30-60min to compile
#SBATCH --partition=booster
#SBATCH --account=envcomp
#SBATCH --output=/p/project1/envcomp/yll/RL/scripts/juelich/install-apex-te.log

module purge
module load Stages/2025
module load GCC
module load cuDNN
module load CUDA
module load NCCL
module load git

cd /p/project1/envcomp/yll/RL

# CUDA env vars — EBROOTCUDA is set by the CUDA module on JUWELS
export CUDA_HOME=$EBROOTCUDA
export CUDA_PATH=$CUDA_HOME
# JUWELS Booster = NVIDIA A100 (sm_80)
export TORCH_CUDA_ARCH_LIST="8.0"
# Parallel compilation jobs (32 CPUs available)
export MAX_JOBS=16

# ── Transformer Engine ────────────────────────────────────────────────────────
# Pinned to the rev required by Megatron-LM (see 3rdparty/Megatron-LM-workspace/
# Megatron-LM/pyproject.toml [tool.uv.sources])
TE_REV="5671fd3675906cda1ade26c24a65d3dedd88eb89"
TE_DIR=$(mktemp -d)

echo "▶ Cloning Transformer Engine @ $TE_REV ..."
git clone https://github.com/NVIDIA/TransformerEngine.git "$TE_DIR"
cd "$TE_DIR"
git checkout "$TE_REV"
git submodule update --init --recursive
cd -

echo "▶ Installing Transformer Engine..."
export NVTE_FRAMEWORK=pytorch
uv pip install --no-build-isolation "${TE_DIR}[pytorch]"

rm -rf "$TE_DIR"

# ── NVIDIA Apex ───────────────────────────────────────────────────────────────
# Apex has no stable PyPI release with CUDA exts; always build from source.
APEX_DIR=$(mktemp -d)

echo "▶ Cloning Apex..."
git clone https://github.com/NVIDIA/apex "$APEX_DIR"

echo "▶ Installing Apex (cpp_ext + cuda_ext)..."
# CUDA toolkit is 12.6 but PyTorch was built with CUDA 12.9 — Apex's strict
# version check blocks --cuda_ext, so only build the C++ extension.
APEX_CPP_EXT=1 \
  uv pip install -v --no-build-isolation "$APEX_DIR"

rm -rf "$APEX_DIR"

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "✓ Verify with:"
echo "  uv run python -c 'import transformer_engine; print(\"TE:\", transformer_engine.__version__)'"
echo "  uv run python -c 'import apex; print(\"Apex OK\")'"
