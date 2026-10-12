#!/bin/bash
#SBATCH --job-name=vllm-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --account=coreai_nvfm_llm
#SBATCH --output=vllm-build-%j.log

###############################################################################
# Build custom vLLM + FlashInfer environment and save as .sqsh.
# Clones repos fresh from git each time — portable across clusters.
#
# Base: vllm/vllm-openai:v0.17.0-cu130
#
# Usage:
#   cd /lustre/fsw/portfolios/coreai/users/tbarnatan/repos/vllm
#   sbatch build-hsg.sh
#   # Then use serve scripts with SQSH_OVERRIDE pointing to the saved .sqsh
###############################################################################
set -euo pipefail

VLLM_REPO="${VLLM_REPO:-https://github.com/TomerBN-Nvidia/vllm.git}"
VLLM_BRANCH="${VLLM_BRANCH:-ultra-rl-v0.17}"
FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/TomerBN-Nvidia/flashinfer.git}"
FLASHINFER_BRANCH="${FLASHINFER_BRANCH:-ultra-rl}"

CONTAINER="vllm/vllm-openai:v0.17.0-cu130"
SQSH_PATH="${SQSH_PATH:-/lustre/fsw/portfolios/coreai/users/tbarnatan/containers/vllm-hsg.sqsh}"
MOUNTS="/lustre:/lustre"

mkdir -p "$(dirname "${SQSH_PATH}")"

srun --nodes=1 --ntasks=1 \
    --container-image=${CONTAINER} \
    --container-mounts=${MOUNTS} \
    --container-save=${SQSH_PATH} \
    bash -c '
set -euo pipefail

echo "========================================="
echo "Job: '"$SLURM_JOB_ID"'"
echo "Node: $(hostname)"
echo "Date: $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "========================================="

VLLM_REPO="'"${VLLM_REPO}"'"
VLLM_BRANCH="'"${VLLM_BRANCH}"'"
FLASHINFER_REPO="'"${FLASHINFER_REPO}"'"
FLASHINFER_BRANCH="'"${FLASHINFER_BRANCH}"'"
WORKDIR="/opt/vllm-build"

if ! command -v python &>/dev/null; then
    ln -sf "$(which python3)" /usr/local/bin/python
fi

apt-get update -y && apt-get install -y git nvidia-cuda-dev libcublas-dev-13-0 2>&1 | tail -5

echo "python3: $(python3 --version)"
echo "nvcc:    $(nvcc --version 2>/dev/null | grep release || echo NOT FOUND)"
echo "torch:   $(python3 -c "import torch; print(torch.__version__)")"
echo "cuda:    $(python3 -c "import torch; print(torch.version.cuda)")"

###############################################################################
# Step 1: Install uv
###############################################################################
echo ""
echo ">>> Step 1: Installing uv..."
pip install uv 2>&1 | tail -3

###############################################################################
# Step 2: Clone and install custom vLLM (precompiled)
###############################################################################
echo ""
echo ">>> Step 2: Cloning and installing custom vLLM..."
git clone -b "${VLLM_BRANCH}" "${VLLM_REPO}" "${WORKDIR}/vllm" 2>&1 | tail -3
cd "${WORKDIR}/vllm"
echo "Branch: $(git branch --show-current), HEAD: $(git rev-parse --short HEAD)"
git log --oneline -3

export VLLM_USE_PRECOMPILED=1
export SETUPTOOLS_SCM_PRETEND_VERSION="0.17.0"
uv pip install --system -e . --prerelease=allow --torch-backend=auto \
    --index-strategy unsafe-best-match 2>&1

###############################################################################
# Step 3: Clone and install custom FlashInfer
###############################################################################
echo ""
echo ">>> Step 3: Removing base FlashInfer..."
pip uninstall -y flashinfer-cubin flashinfer-jit-cache flashinfer-python 2>/dev/null || true

echo ""
echo ">>> Cloning and installing custom FlashInfer..."
git clone -b "${FLASHINFER_BRANCH}" "${FLASHINFER_REPO}" "${WORKDIR}/flashinfer" 2>&1 | tail -3
cd "${WORKDIR}/flashinfer"
echo "Branch: $(git branch --show-current), HEAD: $(git rev-parse --short HEAD)"
git submodule update --init --recursive 2>&1 | tail -5

export FLASHINFER_CUDA_ARCH_LIST=10.0a
uv pip install --system --prerelease=allow --no-build-isolation -e . 2>&1

echo ""
echo ">>> Installing FlashInfer cubin..."
uv pip install --system -e "./flashinfer-cubin" 2>&1

echo ""
echo ">>> Installing FlashInfer jit cache..."
uv pip install --system flashinfer-jit-cache --index flashinfer-internal-pypi=https://gitlab-master.nvidia.com/api/v4/projects/65523/packages/pypi/simple 2>&1

###############################################################################
# Step 4: Additional dependencies
###############################################################################
echo ""
echo ">>> Step 4: Installing modelopt + cuDNN..."
uv pip install --system nvidia-modelopt 2>&1 | tail -5
uv pip install --system --force-reinstall nvidia-cudnn-cu12 nvidia-cudnn-frontend 2>&1 | tail -5

###############################################################################
# Step 5: Verify
###############################################################################
echo ""
echo ">>> Step 5: Verifying installation"
python3 -c "import vllm; print(f'"'"'vLLM: {vllm.__version__}'"'"')"
python3 -c "import flashinfer; print(f'"'"'FlashInfer: {flashinfer.__version__}'"'"')"
python3 -c "import torch; print(f'"'"'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}'"'"')"

echo ""
echo ">>> Build complete. Container will be saved to: '"${SQSH_PATH}"'"
'

echo "Squash file saved: ${SQSH_PATH}"
ls -lh "${SQSH_PATH}"
