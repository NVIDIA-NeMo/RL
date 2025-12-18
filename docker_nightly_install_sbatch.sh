#!/bin/bash
#SBATCH -p gb200
#SBATCH -A coreai_dlalgo_llm
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=nemo_rl_nightly_install
#SBATCH --output=nightly_install_%j.log

# docker_nightly_install_sbatch.sh - Download latest nemo-rl nightly container via SLURM

set -eoux pipefail

# Remove conda/miniconda from PATH to avoid binary incompatibility (ARM vs x86)
# The miniconda zstd binary may not work on compute nodes
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -E '(conda|miniconda)' | tr '\n' ':' | sed 's/:$//')

# Use SLURM submit directory as container directory
# SLURM_SUBMIT_DIR is the directory where sbatch was called from
CONTAINER_DIR="${SLURM_SUBMIT_DIR}"
DATE=$(date +%Y%m%d)

cd "$CONTAINER_DIR"

# Set enroot cache to a writable directory (avoid /lustre/fsw/portfolios permission issue)
export ENROOT_CACHE_PATH="${CONTAINER_DIR}/.enroot_cache"
export ENROOT_DATA_PATH="${CONTAINER_DIR}/.enroot_data"
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH"

OUTPUT_FILE="nemo_rl_nightly_${DATE}.sqsh"

echo "📦 Downloading nemo-rl nightly container..."
echo "   Directory: $CONTAINER_DIR"
echo "   Date: $DATE"
echo "   Output: $OUTPUT_FILE"
echo "   Cache: $ENROOT_CACHE_PATH"

# Remove existing file if it exists (for re-download)
if [[ -f "$OUTPUT_FILE" ]]; then
    echo "⚠️  Removing existing file: $OUTPUT_FILE"
    rm -f "$OUTPUT_FILE"
fi

# Download latest nightly with explicit output path
enroot import -o "$OUTPUT_FILE" docker://nvcr.io#nvidian/nemo-rl:nightly

# Create symlink to latest
ln -sf nemo_rl_nightly_${DATE}.sqsh nemo_rl_nightly.sqsh

echo "✅ Updated to nightly ${DATE}"
echo "   Container: ${CONTAINER_DIR}/nemo_rl_nightly.sqsh"
ls -la ${CONTAINER_DIR}/nemo_rl_nightly*.sqsh

