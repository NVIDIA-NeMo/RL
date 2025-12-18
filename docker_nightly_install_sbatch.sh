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

# Use SLURM submit directory as container directory
CONTAINER_DIR="${SLURM_SUBMIT_DIR}"
DATE=$(date +%Y%m%d)

cd "$CONTAINER_DIR"

echo "📦 Downloading nemo-rl nightly container..."
echo "   Directory: $CONTAINER_DIR"
echo "   Date: $DATE"

# Download latest nightly
enroot import docker://nvcr.io#nvidian/nemo-rl:nightly

# Rename with date and create symlink
mv nvidian+nemo-rl+nightly.sqsh nemo_rl_nightly_${DATE}.sqsh
ln -sf nemo_rl_nightly_${DATE}.sqsh nemo_rl_nightly.sqsh

echo "✅ Updated to nightly ${DATE}"
echo "   Container: ${CONTAINER_DIR}/nemo_rl_nightly.sqsh"
ls -la ${CONTAINER_DIR}/nemo_rl_nightly*.sqsh

