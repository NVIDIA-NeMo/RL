#!/bin/bash
set -euo pipefail

# Direct-install script for JUPITER login nodes (no Slurm submission required).

module purge
module load Stages/2025
module load CUDA
module load GCC
module load git

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

cd "$REPO_ROOT"
git submodule update --init --recursive

# Install core + fsdp (flash-attn) + vllm extras.
uv pip install -U pip setuptools wheel ninja packaging cmake
uv sync --extra fsdp --extra vllm

echo "Installation complete. Verify with:"
echo "  uv run python -c 'import flash_attn; print(flash_attn.__version__)'"
echo "  uv run python -c 'import vllm; print(vllm.__version__)'"
