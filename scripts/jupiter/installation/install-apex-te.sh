#!/bin/bash
set -euo pipefail

# Direct-install script for JUPITER login nodes (no Slurm submission required).

module purge
module load Stages/2025
module load GCC
module load cuDNN
module load CUDA
module load NCCL
module load git

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${EBROOTCUDA:-}" ]]; then
  export CUDA_HOME="$EBROOTCUDA"
elif command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME
  CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
fi
export CUDA_PATH="${CUDA_HOME:-}"

# If not provided, detect the current GPU architecture (e.g., 9.0 for H100).
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  DETECTED_ARCH="$(uv run python - <<'PY'
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"{major}.{minor}")
PY
)"
  if [[ -n "${DETECTED_ARCH}" ]]; then
    export TORCH_CUDA_ARCH_LIST="${DETECTED_ARCH}"
  fi
fi

CPUS="$(nproc)"
if (( CPUS > 16 )); then
  export MAX_JOBS="${MAX_JOBS:-16}"
else
  export MAX_JOBS="${MAX_JOBS:-$CPUS}"
fi

# Transformer Engine: pinned to the revision required by Megatron-LM.
TE_REV="5671fd3675906cda1ade26c24a65d3dedd88eb89"
TE_DIR="$(mktemp -d)"

echo "Cloning Transformer Engine @ ${TE_REV} ..."
git clone https://github.com/NVIDIA/TransformerEngine.git "$TE_DIR"
cd "$TE_DIR"
git checkout "$TE_REV"
git submodule update --init --recursive
cd "$REPO_ROOT"

echo "Installing Transformer Engine..."
export NVTE_FRAMEWORK=pytorch
uv pip install --no-build-isolation "${TE_DIR}[pytorch]"

rm -rf "$TE_DIR"

# Apex has no stable PyPI release with CUDA extensions; build from source.
APEX_DIR="$(mktemp -d)"

echo "Cloning Apex..."
git clone https://github.com/NVIDIA/apex "$APEX_DIR"

echo "Installing Apex (cpp_ext + cuda_ext flags from env as supported)..."
APEX_CPP_EXT=1 uv pip install -v --no-build-isolation "$APEX_DIR"

rm -rf "$APEX_DIR"

echo ""
echo "Verify with:"
echo "  uv run python -c 'import transformer_engine; print(\"TE:\", transformer_engine.__version__)'"
echo "  uv run python -c 'import apex; print(\"Apex OK\")'"
