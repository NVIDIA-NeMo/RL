#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --job-name=pr3436-nightly-smoke

set -euo pipefail

: "${CONTAINER:?Set CONTAINER to an immutable nightly .sqsh path}"
: "${SOURCE_DIR:?Set SOURCE_DIR to the NeMo-RL checkout}"
: "${VENV_DIR:?Set VENV_DIR to a Lustre path}"
: "${PROBE_DIR:?Set PROBE_DIR to a Lustre path}"
: "${EXPECTED_HEAD:?Set EXPECTED_HEAD to the source commit}"
: "${EXPECTED_DEEPEP_COMMIT:?Set EXPECTED_DEEPEP_COMMIT to the DeepEP pin}"

mkdir -p "${PROBE_DIR}" "${VENV_DIR}"
test -f "${CONTAINER}"
test -d "${SOURCE_DIR}/.git"

export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
export UV_CACHE_DIR="${PROBE_DIR}/uv-cache"
export TMPDIR="${PROBE_DIR}/tmp"
mkdir -p "${TMPDIR}"

srun \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  --export=ALL \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre \
  bash -lc '
    set -euo pipefail
    cd "${SOURCE_DIR}"
    test "$(git rev-parse HEAD)" = "${EXPECTED_HEAD}"
    git diff --quiet
    git diff --cached --quiet
    uv lock --check
    grep -F "deepseek-ai/DeepEP.git?rev=${EXPECTED_DEEPEP_COMMIT}#${EXPECTED_DEEPEP_COMMIT}" uv.lock
    python -V
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    python - <<"PY"
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"visible_gpus={torch.cuda.device_count()}")
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
PY
    uv sync --locked --extra automodel
    UV_NO_SYNC=1 uv run python - <<"PY"
import importlib.metadata

import deep_ep
import nemo_rl
import torch
import transformer_engine.pytorch

print(f"nemo_rl={nemo_rl.__file__}")
print("deep_ep=" + importlib.metadata.version("deep-ep"))
print(f"deep_ep_module={deep_ep.__file__}")
print("transformer_engine=" + importlib.metadata.version("transformer-engine"))
print(f"cuda_device={torch.cuda.get_device_name(0)}")
assert importlib.metadata.version("deep-ep").endswith("+17cfb81")
print("NIGHTLY_H100_SMOKE_PASS")
PY
  '
