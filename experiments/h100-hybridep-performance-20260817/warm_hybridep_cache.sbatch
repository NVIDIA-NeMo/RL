#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=warm-hybridep-cache
#SBATCH --output=warm-hybridep-cache-%j.log

set -euo pipefail

: "${CONTAINER:?Set CONTAINER to the immutable experiment image}"
: "${MOUNTS:?Set MOUNTS to the shared filesystem mounts}"
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the NeMo-RL worktree}"
: "${UV_CACHE_DIR_OVERRIDE:?Set UV_CACHE_DIR_OVERRIDE to shared storage}"

if [[ ${PROJECT_ROOT} != /* ]]; then
  echo "PROJECT_ROOT must be absolute" >&2
  exit 2
fi
if [[ ${UV_CACHE_DIR_OVERRIDE} != /* ]]; then
  echo "UV_CACHE_DIR_OVERRIDE must be absolute" >&2
  exit 2
fi

mkdir -p "${UV_CACHE_DIR_OVERRIDE}"

# shellcheck disable=SC2016  # The container shell expands warm_venv.
srun \
  --container-image="${CONTAINER}" \
  --container-mounts="${MOUNTS},${UV_CACHE_DIR_OVERRIDE}:/root/.cache/uv" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc '
    set -euo pipefail
    export UV_CACHE_DIR=/root/.cache/uv
    warm_venv=/tmp/hybridep-cache-warm
    uv venv --system-site-packages "${warm_venv}"
    uv pip install \
      --python "${warm_venv}/bin/python" \
      --reinstall \
      --no-deps \
      "deep_ep @ git+https://github.com/deepseek-ai/DeepEP.git@17cfb817bccec3a9c247013360cc550c2bac441e"
    "${warm_venv}/bin/python" -c "import deep_ep; print(deep_ep.__file__)"
  '
