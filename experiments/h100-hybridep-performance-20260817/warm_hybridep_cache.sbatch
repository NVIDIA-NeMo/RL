#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
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
export PROJECT_ROOT UV_CACHE_DIR_OVERRIDE

# shellcheck disable=SC2016  # The container shell expands warm_venv.
srun \
  --container-image="${CONTAINER}" \
  --container-mounts="${MOUNTS},${UV_CACHE_DIR_OVERRIDE}:/root/.cache/uv" \
  --container-workdir="${PROJECT_ROOT}" \
  --no-container-mount-home \
  bash -lc '
    set -euo pipefail
    export HYBRID_EP_MULTINODE=1
    export MAX_JOBS="${MAX_JOBS:-8}"
    export NVTE_CUDA_ARCHS=90
    export RDMA_CORE_HOME=/usr
    export TORCH_CUDA_ARCH_LIST=9.0
    export USE_NIXL=0
    export UV_CACHE_DIR=/root/.cache/uv
    warm_venv="/tmp/hybridep-cache-warm-${SLURM_JOB_ID}"
    export UV_PROJECT_ENVIRONMENT="${warm_venv}"
    command -v uv
    uv --version
    uv sync \
      --locked \
      --extra mcore \
      --directory "${PROJECT_ROOT}" \
      --no-install-project
    "${warm_venv}/bin/python" -c "import deep_ep; print(deep_ep.__file__)"
  '
