#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:20:00
#SBATCH --job-name=pr3436-hybridep-dispatch

set -euo pipefail

: "${CONTAINER:?Set CONTAINER to an immutable nightly .sqsh path}"
: "${SOURCE_DIR:?Set SOURCE_DIR to the NeMo-RL checkout}"
: "${PROBE_DIR:?Set PROBE_DIR to a Lustre path}"
: "${PROBE_SCRIPT:?Set PROBE_SCRIPT to probe_deepep_dispatch.py}"
: "${EXPECTED_HEAD:?Set EXPECTED_HEAD to the source commit}"
: "${EXPECTED_DEEPEP_COMMIT:?Set EXPECTED_DEEPEP_COMMIT to the DeepEP pin}"

mkdir -p "${PROBE_DIR}"
test -f "${CONTAINER}"
test -f "${PROBE_SCRIPT}"
test -d "${SOURCE_DIR}/.git"
test "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
git -C "${SOURCE_DIR}" diff --quiet
git -C "${SOURCE_DIR}" diff --cached --quiet

export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed -n '1p')
export MASTER_PORT=$((6000 + SLURM_JOB_ID % 1500))
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export HYBRID_EP_MULTINODE=0
export TORCH_CUDA_ARCH_LIST=9.0
export USE_MNNVL=0
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8

echo "master=${MASTER_ADDR}:${MASTER_PORT}"
echo "hybrid_ep_multinode=${HYBRID_EP_MULTINODE}"
echo "torch_cuda_arch_list=${TORCH_CUDA_ARCH_LIST}"
echo "use_mnnvl=${USE_MNNVL}"
echo "nvlink_domain_size=${NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN}"

container_name="pr3436-deepep-${SLURM_JOB_ID}"
cleanup() {
  srun \
    --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    enroot remove -f "${container_name}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

srun \
  --nodes="${SLURM_NNODES}" \
  --ntasks="${SLURM_NNODES}" \
  --ntasks-per-node=1 \
  --no-container-mount-home \
  --container-name="${container_name}" \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="${SOURCE_DIR}" \
  bash -lc '
    set -euo pipefail
    export PATH="/root/.local/bin:${PATH}"
    export UV_PROJECT_ENVIRONMENT="/tmp/pr3436-hybridep-${SLURM_JOB_ID}"
    cd "${SOURCE_DIR}"
    uv cache clean deep-ep || true
    uv sync --locked --extra mcore --reinstall-package deep-ep
    uv run --locked --extra mcore python -c "import deep_ep, hybrid_ep_cpp; print(deep_ep.__file__); print(hybrid_ep_cpp.__file__)"
  '

srun \
  --nodes="${SLURM_NNODES}" \
  --ntasks="${SLURM_NTASKS}" \
  --ntasks-per-node=8 \
  --no-container-mount-home \
  --container-name="${container_name}" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="${SOURCE_DIR}" \
  bash -lc '
    set -euo pipefail
    export PATH="/root/.local/bin:${PATH}"
    export UV_PROJECT_ENVIRONMENT="/tmp/pr3436-hybridep-${SLURM_JOB_ID}"
    cd "${SOURCE_DIR}"
    grep -F "deepseek-ai/DeepEP.git?rev=${EXPECTED_DEEPEP_COMMIT}#${EXPECTED_DEEPEP_COMMIT}" uv.lock
    UV_NO_SYNC=1 uv run --extra mcore python "${PROBE_SCRIPT}"
  '
