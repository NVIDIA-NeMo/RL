#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:15:00
#SBATCH --job-name=pr3436-deepep-dispatch

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
export NVSHMEM_DEBUG=${NVSHMEM_DEBUG:-INFO}
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export NVSHMEM_HCA_PE_MAPPING=${NVSHMEM_HCA_PE_MAPPING:-mlx5_0:1:1,mlx5_3:1:1,mlx5_5:1:1,mlx5_7:1:1,mlx5_9:1:1,mlx5_12:1:1,mlx5_14:1:1,mlx5_16:1:1}
export NVSHMEM_IB_GID_INDEX=${NVSHMEM_IB_GID_INDEX:-3}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

echo "master=${MASTER_ADDR}:${MASTER_PORT}"
echo "nvshmem_hca_pe_mapping=${NVSHMEM_HCA_PE_MAPPING}"
echo "nvshmem_ib_gid_index=${NVSHMEM_IB_GID_INDEX}"

srun \
  --nodes="${SLURM_NNODES}" \
  --ntasks="${SLURM_NTASKS}" \
  --ntasks-per-node=8 \
  --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="${SOURCE_DIR}" \
  bash -lc '
    set -euo pipefail
    export PATH="/root/.local/bin:${PATH}"
    cd "${SOURCE_DIR}"
    grep -F "deepseek-ai/DeepEP.git?rev=${EXPECTED_DEEPEP_COMMIT}#${EXPECTED_DEEPEP_COMMIT}" uv.lock
    uv run --locked --extra automodel python "${PROBE_SCRIPT}"
  '
