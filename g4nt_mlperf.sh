#!/bin/bash
# Allocate a 4-node Ray cluster using the MLPerf+TRT-LLM image.
# Sized for 397B skip-training: 2 generation nodes (2 TRT-LLM replicas TP=4);
# the stub policy leaves the other 2 nodes unreserved.
#
# Usage:
#   bash g4nt_mlperf.sh                      # GB300 (default)
#   PARTITION=gb200 bash g4nt_mlperf.sh      # GB200
#   NODES=8 bash g4nt_mlperf.sh             # different node count
#   TIME=2:0:0 bash g4nt_mlperf.sh          # different walltime
#
# Once the cluster is up, attach with:
#   bash <LOGDIR>/slurm-<JOBID>-attach.sh   (printed below)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

CONTAINER="${CONTAINER:-/lustre/fsw/coreai_comparch_trtllm/erinh/enroot/mlperf-trtllm-qwen35-20260717.sqsh}"
PARTITION="${PARTITION:-gb300}"
NODES="${NODES:-4}"
TIME="${TIME:-4:0:0}"
RUN_LOG_DIR="${REPO_ROOT}/logs/g4nt_mlperf_${PARTITION}"
MLPERF_RECIPE_ROOT="${MLPERF_RECIPE_ROOT:-/lustre/fsw/coreai_comparch_trtllm/erinh/optimized/qwen35_397b_grpo/pytorch}"
TRTLLM_PYTHON="${TRTLLM_PYTHON:-/opt/nemo_rl_venv/bin/python}"

# Preserve baked MLPerf/Gym dependencies while overlaying live source.
# Canonical /opt/nemo-rl paths keep Ray actor git_root in the baked tree.
MOUNTS="/lustre:/lustre"
MOUNTS+=",${REPO_ROOT}/nemo_rl:/opt/nemo-rl/nemo_rl"
MOUNTS+=",${REPO_ROOT}/examples:/opt/nemo-rl/examples"
MOUNTS+=",${REPO_ROOT}/qwen_35:/opt/nemo-rl/qwen_35"
MOUNTS+=",${MLPERF_RECIPE_ROOT}:/workspace/llm"

mkdir -p "${RUN_LOG_DIR}"

SBATCH_OUTPUT=$(
  CONTAINER="${CONTAINER}" \
  MOUNTS="${MOUNTS},/dev/fuse:/dev/fuse" \
  CONTAINER_WORKDIR=/opt/nemo-rl \
  NEMO_RL_PY_EXECUTABLES_TRTLLM="${TRTLLM_PYTHON}" \
  NRL_IGNORE_VERSION_MISMATCH=1 \
  MOUNT_LOG_DIR_IN_CONTAINER=1 \
  NRL_CLEAN_SLURM_PMIX="${NRL_CLEAN_SLURM_PMIX:-1}" \
  RAY_OBJECT_STORE_MEMORY_BYTES="${RAY_OBJECT_STORE_MEMORY_BYTES:-34359738368}" \
  GPUS_PER_NODE=4 \
  BASE_LOG_DIR="${RUN_LOG_DIR}" \
  sbatch \
    --nodes="${NODES}" \
    --account=coreai_comparch_trtllm \
    --partition="${PARTITION}" \
    --job-name="g4nt_mlperf_${PARTITION}" \
    --exclusive \
    --mem=0 \
    --time="${TIME}" \
    --output="${RUN_LOG_DIR}/slurm-%j.out" \
    --error="${RUN_LOG_DIR}/slurm-%j.out" \
    "${REPO_ROOT}/ray.sub"
)

echo "${SBATCH_OUTPUT}" >&2
JOB_ID=$(printf '%s\n' "${SBATCH_OUTPUT}" | grep -o '[0-9]\+' | tail -1)

if [[ -n "${JOB_ID}" ]]; then
  printf '%s\n' "${JOB_ID}" > "${REPO_ROOT}/latest_job_id.txt"
  echo "[INFO] Job ID:     ${JOB_ID}"
  echo "[INFO] Container:  ${CONTAINER}"
  echo "[INFO] RL source:  ${REPO_ROOT}/nemo_rl"
  echo "[INFO] Recipe:     ${MLPERF_RECIPE_ROOT}"
  echo "[INFO] TRT Python: ${TRTLLM_PYTHON}"
  echo "[INFO] Partition:  ${PARTITION}"
  echo "[INFO] Nodes:      ${NODES}"
  echo "[INFO] Logs:       ${RUN_LOG_DIR}/slurm-${JOB_ID}.out"
  echo "[INFO] Attach:     bash ${REPO_ROOT}/${JOB_ID}-attach.sh   (ready once Ray head is up)"
  echo "[INFO] Status:     squeue -j ${JOB_ID}"
fi
