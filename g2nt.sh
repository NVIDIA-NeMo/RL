#!/bin/bash
# Allocate a 2-node GB200 Ray cluster using the nemo-rl+TRT-LLM image.
# Sized for grpo-qwen3-8b-2n4g-megatron-trtllm-async-1off (1 train + 1 gen node).
#
# Usage (from ~/RL or anywhere):
#   bash g2nt.sh
#
# Once the cluster is up, attach with:
#   bash <LOGDIR>/slurm-<JOBID>-attach.sh   (printed below)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONTAINER="${CONTAINER:-/lustre/fsw/coreai_comparch_trtllm/erinh/enroot/shuyix-nemo-rl-trtllm-20260714-aarch64.squashfs}"
RUN_LOG_DIR="${REPO_ROOT}/logs/g2nt"

mkdir -p "${RUN_LOG_DIR}"

SBATCH_OUTPUT=$(
  CONTAINER="${CONTAINER}" \
  MOUNTS="/lustre:/lustre,${REPO_ROOT}:${REPO_ROOT},/dev/fuse:/dev/fuse" \
  GPUS_PER_NODE=4 \
  BASE_LOG_DIR="${RUN_LOG_DIR}" \
  sbatch \
    --nodes=2 \
    --account=coreai_comparch_trtllm \
    --partition=gb200 \
    --job-name=g2nt \
    --exclusive \
    --mem=0 \
    --time=4:0:0 \
    --output="${RUN_LOG_DIR}/slurm-%j.out" \
    --error="${RUN_LOG_DIR}/slurm-%j.out" \
    "${REPO_ROOT}/ray.sub"
)

echo "${SBATCH_OUTPUT}" >&2
JOB_ID=$(printf '%s\n' "${SBATCH_OUTPUT}" | grep -o '[0-9]\+' | tail -1)

if [[ -n "${JOB_ID}" ]]; then
  printf '%s\n' "${JOB_ID}" > "${REPO_ROOT}/latest_job_id.txt"
  echo "[INFO] Job ID: ${JOB_ID}"
  echo "[INFO] Logs:   ${RUN_LOG_DIR}/slurm-${JOB_ID}.out"
  echo "[INFO] Attach: bash ${RUN_LOG_DIR}/${JOB_ID}-logs/${JOB_ID}-attach.sh   (ready once Ray head is up)"
  echo "[INFO] Status: squeue -j ${JOB_ID}"
fi
