#!/bin/bash
# Allocate a single GB200 node (no container, interactive bash).
#
# Usage:
#   bash g1n.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_LOG_DIR="${REPO_ROOT}/logs/g1n"

mkdir -p "${RUN_LOG_DIR}"

SBATCH_OUTPUT=$(
  sbatch \
    --nodes=1 \
    --account=coreai_comparch_trtllm \
    --partition=gb200 \
    --job-name=g1n \
    --exclusive \
    --mem=0 \
    --time=4:0:0 \
    --output="${RUN_LOG_DIR}/slurm-%j.out" \
    --error="${RUN_LOG_DIR}/slurm-%j.out" \
    --wrap="sleep infinity"
)

echo "${SBATCH_OUTPUT}" >&2
JOB_ID=$(printf '%s\n' "${SBATCH_OUTPUT}" | grep -o '[0-9]\+' | tail -1)

if [[ -n "${JOB_ID}" ]]; then
  printf '%s\n' "${JOB_ID}" > "${REPO_ROOT}/latest_job_id.txt"
  echo "[INFO] Job ID: ${JOB_ID}"
  echo "[INFO] Logs:   ${RUN_LOG_DIR}/slurm-${JOB_ID}.out"
  echo "[INFO] Attach: ssh \$(squeue -j ${JOB_ID} -h -o '%N')"
  echo "[INFO] Status: squeue -j ${JOB_ID}"
fi
