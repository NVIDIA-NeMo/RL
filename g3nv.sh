#!/bin/bash
# Allocate a 3-node GB200 Ray cluster using the nemo-rl nightly-063026 gymvenvs image.
# Minimum allocation for nano V3.5 SWE scale-gen (NUM_VLLM_REPLICAS=1: 2 train + 1 gen).
#
# Usage (from ~/RL or anywhere):
#   bash g3nv.sh
#
# Once the cluster is up, attach with:
#   bash <LOGDIR>/slurm-<JOBID>-attach.sh   (printed below)
#
# Then launch training from the head node:
#   NUM_VLLM_REPLICAS=1 bash examples/swe_bench/run_grpo_nano_v3_5_swe_scale_gen_hsg.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONTAINER="${CONTAINER:-/lustre/fsw/coreai_comparch_trtllm/erinh/enroot/nemo-rl:nightly-063026-gymvenvs.squashfs}"
RUN_LOG_DIR="${REPO_ROOT}/logs/g3nv"

mkdir -p "${RUN_LOG_DIR}"

SBATCH_OUTPUT=$(
  CONTAINER="${CONTAINER}" \
  MOUNTS="/lustre:/lustre,${REPO_ROOT}:${REPO_ROOT},/dev/fuse:/dev/fuse" \
  GPUS_PER_NODE=4 \
  BASE_LOG_DIR="${RUN_LOG_DIR}" \
  sbatch \
    --nodes=3 \
    --account=coreai_comparch_trtllm \
    --partition=gb200 \
    --job-name=g3nv \
    --exclusive \
    --mem=0 \
    --time=4:0:0 \
    --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"data_loading","description":"nano V3.5 SWE scale-gen Ray cluster (3-node idle)"}}' \
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
