#!/bin/bash
# Run 14: /dev/fuse bind-mounted in g3nv.sh → apptainer can mount SIF containers.
# All RC1-4 fixes already in place (gym venvs, NEMO_GYM_VENV_DIR_OVERRIDE, swebench_verified_lyris_sif.jsonl,
# sitecustomize.py co_consts pre-patch for workspace_root).
#
# Pre-requisite: submit a new allocation via g3nv.sh (which now includes /dev/fuse:/dev/fuse)
# and wait for Ray head to be ready, then run this script.
#
# Usage:
#   bash g3nv.sh          # allocates nodes; prints JOBID and head node
#   bash launch_run14.sh  # (once Ray head is up)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_ID=$(cat "${REPO_ROOT}/latest_job_id.txt" 2>/dev/null || echo "")
HEAD_NODE=$(grep "container-name=ray-head" /lustre/fsw/coreai_comparch_trtllm/erinh/RL/logs/g3nv/slurm-${JOB_ID}.out 2>/dev/null | grep -o '\-w lyris[0-9]*' | grep -o 'lyris[0-9]*' | head -1)

if [[ -z "${JOB_ID}" ]]; then
    echo "ERROR: No job ID found. Run g3nv.sh first." >&2
    exit 1
fi

LOGFILE="${REPO_ROOT}/run14_$(date +%Y%m%d_%H%M%S).log"
echo "LOGFILE=${LOGFILE}"
echo "JOB_ID=${JOB_ID}  HEAD=${HEAD_NODE}"

# Delete stale sitecustomize .pyc cache before starting (Lustre metadata staleness protection)
PYC=/lustre/fsw/coreai_comparch_trtllm/erinh/gym_venvs/responses_api_agents/swe_agents/.venv/lib/python3.13/site-packages/__pycache__/sitecustomize.cpython-313.pyc
if [[ -f "${PYC}" ]]; then
    rm -f "${PYC}"
    echo "Deleted stale sitecustomize .pyc"
fi

nohup srun \
  --no-container-mount-home \
  -A coreai_comparch_trtllm \
  -p gb200 \
  --overlap \
  --container-name=ray-head \
  --container-workdir=/lustre/fsw/coreai_comparch_trtllm/erinh/RL \
  --nodes=1 --ntasks=1 \
  -w "${HEAD_NODE}" \
  --jobid "${JOB_ID}" \
  bash -c 'NUM_VLLM_REPLICAS=1 bash /lustre/fsw/coreai_comparch_trtllm/erinh/nemo-rl-for-gen/examples/swe_bench/run_grpo_nano_v3_5_swe_scale_gen_hsg_vllm.sh 2>&1' \
  > "${LOGFILE}" 2>&1 &
echo "Run14 PID: $!"
