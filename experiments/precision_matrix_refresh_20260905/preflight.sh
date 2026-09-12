#!/usr/bin/env bash
set -euo pipefail
: "${REPO:?}"
: "${CONTAINER:?}"
: "${RESULT_DIR:?}"
srun --ntasks=1 --kill-on-bad-exit=1 --wait=30 \
  --no-container-mount-home --container-image="${CONTAINER}" \
  --container-mounts="${REPO}:/source:ro,${RESULT_DIR}:/results,/raid/scratch:/raid/scratch" \
  --output="${RESULT_DIR}/preflight-%N.log" \
  timeout --signal=TERM --kill-after=30s 10m bash -c '
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/source
export XDG_CACHE_HOME=/raid/scratch/${SLURM_JOB_USER:-sna}/preflight-${SLURM_JOB_ID}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd /source
/opt/nemo_rl_venv/bin/python experiments/precision_matrix_refresh_20260905/preflight.py
'
