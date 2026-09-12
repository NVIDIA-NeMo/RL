#!/usr/bin/env bash
set -euo pipefail
: "${REPO:?}"
: "${CONTAINER:?}"
: "${RESULT_DIR:?}"
srun --ntasks=1 --kill-on-bad-exit=1 --wait=30 \
  --no-container-mount-home --container-image="${CONTAINER}" \
  --container-mounts="${REPO}:/source:ro,${RESULT_DIR}:/results" \
  --output="${RESULT_DIR}/inspect-%N.log" \
  timeout --signal=TERM --kill-after=30s 5m bash -c '
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
PYTHON=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
test -x "$PYTHON"
"$PYTHON" /source/experiments/precision_matrix_refresh_20260905/inspect_vllm_runtime.py /results/runtime
'
