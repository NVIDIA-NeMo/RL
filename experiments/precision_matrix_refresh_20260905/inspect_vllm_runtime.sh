#!/usr/bin/env bash
set -euo pipefail
: "${REPO:?}"
: "${CONTAINER:?}"
: "${RESULT_DIR:?}"
srun --ntasks=1 --kill-on-bad-exit=1 --wait=30 \
  --no-container-mount-home --container-image="${CONTAINER}" \
  --container-mounts="${REPO}:/source:ro,${RESULT_DIR}:/results,/raid/scratch:/raid/scratch" \
  --output="${RESULT_DIR}/inspect-%N.log" \
  timeout --signal=TERM --kill-after=30s 5m bash -c '
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd /source
PYTHON=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
test -x "$PYTHON"
"$PYTHON" /source/experiments/precision_matrix_refresh_20260905/inspect_vllm_runtime.py /results/runtime
if [[ "${PROBE_LOCAL_SHARD:-0}" == 1 ]]; then
  "$PYTHON" /source/experiments/precision_matrix_refresh_20260905/probe_vllm_tp_loader.py > /results/tp-loader-probe.json
fi
if [[ "${TEST_LOCAL_EXPERT_RELOAD:-0}" == 1 ]]; then
  TEST_DEPS=/raid/scratch/${SLURM_JOB_USER}/local-expert-tests/${SLURM_JOB_ID}
  mkdir -p "$TEST_DEPS"
  UV_CACHE_DIR="$TEST_DEPS/cache" uv pip install --python "$PYTHON" \
    --target "$TEST_DEPS/packages" "pytest==8.4.2"
  "$PYTHON" -c "import nemo_rl; print(nemo_rl.__file__)"
  PYTHONPATH="$TEST_DEPS/packages:/source" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest \
    -p no:cacheprovider -o addopts= --confcutdir=/source/tests/unit/models/generation \
    /source/tests/unit/models/generation/test_local_expert_reload.py \
    /source/tests/unit/models/generation/test_nccl_reshard_backend.py \
    --junitxml=/results/local-expert-reload.xml -v > /results/local-expert-reload.log 2>&1
fi
'
