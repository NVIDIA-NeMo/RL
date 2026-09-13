#!/usr/bin/env bash
set -euo pipefail
: "${REPO:?}"
: "${CONTAINER:?}"
: "${RESULT_DIR:?}"
srun --ntasks=1 --kill-on-bad-exit=1 --wait=30 \
  --no-container-mount-home --container-image="${CONTAINER}" \
  --container-mounts="${REPO}:/source:ro,${RESULT_DIR}:/results,/raid/scratch:/raid/scratch" \
  --output="${RESULT_DIR}/test-%N.log" \
  timeout --signal=TERM --kill-after=30s 10m bash -c '
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd /source
PYTHON=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
TEST_DEPS=/raid/scratch/${SLURM_JOB_USER}/cache-diagnostic-tests/${SLURM_JOB_ID}
mkdir -p "$TEST_DEPS"
UV_CACHE_DIR="$TEST_DEPS/cache" uv pip install --python "$PYTHON" \
  --target "$TEST_DEPS/packages" "pytest==8.4.2"
PYTHONPATH="$TEST_DEPS/packages:/source" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest \
  -p no:cacheprovider -o addopts= --confcutdir=/source/tests/unit/models/policy \
  /source/tests/unit/models/policy/test_megatron_worker.py \
  -k "cache_release_memory_diagnostics or clear_rope_and_moe_dispatcher_caches" \
  --junitxml=/results/cache-diagnostics.xml -v
'
