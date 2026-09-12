#!/usr/bin/env bash
set -euo pipefail
: "${BEFORE_TAR:?}"
: "${AFTER_TAR:?}"
: "${CONTAINER:?}"
: "${RESULT_DIR:?}"
srun --ntasks=1 --kill-on-bad-exit=1 --wait=30 \
  --no-container-mount-home --container-writable \
  --container-image="${CONTAINER}" \
  --container-mounts="${BEFORE_TAR}:/before.tar:ro,${AFTER_TAR}:/after.tar:ro,${RESULT_DIR}:/results,/raid/scratch:/raid/scratch" \
  --output="${RESULT_DIR}/node-%N.log" \
  timeout --signal=TERM --kill-after=30s 20m bash -c '
set -euo pipefail
root=/raid/scratch/${SLURM_JOB_USER:-sna}/scope-regression-${SLURM_JOB_ID}
mkdir -p "$root/before" "$root/after" "$root/cache" "$root/tmp"
tar -xf /before.tar -C "$root/before"
tar -xf /after.tar -C "$root/after"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export UV_CACHE_DIR="$root/cache/uv" XDG_CACHE_HOME="$root/cache" TMPDIR="$root/tmp"
export TRITON_CACHE_DIR="$root/cache/triton" TORCHINDUCTOR_CACHE_DIR="$root/cache/inductor"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
uv pip install --python "$python" --target "$root/test-deps" --no-deps pytest==9.1.1 iniconfig==2.1.0 pluggy==1.6.0 pygments==2.20.0
cd "$root/before"
export PYTHONPATH="$root/before:$root/test-deps"
set +e
"$python" -m pytest --confcutdir=tests/unit/models/generation -q -o addopts="" \
  tests/unit/models/generation/test_vllm_fp8_hf_overrides.py \
  -k test_bf16_rollout_can_inherit_ignore_patterns --junitxml=/results/before.xml
rc=$?
set -e
test "$rc" -eq 1
cd "$root/after"
export PYTHONPATH="$root/after:$root/test-deps"
"$python" -m pytest --confcutdir=tests/unit/models/generation -q -o addopts="" \
  --junitxml=/results/after.xml \
  tests/unit/models/generation/test_vllm_fp8_hf_overrides.py \
  tests/unit/models/generation/test_vllm_refit_adapter.py \
  tests/unit/models/generation/test_vllm_fp8_quantization.py \
  tests/unit/models/generation/test_vllm_mxfp8_utils.py \
  tests/unit/models/generation/test_mxfp8_prequant.py \
  tests/unit/models/generation/test_nccl_reshard_backend.py
'
