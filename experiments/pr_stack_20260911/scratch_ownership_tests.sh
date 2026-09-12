#!/usr/bin/env bash
set -euo pipefail
: "${BEFORE_TAR:?}" "${AFTER_TAR:?}" "${CONTAINER:?}" "${RESULT_DIR:?}"
srun --ntasks=1 --kill-on-bad-exit=1 --wait=30 \
  --no-container-mount-home --container-writable \
  --container-image="${CONTAINER}" \
  --container-mounts="${BEFORE_TAR}:/before.tar:ro,${AFTER_TAR}:/after.tar:ro,${RESULT_DIR}:/results,/raid/scratch:/raid/scratch" \
  --output="${RESULT_DIR}/node-%N.log" \
  timeout --signal=TERM --kill-after=30s 20m bash -c '
set -euo pipefail
root=/raid/scratch/${SLURM_JOB_USER:-sna}/scratch-ownership-${SLURM_JOB_ID}
mkdir -p "$root/before" "$root/after" "$root/cache" "$root/tmp"
tar -xf /before.tar -C "$root/before"
tar -xf /after.tar -C "$root/after"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export UV_CACHE_DIR="$root/cache/uv" XDG_CACHE_HOME="$root/cache" TMPDIR="$root/tmp"
unset WANDB_API_KEY HF_TOKEN GITHUB_TOKEN GH_TOKEN
python=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
uv pip install --python "$python" --target "$root/test-deps" --no-deps pytest==9.1.1 iniconfig==2.1.0 pluggy==1.6.0 pygments==2.20.0
export PYTHONPATH="$root/test-deps"
"$python" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
cd "$root/before"
set +e
"$python" -m pytest --confcutdir=tests/unit/models/generation -q -o addopts="" tests/unit/models/generation/test_vllm_mxfp8_utils.py -k runtime_parameter_owns_shared_scratch_storage --junitxml=/results/before.xml
rc=$?
set -e
test "$rc" -eq 1
cd "$root/after"
"$python" -m pytest --confcutdir=tests/unit/models/generation -q -o addopts="" tests/unit/models/generation/test_vllm_mxfp8_utils.py --junitxml=/results/after.xml
'
