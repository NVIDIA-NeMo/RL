#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_TAR:?}"
: "${BRIDGE_TAR:?}"
: "${CONTAINER:?}"
: "${RESULT_DIR:?}"
srun --ntasks=1 --kill-on-bad-exit=1 --wait=30 \
  --no-container-mount-home --container-writable \
  --container-image="${CONTAINER}" \
  --container-mounts="${SOURCE_TAR}:/source.tar:ro,${BRIDGE_TAR}:/bridge.tar:ro,${RESULT_DIR}:/results,/raid/scratch:/raid/scratch" \
  --output="${RESULT_DIR}/node-%N.log" \
  timeout --signal=TERM --kill-after=30s 30m bash -c '
set -euo pipefail
root=/raid/scratch/${SLURM_JOB_USER:-sna}/pr-stack-policy-${SLURM_JOB_ID}
mkdir -p "$root/source" "$root/bridge" "$root/cache" "$root/tmp"
tar -xf /source.tar -C "$root/source"
tar -xf /bridge.tar -C "$root/bridge"
export PYTHONDONTWRITEBYTECODE=1
export UV_CACHE_DIR="$root/cache/uv"
export XDG_CACHE_HOME="$root/cache"
export TRITON_CACHE_DIR="$root/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$root/cache/inductor"
export TMPDIR="$root/tmp"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
unset WANDB_API_KEY HF_TOKEN GITHUB_TOKEN GH_TOKEN
python=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
uv pip install --python "$python" --target "$root/test-deps" --no-deps \
  pytest==9.1.1 iniconfig==2.1.0 pluggy==1.6.0 pygments==2.20.0
export PYTHONPATH="$root/source:$root/bridge/src:$root/test-deps"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
cd "$root/source"
"$python" -c '\''import torch,megatron.bridge; print(torch.__version__,torch.cuda.get_device_name(),megatron.bridge.__file__); assert torch.cuda.get_device_capability()[0] >= 10'\''
"$python" -m pytest --confcutdir=tests/unit/models/policy -q -o addopts="" \
  --junitxml=/results/policy.xml \
  tests/unit/models/policy/test_megatron_worker.py::test_native_mxfp8_export_selection
'
