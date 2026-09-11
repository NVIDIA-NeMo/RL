#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_TAR:?}"
: "${SOURCE_SHA:?}"
: "${CONTAINER:?}"
: "${RESULT_DIR:?}"
export SOURCE_SHA RESULT_DIR
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
srun --ntasks=1 --kill-on-bad-exit=1 --wait=30 \
  --no-container-mount-home --container-writable \
  --container-image="${CONTAINER}" \
  --container-mounts="${SOURCE_TAR}:/source.tar:ro,${RESULT_DIR}:/results,/raid/scratch:/raid/scratch" \
  --output="${RESULT_DIR}/node-%N.log" \
  timeout --signal=TERM --kill-after=30s 45m bash -c '
set -euo pipefail
root=/raid/scratch/${SLURM_JOB_USER:-sna}/pr-stack-${SLURM_JOB_ID}
mkdir -p "$root/source" "$root/cache" "$root/tmp"
tar -xf /source.tar -C "$root/source"
export PYTHONDONTWRITEBYTECODE=1
export UV_CACHE_DIR="$root/cache/uv"
export XDG_CACHE_HOME="$root/cache"
export TRITON_CACHE_DIR="$root/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$root/cache/inductor"
export TMPDIR="$root/tmp"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
unset WANDB_API_KEY HF_TOKEN GITHUB_TOKEN GH_TOKEN
python=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
uv pip install --python "$python" --target "$root/test-deps" --no-deps \
  pytest==9.1.1 iniconfig==2.1.0 pluggy==1.6.0 pygments==2.20.0
export PYTHONPATH="$root/source:$root/test-deps"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
cd "$root/source"
"$python" -c '\''import torch,vllm; print(torch.__version__,vllm.__version__,torch.cuda.get_device_name()); assert torch.cuda.get_device_capability()[0] >= 10'\''
"$python" -m pytest --confcutdir=tests/unit/models/generation -q -o addopts="" \
  --junitxml=/results/generation.xml \
  tests/unit/models/generation/test_vllm_refit_adapter.py \
  tests/unit/models/generation/test_vllm_fp8_quantization.py \
  tests/unit/models/generation/test_vllm_mxfp8_utils.py \
  tests/unit/models/generation/test_mxfp8_prequant.py \
  tests/unit/models/generation/test_nccl_reshard_backend.py
'
