#!/usr/bin/env bash
set -euo pipefail
: "${CONTAINER:?}"
: "${RESULT_DIR:?}"
: "${VLLM_PADDING_SOURCE:?}"
PACKAGE=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/lib/python3.13/site-packages/vllm
MOUNTS="${RESULT_DIR}:/results,/raid/scratch:/raid/scratch"
MOUNTS+=",${VLLM_PADDING_SOURCE}/tests/kernels/moe/test_flashinfer_weight_alignment.py:/tests/test_alignment.py:ro"
for file in model_executor/layers/quantization/utils/flashinfer_utils.py \
  model_executor/layers/fused_moe/oracle/unquantized.py; do
  MOUNTS+=",${VLLM_PADDING_SOURCE}/vllm/${file}:${PACKAGE}/${file}:ro"
done
srun --ntasks=1 --kill-on-bad-exit=1 --wait=30 \
  --no-container-mount-home --container-image="${CONTAINER}" \
  --container-mounts="${MOUNTS}" --output="${RESULT_DIR}/test-%N.log" \
  timeout --signal=TERM --kill-after=30s 5m bash -s <<'CONTAINER_SCRIPT'
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
LOCAL_ROOT=/raid/scratch/${SLURM_JOB_USER}/gated-alignment/${SLURM_JOB_ID}
export PYTHONPYCACHEPREFIX=$LOCAL_ROOT/pycache
export UV_CACHE_DIR=$LOCAL_ROOT/cache
mkdir -p "$LOCAL_ROOT"
PY=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
uv pip install --python "$PY" --target "$LOCAL_ROOT/packages" pytest==8.4.2
export PYTHONPATH=$LOCAL_ROOT/packages
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
cd /tmp
"$PY" - <<'PY'
import hashlib
import inspect
import sys
from pathlib import Path
from vllm.model_executor.layers.fused_moe.oracle.unquantized import align_moe_weights_for_fi

source = inspect.getsource(align_moe_weights_for_fi)
path = inspect.getsourcefile(align_moe_weights_for_fi)
print("alignment_source_path:", path, flush=True)
print("alignment_source_sha256:", hashlib.sha256(Path(path).read_bytes()).hexdigest(), flush=True)
print("pycache_prefix:", sys.pycache_prefix, flush=True)
print(source, flush=True)
PY
"$PY" -m pytest -p no:cacheprovider -o addopts= --confcutdir=/tests \
  /tests/test_alignment.py -v --junitxml=/results/alignment.xml
CONTAINER_SCRIPT
