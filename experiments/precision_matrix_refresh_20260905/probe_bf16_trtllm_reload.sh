#!/usr/bin/env bash
set -euo pipefail
: "${REPO:?}"
: "${CONTAINER:?}"
: "${RESULT_DIR:?}"
MOUNTS="${REPO}:/source:ro,${RESULT_DIR}:/results,/raid/scratch:/raid/scratch"
if [[ -n "${VLLM_PADDING_SOURCE:-}" ]]; then
  PACKAGE=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/lib/python3.13/site-packages/vllm
  for file in model_executor/layers/quantization/utils/flashinfer_utils.py \
    model_executor/layers/fused_moe/oracle/unquantized.py; do
    test -f "${VLLM_PADDING_SOURCE}/vllm/${file}"
    MOUNTS+=",${VLLM_PADDING_SOURCE}/vllm/${file}:${PACKAGE}/${file}:ro"
  done
fi
srun --ntasks=1 --kill-on-bad-exit=1 --wait=30 \
  --no-container-mount-home --container-image="${CONTAINER}" \
  --container-mounts="${MOUNTS}" \
  --output="${RESULT_DIR}/probe-%N.log" \
  timeout --signal=TERM --kill-after=30s 25m bash -c '
set -euo pipefail
cd /source
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/source
LOCAL_ROOT=/raid/scratch/${SLURM_JOB_USER}/trtllm-local-refit-probe-v0251
export XDG_CACHE_HOME=$LOCAL_ROOT/cache
export VLLM_CACHE_ROOT=$LOCAL_ROOT/vllm
export TORCHINDUCTOR_CACHE_DIR=$LOCAL_ROOT/inductor
export TRITON_CACHE_DIR=$LOCAL_ROOT/triton
export OMP_NUM_THREADS=4
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
PYTHON=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
test -x "$PYTHON"
"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 \
  --log-dir=/results/torchrun --redirects=3 --tee=3 \
  /source/experiments/precision_matrix_refresh_20260905/probe_bf16_trtllm_reload.py
'
