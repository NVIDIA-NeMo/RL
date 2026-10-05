#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
source /opt/nemo-rl/docker/nano35/environment.sh
export NANO35_VALIDATION_JOB_ID=${SLURM_JOB_ID:?Run inside the validation allocation}
export NANO35_RUN_NAME="nano35-preflight-$NANO35_VALIDATION_JOB_ID"
export NANO35_RUN_DIR=${NANO35_VALIDATION_ROOT:?Set the output root}
export TRTLLM_REQUIRE_CACHED_WHEEL=1 NRL_FORCE_REBUILD_VENVS=false
export WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1
TASK_TMP=$(mktemp -d "/tmp/nano35-preflight-${NANO35_VALIDATION_JOB_ID}.XXXXXXXX")
export HF_HOME="$TASK_TMP/huggingface" HF_MODULES_CACHE="$TASK_TMP/hf-modules"
export TORCHINDUCTOR_CACHE_DIR="$TASK_TMP/torchinductor" TRITON_CACHE_DIR="$TASK_TMP/triton"
export APPTAINER_CACHEDIR="$TASK_TMP/apptainer-cache" APPTAINER_TMPDIR="$TASK_TMP/apptainer-tmp"
mkdir -p "$HF_MODULES_CACHE" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
# Clear launcher bookkeeping before importing TRT-LLM / mpi4py. Ray owns its
# inference workers; these are not ranks of the surrounding srun MPI world.
for key in "${!PMI_@}" "${!PMIX_@}" "${!MPI_@}" "${!OMPI_@}" "${!SLURM_@}"; do
    [[ -z "$key" ]] || unset "$key"
done
cd /opt/nemo-rl
nvidia-smi -L
python-TrtllmAsyncGenerationWorker - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA is not available in the runtime container'
assert torch.cuda.device_count() == 4, 'TP4 preflight requires four visible GPUs'
print('GPU driver check passed:', torch.cuda.get_device_name(0), flush=True)
PY
set +e
python-TrtllmAsyncGenerationWorker tools/nano35/gpu_preflight.py \
    --output-dir "$NANO35_VALIDATION_ROOT/result" \
    --image-sha256 "${NANO35_IMAGE_SHA256:?Set the exact image digest}"
STATUS=$?
set -e
printf '%s\n' "$STATUS" > "$NANO35_VALIDATION_ROOT/exit-code.txt"
if [[ -d /tmp/ray/session_latest/logs ]]; then
    tar -C /tmp/ray/session_latest -czf "$NANO35_VALIDATION_ROOT/ray-logs.tar.gz" logs
fi
rm -rf -- "$TASK_TMP"
exit "$STATUS"
