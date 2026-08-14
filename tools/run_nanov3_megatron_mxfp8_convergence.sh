#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run a short Nano-v3 GRPO stability check on the current four-GPU node.
# NVSHMEM_MAX_CTAS is intentionally supplied only by the caller.

set -euo pipefail

readonly SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/..")
readonly CONFIG_PATH=${CONFIG_PATH:-examples/configs/recipes/llm/grpo-nanov3-30ba3b-1n4g-megatron-mxfp8-convergence.yaml}
# Keep the default image on the same Lustre mount as the checkout so a fresh
# interactive node can start immediately even when sibling user paths are not
# mounted. Callers can still override IMAGE with another SIF or sandbox.
readonly IMAGE=${IMAGE:-${PROJECT_ROOT}/artifacts/containers/nemo-rl-v0.7.0.sif}
readonly RUN_ID=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
readonly LOG_DIR=${LOG_DIR:-${PROJECT_ROOT}/results/nanov3-megatron-mxfp8-convergence-runs/${RUN_ID}}
readonly RUN_LOG=${RUN_LOG:-${LOG_DIR}/run.log}
readonly TRAIN_LOG_BASE=${TRAIN_LOG_BASE:-${LOG_DIR}/training}
readonly EXPECTED_STEPS=${EXPECTED_STEPS:-3}
readonly MAX_AVG_PROB_MULT_ERROR=${MAX_AVG_PROB_MULT_ERROR:-1.10}
# Ray appends a long session/socket suffix. Keep its root independent of the
# descriptive run ID so AF_UNIX paths stay below the platform's 107-byte limit.
readonly RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/nrl-ray-$$}
readonly PYTHON_CACHE_DIR=${PYTHON_CACHE_DIR:-/tmp/nrl-python-cache-nemo-rl-v0.7.0}
readonly XDG_CACHE_DIR=${XDG_CACHE_DIR:-/tmp/nrl-xdg-cache-nemo-rl-v0.7.0}
readonly XDG_CONFIG_DIR=${XDG_CONFIG_DIR:-/tmp/nrl-xdg-config-nemo-rl-v0.7.0}
readonly NRL_JOB_START_EPOCH=${NRL_JOB_START_EPOCH:-$(date +%s)}

if [[ ! -e "${IMAGE}" ]]; then
  echo "ERROR: Apptainer image or sandbox not found: ${IMAGE}" >&2
  exit 1
fi
if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -lt 4 ]]; then
  echo "ERROR: the interactive node must expose at least four GPUs." >&2
  exit 1
fi

mkdir -p \
  "${LOG_DIR}" \
  "${RAY_TEMP_DIR}" \
  "${PYTHON_CACHE_DIR}" \
  "${XDG_CACHE_DIR}" \
  "${XDG_CONFIG_DIR}" \
  /tmp/nrl-container-home \
  /tmp/nrl-flashinfer
cd "${PROJECT_ROOT}"

export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export APPTAINERENV_FLASHINFER_WORKSPACE_BASE=/tmp/nrl-flashinfer
export APPTAINERENV_HF_HOME=${HF_HOME:?HF_HOME must point to the model cache}
export APPTAINERENV_NRL_IGNORE_VERSION_MISMATCH=1
export APPTAINERENV_NRL_JOB_START_EPOCH=${NRL_JOB_START_EPOCH}
export APPTAINERENV_NRL_VERIFY_MEGATRON_MXFP8=1
export APPTAINERENV_MPLCONFIGDIR=${XDG_CONFIG_DIR}/matplotlib
export APPTAINERENV_PYTHONPATH="/workspace/RL:/workspace/RL/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:/workspace/RL/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
export APPTAINERENV_PYTHONPYCACHEPREFIX=${PYTHON_CACHE_DIR}
export APPTAINERENV_RAY_worker_niceness=0
export APPTAINERENV_UV_CACHE_DIR=/tmp/nrl-uv-cache
export APPTAINERENV_XDG_CACHE_HOME=${XDG_CACHE_DIR}
export APPTAINERENV_XDG_CONFIG_HOME=${XDG_CONFIG_DIR}

apptainer exec --no-home --nv \
  --bind "${PROJECT_ROOT}:/workspace/RL" \
  --bind "${HF_HOME}:${HF_HOME}" \
  --pwd /workspace/RL \
  "${IMAGE}" \
  bash -lc "export HOME=/tmp/nrl-container-home; export UV_CACHE_DIR=/tmp/nrl-uv-cache; /opt/nemo_rl_venv/bin/ray start --head --num-cpus=8 --num-gpus=4 --temp-dir ${RAY_TEMP_DIR} --disable-usage-stats --include-dashboard=false; trap '/opt/nemo_rl_venv/bin/ray stop --force' EXIT; uv run --no-project --python /opt/nemo_rl_venv/bin/python examples/run_grpo.py --config ${CONFIG_PATH} logger.log_dir=${TRAIN_LOG_BASE}" \
  2>&1 | tee "${RUN_LOG}"

grep -q "NRL_MXFP8_VERIFY: PASS" "${RUN_LOG}"
apptainer exec --no-home \
  --bind "${PROJECT_ROOT}:/workspace/RL" \
  --pwd /workspace/RL \
  "${IMAGE}" \
  bash -lc "export HOME=/tmp/nrl-container-home; export UV_CACHE_DIR=/tmp/nrl-uv-cache; uv run --no-project --python /opt/nemo_rl_venv/bin/python tools/check_nanov3_mxfp8_convergence.py ${TRAIN_LOG_BASE}/exp_001 --expected-steps ${EXPECTED_STEPS} --max-avg-prob-mult-error ${MAX_AVG_PROB_MULT_ERROR} --require-learning-signal" \
  2>&1 | tee -a "${RUN_LOG}"
echo "PASS: Nano-v3 MXFP8 convergence validation completed. Log: ${RUN_LOG}"
