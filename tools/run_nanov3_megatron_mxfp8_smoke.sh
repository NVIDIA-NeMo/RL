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

# Run a targeted Nano-v3 smoke on four GPUs of the current interactive node:
# two GPUs hold the BF16 source policy and two run dedicated MXFP8 Megatron
# Inference workers. IMAGE may be either a SIF file or an Apptainer sandbox.

set -euo pipefail

readonly SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/..")
readonly PRECISION=${PRECISION:-mxfp8}
readonly SOURCE_PRECISION=${SOURCE_PRECISION:-bf16}
readonly CONFIG_PATH=${CONFIG_PATH:-examples/configs/recipes/llm/grpo-nanov3-30ba3b-1n4g-megatron-mxfp8-smoke.yaml}
readonly REFIT_ITERATIONS=${REFIT_ITERATIONS:-1}
readonly PERF_WARMUP_ITERATIONS=${PERF_WARMUP_ITERATIONS:-3}
readonly PERF_ITERATIONS=${PERF_ITERATIONS:-10}
readonly PERF_BATCH_SIZE=${PERF_BATCH_SIZE:-4}
readonly INPUT_JSONL=${INPUT_JSONL:-}
readonly INPUT_ROW=${INPUT_ROW:-0}
readonly INPUT_REPEATS=${INPUT_REPEATS:-1}
readonly MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-}
readonly LOGPROB_SAMPLE_MODE=${LOGPROB_SAMPLE_MODE:-greedy}
readonly EXECUTION_MODE=${EXECUTION_MODE:-cuda-graphs}
readonly SKIP_PERF=${SKIP_PERF:-false}
readonly REUSE_SMOKE_FOR_LOGPROB=${REUSE_SMOKE_FOR_LOGPROB:-false}
readonly CHECK_INFERENCE_BATCH_VARIANCE=${CHECK_INFERENCE_BATCH_VARIANCE:-false}
# The inference-optimized BF16 control itself measured 1.057. Use the accepted
# 1.10 aggregate cross-backend envelope while continuing to report the max tail.
readonly MAX_PROB_MULT_ERROR=${MAX_PROB_MULT_ERROR:-1.10}
readonly MEGATRON_BRIDGE_DIR="${PROJECT_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
readonly MEGATRON_LM_DIR="${MEGATRON_BRIDGE_DIR}/3rdparty/Megatron-LM"
# Keep the default image on the same Lustre mount as the checkout so a fresh
# interactive node can start immediately even when sibling user paths are not
# mounted. Callers can still override IMAGE with another SIF or sandbox.
readonly IMAGE=${IMAGE:-${PROJECT_ROOT}/artifacts/containers/nemo-rl-v0.7.0.sif}
readonly RUN_ID=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
readonly LOG_DIR=${LOG_DIR:-${PROJECT_ROOT}/results/nanov3-megatron-mxfp8-smoke/${RUN_ID}}
readonly RUN_LOG=${RUN_LOG:-${LOG_DIR}/run.log}
# Ray appends a long session/socket suffix. Keep its root independent of the
# descriptive run ID so AF_UNIX paths stay below the platform's 107-byte limit.
readonly RAY_LOG_DIR=${RAY_LOG_DIR:-/tmp/nrl-ray-$$}
readonly PYTHON_CACHE_DIR=${PYTHON_CACHE_DIR:-/tmp/nrl-python-cache-nemo-rl-v0.7.0}
readonly XDG_CACHE_DIR=${XDG_CACHE_DIR:-/tmp/nrl-xdg-cache-nemo-rl-v0.7.0}
readonly XDG_CONFIG_DIR=${XDG_CONFIG_DIR:-/tmp/nrl-xdg-config-nemo-rl-v0.7.0}

if [[ ! -e "${IMAGE}" ]]; then
  echo "ERROR: Apptainer image or sandbox not found: ${IMAGE}" >&2
  exit 1
fi
if [[ "$(git -C "${MEGATRON_LM_DIR}" branch --show-current)" != "main" ]]; then
  echo "ERROR: Megatron-LM must be on branch main." >&2
  exit 1
fi
if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -lt 4 ]]; then
  echo "ERROR: the interactive node must expose at least four GPUs." >&2
  exit 1
fi

mkdir -p \
  "${LOG_DIR}" \
  "${RAY_LOG_DIR}" \
  "${PYTHON_CACHE_DIR}" \
  "${XDG_CACHE_DIR}" \
  "${XDG_CONFIG_DIR}" \
  /tmp/nrl-container-home \
  /tmp/nrl-flashinfer
cd "${PROJECT_ROOT}"

export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export APPTAINERENV_FLASHINFER_WORKSPACE_BASE=/tmp/nrl-flashinfer
export APPTAINERENV_HF_HOME=${HF_HOME:?HF_HOME must point to the model cache}
# Megatron parses this environment value with int(); 20 is logging.INFO.
export APPTAINERENV_MEGATRON_LOGGING_LEVEL=20
export APPTAINERENV_NRL_IGNORE_VERSION_MISMATCH=1
if [[ "${PRECISION}" == "mxfp8" ]]; then
  export APPTAINERENV_NRL_VERIFY_MEGATRON_MXFP8=1
elif [[ "${PRECISION}" != "bf16" ]]; then
  echo "ERROR: PRECISION must be mxfp8 or bf16, got ${PRECISION}." >&2
  exit 1
fi
if [[ "${SOURCE_PRECISION}" != "mxfp8" && "${SOURCE_PRECISION}" != "bf16" ]]; then
  echo "ERROR: SOURCE_PRECISION must be mxfp8 or bf16, got ${SOURCE_PRECISION}." >&2
  exit 1
fi
export APPTAINERENV_MPLCONFIGDIR=${XDG_CONFIG_DIR}/matplotlib
export APPTAINERENV_PYTHONPATH="/workspace/RL:/workspace/RL/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:/workspace/RL/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
export APPTAINERENV_PYTHONPYCACHEPREFIX=${PYTHON_CACHE_DIR}
export APPTAINERENV_RAY_worker_niceness=0
export APPTAINERENV_UV_CACHE_DIR=/tmp/nrl-uv-cache
export APPTAINERENV_XDG_CACHE_HOME=${XDG_CACHE_DIR}
export APPTAINERENV_XDG_CONFIG_HOME=${XDG_CONFIG_DIR}

smoke_args=(
  --config "${CONFIG_PATH}"
  --precision "${PRECISION}"
  --source-precision "${SOURCE_PRECISION}"
  --ray-log-dir "${RAY_LOG_DIR}"
  --refit-iterations "${REFIT_ITERATIONS}"
  --perf-batch-size "${PERF_BATCH_SIZE}"
  --perf-warmup-iterations "${PERF_WARMUP_ITERATIONS}"
  --perf-iterations "${PERF_ITERATIONS}"
  --logprob-sample-mode "${LOGPROB_SAMPLE_MODE}"
  --execution-mode "${EXECUTION_MODE}"
  --max-prob-mult-error "${MAX_PROB_MULT_ERROR}"
)
if [[ -n "${INPUT_JSONL}" ]]; then
  smoke_args+=(
    --input-jsonl "${INPUT_JSONL}"
    --input-row "${INPUT_ROW}"
    --input-repeats "${INPUT_REPEATS}"
  )
fi
if [[ -n "${MAX_NEW_TOKENS}" ]]; then
  smoke_args+=(--max-new-tokens "${MAX_NEW_TOKENS}")
fi
if [[ "${SKIP_PERF}" == "true" ]]; then
  smoke_args+=(--skip-perf)
fi
if [[ "${REUSE_SMOKE_FOR_LOGPROB}" == "true" ]]; then
  smoke_args+=(--reuse-smoke-for-logprob)
fi
if [[ "${CHECK_INFERENCE_BATCH_VARIANCE}" == "true" ]]; then
  smoke_args+=(--check-inference-batch-variance)
fi

apptainer exec --no-home --nv \
  --bind "${PROJECT_ROOT}:/workspace/RL" \
  --bind "${HF_HOME}:${HF_HOME}" \
  --pwd /workspace/RL \
  "${IMAGE}" \
  env HOME=/tmp/nrl-container-home UV_CACHE_DIR=/tmp/nrl-uv-cache \
  uv run --no-project --python /opt/nemo_rl_venv/bin/python \
  tools/nanov3_megatron_mxfp8_inference_smoke.py "${smoke_args[@]}" \
  2>&1 | tee "${RUN_LOG}"

readonly PRECISION_MARKER=${PRECISION^^}
if [[ "${PRECISION}" == "mxfp8" ]]; then
  grep -q "NRL_MXFP8_VERIFY: PASS" "${RUN_LOG}"
fi
grep -q "NRL_NANOV3_MEGATRON_${PRECISION_MARKER}_SMOKE: PASS" "${RUN_LOG}"
grep -q "NRL_NANOV3_SOURCE_${SOURCE_PRECISION^^}_VERIFY: PASS" "${RUN_LOG}"
if [[ "${SKIP_PERF}" != "true" ]]; then
  grep -q "NRL_NANOV3_MEGATRON_${PRECISION_MARKER}_PERF: PASS" "${RUN_LOG}"
fi
grep -q "NRL_NANOV3_MEGATRON_${PRECISION_MARKER}_LOGPROB: PASS" "${RUN_LOG}"
if [[ "${EXECUTION_MODE}" == "cuda-graphs" ]]; then
  grep -q "NRL_NANOV3_MEGATRON_${PRECISION_MARKER}_CUDA_GRAPH: PASS" "${RUN_LOG}"
else
  grep -q "NRL_NANOV3_MEGATRON_${PRECISION_MARKER}_EAGER: PASS" "${RUN_LOG}"
fi
echo "PASS: Nano-v3 used ${PRECISION_MARKER} Megatron Inference. Log: ${RUN_LOG}"
