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

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16}"
VLLM_TP="${VLLM_TP:-4}"
if ! [[ "${VLLM_TP}" =~ ^[1-9][0-9]*$ ]] || (( 8 % VLLM_TP != 0 )); then
  echo "ERROR: VLLM_TP must be a positive divisor of 8 (got ${VLLM_TP})." >&2
  exit 1
fi
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/users/ruit/enroot-images/nemo-rl:nightly-071526.squashfs}"

_cache_has_entries() {
  local cache_dir="$1"
  find "${cache_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null \
    | grep -q .
}

_seed_cache() {
  local persistent_dir="$1"
  local local_dir="$2"
  local cache_name="$3"

  mkdir -p "${persistent_dir}" "${local_dir}"
  if _cache_has_entries "${persistent_dir}"; then
    rsync -a --exclude '.tmp*' "${persistent_dir}/" "${local_dir}/"
    echo "[CACHE] ${cache_name}: seeded"
  else
    echo "[CACHE] ${cache_name}: cold"
  fi
}

_write_back_cache() {
  local local_dir="$1"
  local persistent_dir="$2"
  local cache_name="$3"

  mkdir -p "${persistent_dir}"
  (
    flock 9
    rsync -a --exclude '.tmp*' "${local_dir}/" "${persistent_dir}/"
  ) 9>"${persistent_dir}.lock"
  echo "[CACHE] ${cache_name}: persisted"
}

_run_worker() {
  : "${HF_HOME:?HF_HOME must point to a mounted persistent cache}"
  : "${BENCH_OUTPUT_JSON:?}"
  : "${BENCH_BUILD_CACHE_ROOT:?}"
  : "${BENCH_RUNTIME_PYTHON:?}"
  : "${BENCH_CANDIDATE_CHUNK_TOKENS:?}"
  : "${BENCH_CANDIDATE_COUNTS:?}"
  : "${BENCH_STABLE_FIRST_CANDIDATE:?}"

  if [[ ! -x "${BENCH_RUNTIME_PYTHON}" ]]; then
    echo "[PREFLIGHT] missing compatible runtime: ${BENCH_RUNTIME_PYTHON}" >&2
    return 1
  fi
  local tensor_parallel_size="${VLLM_TP}"
  unset VLLM_TP

  BENCH_LOCAL_CACHE_ROOT="/tmp/nemo_rl_streaming_prefill/${BENCH_CACHE_NAMESPACE}"
  BENCH_LOCAL_VLLM_CACHE="${BENCH_LOCAL_CACHE_ROOT}/vllm"
  BENCH_LOCAL_INDUCTOR_CACHE="${BENCH_LOCAL_CACHE_ROOT}/inductor"
  BENCH_LOCAL_TRITON_CACHE="${BENCH_LOCAL_CACHE_ROOT}/triton"
  BENCH_PERSISTENT_VLLM_CACHE="${BENCH_BUILD_CACHE_ROOT}/vllm"
  BENCH_PERSISTENT_INDUCTOR_CACHE="${BENCH_BUILD_CACHE_ROOT}/inductor"
  BENCH_PERSISTENT_TRITON_CACHE="${BENCH_BUILD_CACHE_ROOT}/triton"
  _seed_cache "${BENCH_PERSISTENT_VLLM_CACHE}" \
    "${BENCH_LOCAL_VLLM_CACHE}" "vLLM"
  _seed_cache "${BENCH_PERSISTENT_INDUCTOR_CACHE}" \
    "${BENCH_LOCAL_INDUCTOR_CACHE}" \
    "TorchInductor"
  _seed_cache "${BENCH_PERSISTENT_TRITON_CACHE}" \
    "${BENCH_LOCAL_TRITON_CACHE}" "Triton"

  _sync_build_caches() {
    _write_back_cache "${BENCH_LOCAL_VLLM_CACHE}" \
      "${BENCH_PERSISTENT_VLLM_CACHE}" "vLLM"
    _write_back_cache "${BENCH_LOCAL_INDUCTOR_CACHE}" \
      "${BENCH_PERSISTENT_INDUCTOR_CACHE}" "TorchInductor"
    _write_back_cache "${BENCH_LOCAL_TRITON_CACHE}" \
      "${BENCH_PERSISTENT_TRITON_CACHE}" "Triton"
  }
  trap _sync_build_caches EXIT

  export PYTHONPATH="${REPO_ROOT}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
  unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
  export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
  export NRL_IGNORE_VERSION_MISMATCH=1
  export NRL_VLLM_USE_V1=1
  export VLLM_CACHE_ROOT="${BENCH_LOCAL_VLLM_CACHE}"
  export DG_JIT_CACHE_DIR="${BENCH_LOCAL_VLLM_CACHE}/deep_gemm"
  export INDUCTOR_CACHE_DIR="${BENCH_LOCAL_INDUCTOR_CACHE}"
  export TORCHINDUCTOR_CACHE_DIR="${BENCH_LOCAL_INDUCTOR_CACHE}"
  export TRITON_CACHE_DIR="${BENCH_LOCAL_TRITON_CACHE}"

  echo "[CACHE] runtime=${BENCH_RUNTIME_PYTHON}"
  echo "[CACHE] HF: persistent"
  echo "[CACHE] namespace: ${BENCH_CACHE_NAMESPACE}"
  echo "[PREFLIGHT] validating prewarmed vLLM environment"
  "${BENCH_RUNTIME_PYTHON}" -c 'import torch, transformers, vllm; from nemo_rl.models.generation.vllm.streaming_tool_call import StreamingToolCallPrefillManager; assert torch.cuda.is_available(); assert vllm.__version__ == "0.20.0"; print(torch.__version__, transformers.__version__, vllm.__version__, StreamingToolCallPrefillManager.__name__)'
  "${BENCH_RUNTIME_PYTHON}" -m py_compile \
    examples/swe_bench/benchmark_streaming_tool_call_prefill.py
  if [[ "${BENCH_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[PREFLIGHT] complete"
    return
  fi

  echo "[BENCHMARK] ${BENCH_OUTPUT_JSON}"
  echo "[BENCHMARK] candidate_chunk_tokens=${BENCH_CANDIDATE_CHUNK_TOKENS}"
  echo "[BENCHMARK] candidate_counts=${BENCH_CANDIDATE_COUNTS}"
  echo "[BENCHMARK] stable_first_candidate=${BENCH_STABLE_FIRST_CANDIDATE}"
  local stable_first_candidate_flag
  if [[ "${BENCH_STABLE_FIRST_CANDIDATE}" == "1" ]]; then
    stable_first_candidate_flag="--stable-first-candidate"
  elif [[ "${BENCH_STABLE_FIRST_CANDIDATE}" == "0" ]]; then
    stable_first_candidate_flag="--no-stable-first-candidate"
  else
    echo "BENCH_STABLE_FIRST_CANDIDATE must be 0 or 1" >&2
    return 2
  fi
  "${BENCH_RUNTIME_PYTHON}" \
    examples/swe_bench/benchmark_streaming_tool_call_prefill.py \
    --model "${MODEL_PATH}" \
    --tensor-parallel-size "${tensor_parallel_size}" \
    --gpu-memory-utilization 0.7 \
    --max-model-len 131072 \
    --immutable-prefix-tokens 32768 \
    --tool-output-tokens 8192 \
    --mutable-suffix-tokens 32 \
    --candidate-chunk-tokens "${BENCH_CANDIDATE_CHUNK_TOKENS}" \
    --candidate-counts "${BENCH_CANDIDATE_COUNTS}" \
    "${stable_first_candidate_flag}" \
    --stability-margin-tokens 8 \
    --cleanup-delay-seconds 0.05 \
    --warmup-repeats "${BENCH_WARMUP_REPEATS}" \
    --repeats "${BENCH_REPEATS}" \
    --output "${BENCH_OUTPUT_JSON}"
}

if [[ "${1:-}" == "--worker" ]]; then
  _run_worker
  exit 0
fi

: "${HF_HOME:?Export HF_HOME before launching the benchmark}"

ACCOUNT="${ACCOUNT:-nemotron_sw_post}"
PARTITION="${PARTITION:-interactive,batch}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/streaming_tool_call_prefill_benchmark}"
BENCH_REPEATS="${BENCH_REPEATS:-5}"
BENCH_WARMUP_REPEATS="${BENCH_WARMUP_REPEATS:-1}"
BENCH_CANDIDATE_CHUNK_TOKENS="${BENCH_CANDIDATE_CHUNK_TOKENS:-1024}"
BENCH_CANDIDATE_COUNTS="${BENCH_CANDIDATE_COUNTS:-1,2,4,all}"
BENCH_STABLE_FIRST_CANDIDATE="${BENCH_STABLE_FIRST_CANDIDATE:-0}"
GPU_ARCH="${GPU_ARCH:-h100-sm90}"
BENCH_CACHE_LAYOUT_VERSION="${BENCH_CACHE_LAYOUT_VERSION:-stablelocalv1}"

lock_hash="$(sha256sum "${REPO_ROOT}/uv.lock" | cut -c1-12)"
container_hash="$(printf '%s' "${CONTAINER}" | sha256sum | cut -c1-12)"
model_key="$(printf '%s' "$(basename "${MODEL_PATH}")" | tr -c '[:alnum:]_.-' '_')"
BENCH_CACHE_NAMESPACE="${BENCH_CACHE_NAMESPACE:-${GPU_ARCH}-${container_hash}-${lock_hash}-${model_key}-tp${VLLM_TP}-${BENCH_CACHE_LAYOUT_VERSION}}"
BENCH_CACHE_ROOT="${BENCH_CACHE_ROOT:-${REPO_ROOT}/results/cache/streaming_tool_call_prefill/${BENCH_CACHE_NAMESPACE}}"
BENCH_BUILD_CACHE_ROOT="${BENCH_BUILD_CACHE_ROOT:-${BENCH_CACHE_ROOT}/build}"
runtime_container_key="$(printf '%s' "$(basename "${CONTAINER}")" | tr -c '[:alnum:]_.-' '_')"
BENCH_RUNTIME_NAMESPACE="${BENCH_RUNTIME_NAMESPACE:-${model_key}-${runtime_container_key}-${lock_hash}}"
BENCH_RUNTIME_ROOT="${BENCH_RUNTIME_ROOT:-${REPO_ROOT}/results/cache/swe_scale/${BENCH_RUNTIME_NAMESPACE}}"
BENCH_RUNTIME_PYTHON="${BENCH_RUNTIME_PYTHON:-${BENCH_RUNTIME_ROOT}/ray_venvs/vllm_shared/bin/python}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
BENCH_OUTPUT_JSON="${BENCH_OUTPUT_JSON:-${RESULT_ROOT}/fixed-prompt-prefill-${timestamp}.json}"
mkdir -p "${RESULT_ROOT}" "${BENCH_BUILD_CACHE_ROOT}"

export REPO_ROOT MODEL_PATH VLLM_TP
export BENCH_REPEATS BENCH_WARMUP_REPEATS BENCH_CACHE_NAMESPACE
export BENCH_CANDIDATE_CHUNK_TOKENS BENCH_CANDIDATE_COUNTS
export BENCH_STABLE_FIRST_CANDIDATE
export BENCH_OUTPUT_JSON BENCH_BUILD_CACHE_ROOT BENCH_RUNTIME_PYTHON
export BENCH_PREFLIGHT_ONLY="${BENCH_PREFLIGHT_ONLY:-0}"

echo "[LAUNCH] account=${ACCOUNT} partition=${PARTITION} nodes=1 gpus_per_node=${VLLM_TP} exclusive=false"
echo "[LAUNCH] cache_namespace=${BENCH_CACHE_NAMESPACE}"
echo "[LAUNCH] runtime_namespace=${BENCH_RUNTIME_NAMESPACE}"
echo "[LAUNCH] output=${BENCH_OUTPUT_JSON}"

srun \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node="${VLLM_TP}" \
  --time="${TIME_LIMIT}" \
  --job-name=prefill-fixed \
  --output="${RESULT_ROOT}/prefill-fixed-%j.out" \
  --error="${RESULT_ROOT}/prefill-fixed-%j.err" \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="${REPO_ROOT}" \
  /usr/bin/env bash "${BASH_SOURCE[0]}" --worker
