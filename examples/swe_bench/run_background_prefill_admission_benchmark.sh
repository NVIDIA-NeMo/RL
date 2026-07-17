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
  find "$1" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null | grep -q .
}

_seed_cache() {
  local persistent_dir="$1"
  local local_dir="$2"
  local cache_name="$3"
  mkdir -p "${persistent_dir}" "${local_dir}"
  if _cache_has_entries "${persistent_dir}"; then
    rsync -a --exclude '.tmp*' "${persistent_dir}/" "${local_dir}/"
    echo "[CACHE] ${cache_name}=seeded namespace=${BENCH_CACHE_NAMESPACE}"
  else
    echo "[CACHE] ${cache_name}=cold namespace=${BENCH_CACHE_NAMESPACE}"
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
  echo "[CACHE] ${cache_name}=writeback-complete namespace=${BENCH_CACHE_NAMESPACE}"
}

_run_worker() {
  : "${HF_HOME:?HF_HOME must point to a mounted persistent cache}"
  : "${BENCH_OUTPUT_JSON:?}"
  : "${BENCH_BUILD_CACHE_ROOT:?}"
  : "${BENCH_RUNTIME_PYTHON:?}"

  if [[ ! -x "${BENCH_RUNTIME_PYTHON}" ]]; then
    echo "[PREFLIGHT] missing compatible runtime: ${BENCH_RUNTIME_PYTHON}" >&2
    return 1
  fi
  local tensor_parallel_size="${VLLM_TP}"
  unset VLLM_TP

  local_cache_root="/tmp/nemo_rl_streaming_prefill/${BENCH_CACHE_NAMESPACE}"
  local_vllm_cache="${local_cache_root}/vllm"
  local_inductor_cache="${local_cache_root}/inductor"
  local_triton_cache="${local_cache_root}/triton"
  persistent_vllm_cache="${BENCH_BUILD_CACHE_ROOT}/vllm"
  persistent_inductor_cache="${BENCH_BUILD_CACHE_ROOT}/inductor"
  persistent_triton_cache="${BENCH_BUILD_CACHE_ROOT}/triton"

  _seed_cache "${persistent_vllm_cache}" "${local_vllm_cache}" vllm
  _seed_cache "${persistent_inductor_cache}" "${local_inductor_cache}" inductor
  _seed_cache "${persistent_triton_cache}" "${local_triton_cache}" triton

  _sync_build_caches() {
    _write_back_cache "${local_vllm_cache}" "${persistent_vllm_cache}" vllm
    _write_back_cache "${local_inductor_cache}" "${persistent_inductor_cache}" inductor
    _write_back_cache "${local_triton_cache}" "${persistent_triton_cache}" triton
  }
  trap _sync_build_caches EXIT

  export PYTHONPATH="${REPO_ROOT}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
  unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
  export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
  export NRL_IGNORE_VERSION_MISMATCH=1
  export NRL_VLLM_USE_V1=1
  export VLLM_CACHE_ROOT="${local_vllm_cache}"
  export DG_JIT_CACHE_DIR="${local_vllm_cache}/deep_gemm"
  export INDUCTOR_CACHE_DIR="${local_inductor_cache}"
  export TORCHINDUCTOR_CACHE_DIR="${local_inductor_cache}"
  export TRITON_CACHE_DIR="${local_triton_cache}"

  echo "[CACHE] runtime=${BENCH_RUNTIME_PYTHON} hf=${HF_HOME} hf_datasets=${HF_DATASETS_CACHE} vllm_local=${VLLM_CACHE_ROOT} vllm_persistent=${persistent_vllm_cache} inductor=${INDUCTOR_CACHE_DIR} triton=${TRITON_CACHE_DIR}"
  echo "[PREFLIGHT] validating prewarmed vLLM environment"
  "${BENCH_RUNTIME_PYTHON}" -c 'import torch, transformers, vllm; from examples.swe_bench import benchmark_background_prefill_admission; assert torch.cuda.is_available(); assert vllm.__version__ == "0.20.0"; print(torch.__version__, transformers.__version__, vllm.__version__, benchmark_background_prefill_admission.__file__)'

  size_sweep_args=()
  if [[ -n "${BENCH_CANDIDATE_CHUNK_TOKEN_SWEEP}" ]]; then
    size_sweep_args=(
      --candidate-chunk-token-sweep "${BENCH_CANDIDATE_CHUNK_TOKEN_SWEEP}"
      --size-sweep-overlap-seconds "${BENCH_SIZE_SWEEP_OVERLAP_SECONDS}"
      --size-sweep-repeats "${BENCH_SIZE_SWEEP_REPEATS}"
    )
  fi
  cache_page_args=()
  if (( BENCH_CACHE_PAGE_SIZE_TOKENS > 0 )); then
    cache_page_args=(--cache-page-size-tokens "${BENCH_CACHE_PAGE_SIZE_TOKENS}")
  fi

  "${BENCH_RUNTIME_PYTHON}" \
    examples/swe_bench/benchmark_background_prefill_admission.py \
    --model "${MODEL_PATH}" \
    --tensor-parallel-size "${tensor_parallel_size}" \
    --gpu-memory-utilization 0.7 \
    --max-model-len 131072 \
    --immutable-prefix-tokens 32768 \
    --tool-output-tokens 8192 \
    --mutable-suffix-tokens 32 \
    --candidate-chunk-tokens "${BENCH_CANDIDATE_CHUNK_TOKENS}" \
    --stability-margin-tokens 8 \
    --background-priority "${BENCH_BACKGROUND_PRIORITY}" \
    "${cache_page_args[@]}" \
    --overlap-seconds "${BENCH_OVERLAP_SECONDS}" \
    --cleanup-delay-seconds 0.05 \
    --warmup-repeats "${BENCH_WARMUP_REPEATS}" \
    --repeats "${BENCH_REPEATS}" \
    --contention-repeats "${BENCH_CONTENTION_REPEATS}" \
    --contention-foreground-tokens "${BENCH_CONTENTION_FOREGROUND_TOKENS}" \
    "${size_sweep_args[@]}" \
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
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/streaming_tool_call_background_prefill_benchmark}"
BENCH_REPEATS="${BENCH_REPEATS:-5}"
BENCH_WARMUP_REPEATS="${BENCH_WARMUP_REPEATS:-1}"
BENCH_CANDIDATE_CHUNK_TOKENS="${BENCH_CANDIDATE_CHUNK_TOKENS:-1024}"
BENCH_BACKGROUND_PRIORITY="${BENCH_BACKGROUND_PRIORITY:-1}"
BENCH_CACHE_PAGE_SIZE_TOKENS="${BENCH_CACHE_PAGE_SIZE_TOKENS:-1152}"
if ! [[ "${BENCH_CACHE_PAGE_SIZE_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: BENCH_CACHE_PAGE_SIZE_TOKENS must be a non-negative integer." >&2
  exit 1
fi
BENCH_CONTENTION_REPEATS="${BENCH_CONTENTION_REPEATS:-10}"
BENCH_CONTENTION_FOREGROUND_TOKENS="${BENCH_CONTENTION_FOREGROUND_TOKENS:-2048}"
BENCH_OVERLAP_SECONDS="${BENCH_OVERLAP_SECONDS:-0,0.025,0.05,0.075,0.1,0.15,0.25}"
BENCH_CANDIDATE_CHUNK_TOKEN_SWEEP="${BENCH_CANDIDATE_CHUNK_TOKEN_SWEEP:-}"
BENCH_SIZE_SWEEP_OVERLAP_SECONDS="${BENCH_SIZE_SWEEP_OVERLAP_SECONDS:-0.075}"
BENCH_SIZE_SWEEP_REPEATS="${BENCH_SIZE_SWEEP_REPEATS:-5}"
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
BENCH_OUTPUT_JSON="${BENCH_OUTPUT_JSON:-${RESULT_ROOT}/background-prefill-${timestamp}.json}"
mkdir -p "${RESULT_ROOT}" "${BENCH_BUILD_CACHE_ROOT}"

export REPO_ROOT MODEL_PATH VLLM_TP
export BENCH_REPEATS BENCH_WARMUP_REPEATS BENCH_CACHE_NAMESPACE
export BENCH_BACKGROUND_PRIORITY BENCH_CANDIDATE_CHUNK_TOKENS BENCH_OVERLAP_SECONDS
export BENCH_CACHE_PAGE_SIZE_TOKENS
export BENCH_CONTENTION_REPEATS BENCH_CONTENTION_FOREGROUND_TOKENS
export BENCH_CANDIDATE_CHUNK_TOKEN_SWEEP BENCH_SIZE_SWEEP_OVERLAP_SECONDS
export BENCH_SIZE_SWEEP_REPEATS
export BENCH_OUTPUT_JSON BENCH_BUILD_CACHE_ROOT BENCH_RUNTIME_PYTHON

echo "[LAUNCH] account=${ACCOUNT} partition=${PARTITION} nodes=1 gpus_per_node=${VLLM_TP} exclusive=false"
echo "[LAUNCH] cache_namespace=${BENCH_CACHE_NAMESPACE}"
echo "[LAUNCH] runtime_namespace=${BENCH_RUNTIME_NAMESPACE}"
echo "[LAUNCH] output=${BENCH_OUTPUT_JSON}"
echo "[LAUNCH] candidate_chunk_token_sweep=${BENCH_CANDIDATE_CHUNK_TOKEN_SWEEP:-disabled}"
echo "[LAUNCH] cache_page_size_tokens=${BENCH_CACHE_PAGE_SIZE_TOKENS} (0 disables page-aware admission)"

srun \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node="${VLLM_TP}" \
  --time="${TIME_LIMIT}" \
  --job-name=background-prefill \
  --output="${RESULT_ROOT}/background-prefill-%j.out" \
  --error="${RESULT_ROOT}/background-prefill-%j.err" \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="${REPO_ROOT}" \
  /usr/bin/env bash "${BASH_SOURCE[0]}" --worker
