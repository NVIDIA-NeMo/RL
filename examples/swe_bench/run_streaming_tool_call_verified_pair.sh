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

# Submit a strict deterministic SWE-bench Verified rollout pair.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LAUNCHER="${REPO_ROOT}/examples/swe_bench/run_grpo_swe2_scale_gen.sh"
AUDITOR="${REPO_ROOT}/examples/swe_bench/verified_trajectory_audit.py"
AUDIT_PYTHON="${AUDIT_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
ARTIFACT_PREWARMER="${REPO_ROOT}/examples/swe_bench/prewarm_swebench_artifacts.sh"
VERIFIED_DATA_PATH="${VERIFIED_DATA_PATH:-${REPO_ROOT}/results/swebench_verified/swebench_verified_500.jsonl}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
PAIR_DIR="${PAIR_DIR:-${REPO_ROOT}/results/streaming_tool_call_verified/${RUN_ID}}"
PAIR_PERSISTENT_CACHE_ROOT="${PAIR_PERSISTENT_CACHE_ROOT:-}"
PAIR_SHARED_PERSISTENT_CACHE="${PAIR_SHARED_PERSISTENT_CACHE:-}"
WANDB_GROUP="${WANDB_GROUP:-streaming-tool-call-verified-temp0-${RUN_ID}}"
SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-}"
NUM_VLLM_REPLICAS="${NUM_VLLM_REPLICAS:-32}"
TRAJECTORY_COLLECTION_BATCH_SIZE="${TRAJECTORY_COLLECTION_BATCH_SIZE:-128}"
EXPECTED_COUNT="${EXPECTED_COUNT:-500}"
PREWARM_SWEBENCH_ARTIFACTS="${PREWARM_SWEBENCH_ARTIFACTS:-1}"
EXACT_INCREMENTAL_TOKENIZER="${EXACT_INCREMENTAL_TOKENIZER:-0}"
FINAL_ONLY_INCREMENTAL_TOKENIZER="${FINAL_ONLY_INCREMENTAL_TOKENIZER:-0}"
FINAL_ONLY_PREFILL="${FINAL_ONLY_PREFILL:-0}"
PREFIX_SEEDED_START="${PREFIX_SEEDED_START:-0}"
PREFILL_AFTER_ADMISSION="${PREFILL_AFTER_ADMISSION:-0}"
BACKGROUND_PREFILL_COMPLETION="${BACKGROUND_PREFILL_COMPLETION:-0}"
STABLE_FIRST_SNAPSHOT_PREFILL="${STABLE_FIRST_SNAPSHOT_PREFILL:-0}"
COMPACT_REQUEST_CONTEXT="${COMPACT_REQUEST_CONTEXT:-0}"
INCREMENTAL_TOKENIZER_CHECKPOINT_INTERVAL="${INCREMENTAL_TOKENIZER_CHECKPOINT_INTERVAL:-8}"
COUNTERFACTUAL_FULL_TOKENIZER_TIMING="${COUNTERFACTUAL_FULL_TOKENIZER_TIMING:-0}"
DETAILED_RUNTIME_METRICS="${DETAILED_RUNTIME_METRICS:-1}"
BASE_CONCURRENCY="${BASE_CONCURRENCY:-${EXPECTED_COUNT}}"
# Each PAIR_ARMS entry accepts
# name:streaming_enabled[:snapshot_poll_seconds[:min_chunk_chars[:tokenizer_only[:final_only_prefill[:prefix_seeded_start[:prefill_after_admission[:compact_request_context[:initial_chunk_chars[:snapshot_long_poll_timeout_seconds[:stable_first_snapshot_prefill[:same_request_final_decode]]]]]]]]]]].
SNAPSHOT_POLL_INTERVAL_SECONDS="${SNAPSHOT_POLL_INTERVAL_SECONDS:-0.1}"
SNAPSHOT_LONG_POLL_TIMEOUT_SECONDS="${SNAPSHOT_LONG_POLL_TIMEOUT_SECONDS:-1.0}"
STREAMING_INITIAL_CHUNK_CHARS="${STREAMING_INITIAL_CHUNK_CHARS:-256}"
PAIR_ARMS="${PAIR_ARMS:-streaming_off:0 streaming_on:1}"
SEQUENTIAL_ARMS="${SEQUENTIAL_ARMS:-0}"

if [ "${SEQUENTIAL_ARMS}" != "0" ] && [ "${SEQUENTIAL_ARMS}" != "1" ]; then
  echo "ERROR: SEQUENTIAL_ARMS must be 0 or 1" >&2
  exit 1
fi

if [ -n "${PAIR_PERSISTENT_CACHE_ROOT}" ] && [ -n "${PAIR_SHARED_PERSISTENT_CACHE}" ]; then
  echo "ERROR: set only one of PAIR_PERSISTENT_CACHE_ROOT (per-arm) or PAIR_SHARED_PERSISTENT_CACHE" >&2
  exit 1
fi

if [ ! -x "${AUDIT_PYTHON}" ]; then
  echo "ERROR: AUDIT_PYTHON is not executable: ${AUDIT_PYTHON}" >&2
  echo "Set AUDIT_PYTHON to an existing Python interpreter; the auditor uses only the standard library." >&2
  exit 1
fi

"${AUDIT_PYTHON}" "${AUDITOR}" verify \
  --manifest "${VERIFIED_DATA_PATH}" \
  --expected-count "${EXPECTED_COUNT}"

if [ "${PREWARM_SWEBENCH_ARTIFACTS}" = "1" ]; then
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[DRY_RUN] Would prewarm and offline-verify SWE-bench artifacts for ${VERIFIED_DATA_PATH}"
  else
    bash "${ARTIFACT_PREWARMER}" "${VERIFIED_DATA_PATH}"
  fi
  SWE_BENCH_ARTIFACT_CACHE_OFFLINE="${SWE_BENCH_ARTIFACT_CACHE_OFFLINE:-1}"
else
  SWE_BENCH_ARTIFACT_CACHE_OFFLINE="${SWE_BENCH_ARTIFACT_CACHE_OFFLINE:-0}"
fi

if [ "${SWE_BENCH_ARTIFACT_CACHE_OFFLINE}" != "0" ] && [ "${SWE_BENCH_ARTIFACT_CACHE_OFFLINE}" != "1" ]; then
  echo "ERROR: SWE_BENCH_ARTIFACT_CACHE_OFFLINE must be 0 or 1" >&2
  exit 1
fi

mkdir -p "${PAIR_DIR}/slurm"

submit_arm() {
  local arm="$1"
  local streaming_enabled="$2"
  local snapshot_poll_interval_seconds="$3"
  local min_chunk_chars="$4"
  local tokenizer_only="$5"
  local requested_final_only_prefill="$6"
  local requested_prefix_seeded_start="$7"
  local requested_prefill_after_admission="$8"
  local requested_compact_request_context="${9:-}"
  local requested_initial_chunk_chars="${10:-}"
  local requested_snapshot_long_poll_timeout_seconds="${11:-}"
  local requested_stable_first_snapshot_prefill="${12:-}"
  local requested_same_request_final_decode="${13:-}"
  local exact_incremental_tokenizer="${EXACT_INCREMENTAL_TOKENIZER}"
  local final_only_incremental_tokenizer="${FINAL_ONLY_INCREMENTAL_TOKENIZER}"
  local final_only_prefill="${requested_final_only_prefill:-${FINAL_ONLY_PREFILL}}"
  local prefix_seeded_start="${requested_prefix_seeded_start:-${PREFIX_SEEDED_START}}"
  local prefill_after_admission="${requested_prefill_after_admission:-${PREFILL_AFTER_ADMISSION}}"
  local background_prefill_completion="${BACKGROUND_PREFILL_COMPLETION}"
  local compact_request_context="${requested_compact_request_context:-${COMPACT_REQUEST_CONTEXT}}"
  local initial_chunk_chars="${requested_initial_chunk_chars:-${STREAMING_INITIAL_CHUNK_CHARS}}"
  local snapshot_long_poll_timeout_seconds="${requested_snapshot_long_poll_timeout_seconds:-${SNAPSHOT_LONG_POLL_TIMEOUT_SECONDS}}"
  local stable_first_snapshot_prefill="${requested_stable_first_snapshot_prefill:-${STABLE_FIRST_SNAPSHOT_PREFILL}}"
  local same_request_final_decode="${requested_same_request_final_decode:-0}"
  local arm_persistent_cache=""
  if [ -n "${PAIR_SHARED_PERSISTENT_CACHE}" ]; then
    arm_persistent_cache="${PAIR_SHARED_PERSISTENT_CACHE}"
    mkdir -p "${arm_persistent_cache}"
  elif [ -n "${PAIR_PERSISTENT_CACHE_ROOT}" ]; then
    arm_persistent_cache="${PAIR_PERSISTENT_CACHE_ROOT}/${arm}"
    mkdir -p "${arm_persistent_cache}"
  fi
  local arm_pip_cache="${PIP_CACHE_DIR:-${arm_persistent_cache:+${arm_persistent_cache}/pip}}"
  if [ "${streaming_enabled}" != "1" ] || [ "${tokenizer_only}" != "1" ]; then
    exact_incremental_tokenizer=0
    final_only_incremental_tokenizer=0
    final_only_prefill=0
    prefix_seeded_start=0
    prefill_after_admission=0
    background_prefill_completion=0
    stable_first_snapshot_prefill=0
    same_request_final_decode=0
    compact_request_context=0
  elif [ "${exact_incremental_tokenizer}" != "1" ]; then
    final_only_incremental_tokenizer=0
    final_only_prefill=0
    prefix_seeded_start=0
    prefill_after_admission=0
    background_prefill_completion=0
    stable_first_snapshot_prefill=0
    same_request_final_decode=0
    compact_request_context=0
  elif [ "${final_only_incremental_tokenizer}" != "1" ]; then
    final_only_prefill=0
    prefix_seeded_start=0
    prefill_after_admission=0
    background_prefill_completion=0
    stable_first_snapshot_prefill=0
    same_request_final_decode=0
  elif [ "${final_only_prefill}" != "1" ]; then
    prefix_seeded_start=0
    prefill_after_admission=0
    background_prefill_completion=0
    stable_first_snapshot_prefill=0
    same_request_final_decode=0
  elif [ "${prefill_after_admission}" != "1" ] || [ "${prefix_seeded_start}" != "1" ]; then
    background_prefill_completion=0
    stable_first_snapshot_prefill=0
    same_request_final_decode=0
  fi
  if [ "${requested_same_request_final_decode:-0}" = "1" ] && [ "${same_request_final_decode}" != "1" ]; then
    echo "ERROR: same-request final decode arm '${arm}' does not enable the complete prefill path" >&2
    exit 1
  fi
  if [ "${same_request_final_decode}" = "1" ] && [ "${background_prefill_completion}" != "1" ]; then
    echo "ERROR: same-request final decode arm '${arm}' requires background prefill completion" >&2
    exit 1
  fi
  local counterfactual_full_tokenizer_timing="${COUNTERFACTUAL_FULL_TOKENIZER_TIMING}"
  if [ "${exact_incremental_tokenizer}" != "1" ]; then
    counterfactual_full_tokenizer_timing=0
  fi
  local experiment_name="streamtool-verified-temp0-r${NUM_VLLM_REPLICAS}-${RUN_ID}-${arm}"

  REPO_ROOT="${REPO_ROOT}" \
  PERSISTENT_CACHE="${arm_persistent_cache}" \
  PIP_CACHE_DIR="${arm_pip_cache}" \
  NUM_VLLM_REPLICAS="${NUM_VLLM_REPLICAS}" \
  TRAJECTORY_COLLECTION=1 \
  TRAJECTORY_COLLECTION_BATCH_SIZE="${TRAJECTORY_COLLECTION_BATCH_SIZE}" \
  TRAIN_DATA_PATH="${VERIFIED_DATA_PATH}" \
  VAL_DATA_PATH="${VERIFIED_DATA_PATH}" \
  TEMPERATURE=0.0 \
  TOP_P=1.0 \
  STREAMING_MIN_CHUNK_CHARS="${min_chunk_chars}" \
  STREAMING_INITIAL_CHUNK_CHARS="${initial_chunk_chars}" \
  STREAMING_INCREMENTAL_TOKENIZER_ONLY="${tokenizer_only}" \
  EXACT_INCREMENTAL_TOKENIZER="${exact_incremental_tokenizer}" \
  FINAL_ONLY_INCREMENTAL_TOKENIZER="${final_only_incremental_tokenizer}" \
  FINAL_ONLY_PREFILL="${final_only_prefill}" \
  PREFIX_SEEDED_START="${prefix_seeded_start}" \
  PREFILL_AFTER_ADMISSION="${prefill_after_admission}" \
  BACKGROUND_PREFILL_COMPLETION="${background_prefill_completion}" \
  SAME_REQUEST_FINAL_DECODE="${same_request_final_decode}" \
  STABLE_FIRST_SNAPSHOT_PREFILL="${stable_first_snapshot_prefill}" \
  COMPACT_REQUEST_CONTEXT="${compact_request_context}" \
  FINAL_ONLY_PREFILL_COMPLETION_GRACE_SECONDS="${FINAL_ONLY_PREFILL_COMPLETION_GRACE_SECONDS:-0.0}" \
  INCREMENTAL_TOKENIZER_CHECKPOINT_INTERVAL="${INCREMENTAL_TOKENIZER_CHECKPOINT_INTERVAL}" \
  COUNTERFACTUAL_FULL_TOKENIZER_TIMING="${counterfactual_full_tokenizer_timing}" \
  DETAILED_RUNTIME_METRICS="${DETAILED_RUNTIME_METRICS}" \
  BASE_CONCURRENCY="${BASE_CONCURRENCY}" \
  SWE_BENCH_ARTIFACT_CACHE_OFFLINE="${SWE_BENCH_ARTIFACT_CACHE_OFFLINE}" \
  STREAMING_TOOL_CALL="${streaming_enabled}" \
  SNAPSHOT_POLL_INTERVAL_SECONDS="${snapshot_poll_interval_seconds}" \
  SNAPSHOT_LONG_POLL_TIMEOUT_SECONDS="${snapshot_long_poll_timeout_seconds}" \
  LOG_GYM_RESPONSES=true \
  SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY}" \
  SBATCH_TIME="${SBATCH_TIME:-4:0:0}" \
  WANDB_GROUP="${WANDB_GROUP}" \
  EXP_SUFFIX="${experiment_name}" \
  BASE_LOG_DIR="${PAIR_DIR}/slurm" \
  LOGGER_LOG_DIR="${PAIR_DIR}/${arm}" \
  DRY_RUN="${DRY_RUN:-0}" \
  bash "${LAUNCHER}"

  if [ "${DRY_RUN:-0}" != "1" ]; then
    if [ "${SUBMIT_MODE:-sbatch}" = "direct" ]; then
      printf '%s\n' "${SLURM_JOB_ID}" > "${PAIR_DIR}/${arm}_job_id.txt"
    else
      cp "${REPO_ROOT}/latest_scale_gen_job_id.txt" "${PAIR_DIR}/${arm}_job_id.txt"
    fi
  fi
}

for arm_spec in ${PAIR_ARMS}; do
  arm=""
  streaming_enabled=""
  snapshot_poll_interval_seconds=""
  min_chunk_chars=""
  tokenizer_only=""
  final_only_prefill=""
  prefix_seeded_start=""
  prefill_after_admission=""
  compact_request_context=""
  initial_chunk_chars=""
  snapshot_long_poll_timeout_seconds=""
  stable_first_snapshot_prefill=""
  same_request_final_decode=""
  extra=""
  IFS=':' read -r arm streaming_enabled snapshot_poll_interval_seconds min_chunk_chars tokenizer_only final_only_prefill prefix_seeded_start prefill_after_admission compact_request_context initial_chunk_chars snapshot_long_poll_timeout_seconds stable_first_snapshot_prefill same_request_final_decode extra <<< "${arm_spec}"
  snapshot_poll_interval_seconds="${snapshot_poll_interval_seconds:-${SNAPSHOT_POLL_INTERVAL_SECONDS}}"
  tokenizer_only="${tokenizer_only:-0}"
  if [ -z "${arm}" ] || [ -n "${extra}" ] || { [ "${streaming_enabled}" != "0" ] && [ "${streaming_enabled}" != "1" ]; } || { [ -n "${min_chunk_chars}" ] && ! [[ "${min_chunk_chars}" =~ ^[1-9][0-9]*$ ]]; } || { [ "${tokenizer_only}" != "0" ] && [ "${tokenizer_only}" != "1" ]; } || { [ "${tokenizer_only}" = "1" ] && [ "${streaming_enabled}" != "1" ]; } || { [ -n "${final_only_prefill}" ] && [ "${final_only_prefill}" != "0" ] && [ "${final_only_prefill}" != "1" ]; } || { [ "${final_only_prefill:-0}" = "1" ] && [ "${tokenizer_only}" != "1" ]; } || { [ -n "${prefix_seeded_start}" ] && [ "${prefix_seeded_start}" != "0" ] && [ "${prefix_seeded_start}" != "1" ]; } || { [ "${prefix_seeded_start:-0}" = "1" ] && [ "${final_only_prefill:-0}" != "1" ]; } || { [ -n "${prefill_after_admission}" ] && [ "${prefill_after_admission}" != "0" ] && [ "${prefill_after_admission}" != "1" ]; } || { [ "${prefill_after_admission:-0}" = "1" ] && [ "${final_only_prefill:-0}" != "1" ]; } || { [ -n "${compact_request_context}" ] && [ "${compact_request_context}" != "0" ] && [ "${compact_request_context}" != "1" ]; } || { [ "${compact_request_context:-0}" = "1" ] && [ "${tokenizer_only}" != "1" ]; } || { [ -n "${initial_chunk_chars}" ] && ! [[ "${initial_chunk_chars}" =~ ^[1-9][0-9]*$ ]]; } || { [ -n "${snapshot_long_poll_timeout_seconds}" ] && ! [[ "${snapshot_long_poll_timeout_seconds}" =~ ^(0\.[0-9]*[1-9][0-9]*|[1-4](\.[0-9]+)?|5(\.0+)?)$ ]]; } || { [ -n "${stable_first_snapshot_prefill}" ] && [ "${stable_first_snapshot_prefill}" != "0" ] && [ "${stable_first_snapshot_prefill}" != "1" ]; } || { [ "${stable_first_snapshot_prefill:-0}" = "1" ] && { [ "${prefill_after_admission:-0}" != "1" ] || [ "${prefix_seeded_start:-0}" != "1" ]; }; } || { [ -n "${same_request_final_decode}" ] && [ "${same_request_final_decode}" != "0" ] && [ "${same_request_final_decode}" != "1" ]; }; then
    echo "ERROR: invalid PAIR_ARMS entry '${arm_spec}'; expected name:streaming_enabled[:poll_seconds[:min_chunk_chars[:tokenizer_only[:final_only_prefill[:prefix_seeded_start[:prefill_after_admission[:compact_request_context[:initial_chunk_chars[:snapshot_long_poll_timeout_seconds[:stable_first_snapshot_prefill[:same_request_final_decode]]]]]]]]]]]" >&2
    exit 1
  fi
  submit_arm "${arm}" "${streaming_enabled}" "${snapshot_poll_interval_seconds}" "${min_chunk_chars}" "${tokenizer_only}" "${final_only_prefill}" "${prefix_seeded_start}" "${prefill_after_admission}" "${compact_request_context}" "${initial_chunk_chars}" "${snapshot_long_poll_timeout_seconds}" "${stable_first_snapshot_prefill}" "${same_request_final_decode}"
  if [ "${SEQUENTIAL_ARMS}" = "1" ] && [ "${DRY_RUN:-0}" != "1" ] && [ "${SUBMIT_MODE:-sbatch}" != "direct" ]; then
    previous_job_id="$(cat "${REPO_ROOT}/latest_scale_gen_job_id.txt")"
    SBATCH_DEPENDENCY="afterany:${previous_job_id}"
  fi
done

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "Strict Verified pair dry run complete"
else
  echo "Strict Verified pair submitted"
fi
echo "Pair directory: ${PAIR_DIR}"
echo "Manifest: ${VERIFIED_DATA_PATH}"
echo "SWE-bench artifact cache offline: ${SWE_BENCH_ARTIFACT_CACHE_OFFLINE}"
echo "W&B group: ${WANDB_GROUP}"
echo "Arms: ${PAIR_ARMS}"
echo "Sequential arms: ${SEQUENTIAL_ARMS}"
if [ -n "${PAIR_PERSISTENT_CACHE_ROOT}" ]; then
  echo "Per-arm persistent cache root: ${PAIR_PERSISTENT_CACHE_ROOT}"
fi
if [ -n "${PAIR_SHARED_PERSISTENT_CACHE}" ]; then
  echo "Shared persistent cache: ${PAIR_SHARED_PERSISTENT_CACHE}"
fi
echo "Raw trajectories: ${PAIR_DIR}/<arm>/exp_001/trajectory_collection.jsonl"
