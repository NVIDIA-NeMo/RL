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
ARTIFACT_PREWARMER="${REPO_ROOT}/examples/swe_bench/prewarm_swebench_artifacts.sh"
VERIFIED_DATA_PATH="${VERIFIED_DATA_PATH:-${REPO_ROOT}/results/swebench_verified/swebench_verified_500.jsonl}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
PAIR_DIR="${PAIR_DIR:-${REPO_ROOT}/results/streaming_tool_call_verified/${RUN_ID}}"
WANDB_GROUP="${WANDB_GROUP:-streaming-tool-call-verified-temp0-${RUN_ID}}"
NUM_VLLM_REPLICAS="${NUM_VLLM_REPLICAS:-32}"
TRAJECTORY_COLLECTION_BATCH_SIZE="${TRAJECTORY_COLLECTION_BATCH_SIZE:-128}"
EXPECTED_COUNT="${EXPECTED_COUNT:-500}"
PREWARM_SWEBENCH_ARTIFACTS="${PREWARM_SWEBENCH_ARTIFACTS:-1}"
# Each PAIR_ARMS entry accepts
# name:streaming_enabled[:snapshot_poll_seconds[:min_chunk_chars]].
SNAPSHOT_POLL_INTERVAL_SECONDS="${SNAPSHOT_POLL_INTERVAL_SECONDS:-0.1}"
PAIR_ARMS="${PAIR_ARMS:-streaming_off:0 streaming_on:1}"

python3 "${AUDITOR}" verify \
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
  local experiment_name="streamtool-verified-temp0-r${NUM_VLLM_REPLICAS}-${RUN_ID}-${arm}"

  REPO_ROOT="${REPO_ROOT}" \
  NUM_VLLM_REPLICAS="${NUM_VLLM_REPLICAS}" \
  TRAJECTORY_COLLECTION=1 \
  TRAJECTORY_COLLECTION_BATCH_SIZE="${TRAJECTORY_COLLECTION_BATCH_SIZE}" \
  TRAIN_DATA_PATH="${VERIFIED_DATA_PATH}" \
  VAL_DATA_PATH="${VERIFIED_DATA_PATH}" \
  TEMPERATURE=0.0 \
  TOP_P=1.0 \
  STREAMING_MIN_CHUNK_CHARS="${min_chunk_chars}" \
  SWE_BENCH_ARTIFACT_CACHE_OFFLINE="${SWE_BENCH_ARTIFACT_CACHE_OFFLINE}" \
  STREAMING_TOOL_CALL="${streaming_enabled}" \
  SNAPSHOT_POLL_INTERVAL_SECONDS="${snapshot_poll_interval_seconds}" \
  LOG_GYM_RESPONSES=true \
  SBATCH_TIME="${SBATCH_TIME:-4:0:0}" \
  WANDB_GROUP="${WANDB_GROUP}" \
  EXP_SUFFIX="${experiment_name}" \
  BASE_LOG_DIR="${PAIR_DIR}/slurm" \
  LOGGER_LOG_DIR="${PAIR_DIR}/${arm}" \
  DRY_RUN="${DRY_RUN:-0}" \
  bash "${LAUNCHER}"

  if [ "${DRY_RUN:-0}" != "1" ]; then
    cp "${REPO_ROOT}/latest_scale_gen_job_id.txt" "${PAIR_DIR}/${arm}_job_id.txt"
  fi
}

for arm_spec in ${PAIR_ARMS}; do
  arm=""
  streaming_enabled=""
  snapshot_poll_interval_seconds=""
  min_chunk_chars=""
  extra=""
  IFS=':' read -r arm streaming_enabled snapshot_poll_interval_seconds min_chunk_chars extra <<< "${arm_spec}"
  snapshot_poll_interval_seconds="${snapshot_poll_interval_seconds:-${SNAPSHOT_POLL_INTERVAL_SECONDS}}"
  if [ -z "${arm}" ] || [ -n "${extra}" ] || { [ "${streaming_enabled}" != "0" ] && [ "${streaming_enabled}" != "1" ]; } || { [ -n "${min_chunk_chars}" ] && ! [[ "${min_chunk_chars}" =~ ^[1-9][0-9]*$ ]]; }; then
    echo "ERROR: invalid PAIR_ARMS entry '${arm_spec}'; expected name:0[:poll_seconds[:min_chunk_chars]] or name:1[:poll_seconds[:min_chunk_chars]]" >&2
    exit 1
  fi
  submit_arm "${arm}" "${streaming_enabled}" "${snapshot_poll_interval_seconds}" "${min_chunk_chars}"
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
echo "Raw trajectories: ${PAIR_DIR}/<arm>/exp_001/trajectory_collection.jsonl"
