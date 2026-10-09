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

# Batch script for a training job attached to an external vLLM pool job
# (run_in_allocation.sh, EXTERNAL_VLLM_MODE=serve) that was not yet READY when
# the training job was submitted. The pool's URLs are only known once it holds
# nodes, so they are resolved here, at job start: wait for READY from the
# expected pool job, put each pool's URL in place of its placeholder in
# COMMAND, then run ray.sub.
set -euo pipefail

: "${EXTERNAL_VLLM_POOL_DIR:?EXTERNAL_VLLM_POOL_DIR is required}"
: "${EXTERNAL_VLLM_POOL_JOB:?EXTERNAL_VLLM_POOL_JOB is required}"
: "${EXTERNAL_VLLM_POOLS:?EXTERNAL_VLLM_POOLS is required}"
: "${COMMAND:?COMMAND is required}"
: "${RAY_SUB:?RAY_SUB is required}"

timeout_s="${EXTERNAL_VLLM_ATTACH_TIMEOUT:-3600}"
poll_s="${EXTERNAL_VLLM_ATTACH_POLL_S:-15}"
deadline=$((SECONDS + timeout_s))
echo "[INFO] $(date -Iseconds) waiting for judge pool job ${EXTERNAL_VLLM_POOL_JOB} (${EXTERNAL_VLLM_POOL_DIR})"
while true; do
  state=$(squeue -h -j "${EXTERNAL_VLLM_POOL_JOB}" -o %T 2>/dev/null | head -1)
  if [[ -z "${state}" ]]; then
    echo "[FATAL] judge pool job ${EXTERNAL_VLLM_POOL_JOB} is gone" >&2
    exit 1
  fi
  # READY must come from the expected pool job, not a previous one.
  if [[ -f "${EXTERNAL_VLLM_POOL_DIR}/READY" \
        && "$(cat "${EXTERNAL_VLLM_POOL_DIR}/job_id" 2>/dev/null)" == "${EXTERNAL_VLLM_POOL_JOB}" ]]; then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "[FATAL] judge pool job ${EXTERNAL_VLLM_POOL_JOB} not READY after ${timeout_s}s (state ${state})" >&2
    exit 1
  fi
  sleep "${poll_s}"
done
echo "[INFO] $(date -Iseconds) judge pool READY since $(cat "${EXTERNAL_VLLM_POOL_DIR}/READY")"

for pool in ${EXTERNAL_VLLM_POOLS}; do
  url=$(cat "${EXTERNAL_VLLM_POOL_DIR}/${pool,,}_url" 2>/dev/null || true)
  placeholder_var="${pool}_URL_PLACEHOLDER"
  placeholder="${!placeholder_var:-}"
  if [[ -z "${url}" || -z "${placeholder}" ]]; then
    echo "[FATAL] no URL or placeholder for pool ${pool}" >&2
    exit 1
  fi
  COMMAND="${COMMAND//${placeholder}/${url}}"
  echo "[INFO] attach: ${pool} -> ${url}"
done
export COMMAND
exec bash "${RAY_SUB}"
