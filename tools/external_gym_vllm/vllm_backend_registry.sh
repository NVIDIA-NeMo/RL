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

# File-backed registry helpers for an in-allocation external Gym vLLM pool.

set -euo pipefail

EXTERNAL_VLLM_STATE_DIR="${EXTERNAL_VLLM_STATE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
EXTERNAL_VLLM_GROUP_ID="${EXTERNAL_VLLM_GROUP_ID:-default}"
REGISTRY_FILE="${EXTERNAL_VLLM_STATE_DIR}/.registry_${EXTERNAL_VLLM_GROUP_ID}"
REGISTRY_LOCK="${REGISTRY_FILE}.lock"

_ensure_registry() {
  touch "${REGISTRY_FILE}" "${REGISTRY_LOCK}"
}

registry_add() {
  local backend_id="$1" ip="$2" port="$3"
  _ensure_registry
  (
    flock -w 10 200
    grep -v "^${backend_id} " "${REGISTRY_FILE}" > "${REGISTRY_FILE}.tmp" 2>/dev/null || true
    echo "${backend_id} ${ip} ${port} $(date +%s) ready" >> "${REGISTRY_FILE}.tmp"
    mv "${REGISTRY_FILE}.tmp" "${REGISTRY_FILE}"
  ) 200>"${REGISTRY_LOCK}"
}

registry_remove() {
  local backend_id="$1"
  _ensure_registry
  (
    flock -w 10 200
    grep -v "^${backend_id} " "${REGISTRY_FILE}" > "${REGISTRY_FILE}.tmp" 2>/dev/null || true
    mv "${REGISTRY_FILE}.tmp" "${REGISTRY_FILE}"
  ) 200>"${REGISTRY_LOCK}"
}

registry_list() {
  _ensure_registry
  (
    flock -s -w 10 200
    cat "${REGISTRY_FILE}"
  ) 200>"${REGISTRY_LOCK}"
}

registry_list_ready() {
  registry_list | awk '$5 == "ready" { print $2 ":" $3 }'
}

registry_count_ready() {
  registry_list_ready | wc -l
}
