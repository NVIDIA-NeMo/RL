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

# Restart the lightweight GenRM load balancer after an unexpected exit.

set -uo pipefail

PYTHON="${PYTHON:-python3}"
LB_SCRIPT="$(dirname "$0")/genrm_lb.py"
PORT="${1:?port required}"
REGISTRY_DIR="${2:?registry directory required}"
GROUP_ID="${3:?group ID required}"

trap 'kill %1 2>/dev/null; exit 0' TERM INT

fast_failures=0
while true; do
  started_at=$(date +%s)
  echo "$(date) [WATCHDOG] Starting load balancer on port ${PORT}"
  "${PYTHON}" "${LB_SCRIPT}" \
    --port "${PORT}" \
    --registry-dir "${REGISTRY_DIR}" \
    --group-id "${GROUP_ID}"
  status=$?
  elapsed=$(( $(date +%s) - started_at ))
  echo "$(date) [WATCHDOG] Load balancer exited (${status}) after ${elapsed}s"

  if (( elapsed < 5 )); then
    fast_failures=$((fast_failures + 1))
    if (( fast_failures >= 3 )); then
      echo "$(date) [WATCHDOG] ERROR: load balancer failed three times during startup"
      exit 1
    fi
  else
    fast_failures=0
  fi
  sleep 2
done
