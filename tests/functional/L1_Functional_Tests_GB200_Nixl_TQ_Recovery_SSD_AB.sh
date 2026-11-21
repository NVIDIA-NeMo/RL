#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

# GPU_COUNT=2
# Select Nixl_TQ_Recovery_SSD_AB with test_platform=gb200 for an isolated
# diagnostic run. Normal CI selection and existing shards stay unchanged.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
cd "$PROJECT_ROOT"

source "$SCRIPT_DIR/../scripts/detect_rdma.sh"
if ! rdma_device_available; then
    echo "[FAIL] The SSD comparison requires a usable RDMA device" >&2
    exit 1
fi
export NEMO_RL_REQUIRE_NIXL=1

RESULT_DIR=$(mktemp -d "$SCRIPT_DIR/nixl-ssd-comparison.XXXXXX")
printf 'trial\tvariant\texit_code\n' > "$RESULT_DIR/results.tsv"
FAILED=0
TRIAL=0
# Alternate order across two pairs so a warm-up cannot masquerade as SSD impact.
for VARIANT in nixl nixl_no_ssd nixl_no_ssd nixl; do
    TRIAL=$((TRIAL + 1))
    printf '\n=== NIXL SSD comparison: trial %s, variant %s ===\n' "$TRIAL" "$VARIANT"
    # A failed training process must not prevent the other mode from being measured.
    set +e
    uv run --no-sync bash "$SCRIPT_DIR/grpo_dp_nixl_tq_recovery.sh" "$VARIANT" \
        2>&1 | tee "$RESULT_DIR/trial-$TRIAL-$VARIANT.log"
    STATUS=$?
    set -e
    printf '%s\t%s\t%s\n' "$TRIAL" "$VARIANT" "$STATUS" | tee -a "$RESULT_DIR/results.tsv"
    if (( STATUS != 0 )); then
        FAILED=1
    fi
    if [[ "$VARIANT" == "nixl" ]]; then
        TEST_NAME=grpo_dp_nixl_tq_recovery
    else
        TEST_NAME=grpo_dp_nixl_no_ssd_tq_recovery
    fi
    if [[ -d "$SCRIPT_DIR/$TEST_NAME" ]]; then
        mv "$SCRIPT_DIR/$TEST_NAME" "$RESULT_DIR/trial-$TRIAL-$VARIANT"
    fi
done

printf '\n=== NIXL SSD comparison results: %s ===\n' "$RESULT_DIR"
cat "$RESULT_DIR/results.tsv"
exit "$FAILED"
