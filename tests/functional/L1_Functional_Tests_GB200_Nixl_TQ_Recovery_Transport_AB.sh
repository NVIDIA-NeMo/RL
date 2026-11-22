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
# Isolate UCX device selection, not SSD, with the native NIXL recovery case.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
cd "$PROJECT_ROOT"
source "$SCRIPT_DIR/../scripts/detect_rdma.sh"
if ! rdma_device_available; then
    echo "[FAIL] The transport comparison requires a usable RDMA device" >&2
    exit 1
fi
command -v ibv_devinfo >/dev/null

RESULT_DIR=$(mktemp -d "$SCRIPT_DIR/nixl-transport-comparison.XXXXXX")
RAILS=$(uv run --no-sync python -c \
    'from nemo_rl.data_plane.adapters.transfer_queue_env import rdma_rails; print("\n".join(rdma_rails()))')
SINGLE_DEVICE=""
while IFS= read -r DEVICE; do
    [[ -n "$DEVICE" ]] || continue
    if ibv_devinfo -d "$DEVICE" -i 1 > "$RESULT_DIR/verbs-$DEVICE.log" 2>&1; then
        cat "$RESULT_DIR/verbs-$DEVICE.log"
        SINGLE_DEVICE="$DEVICE:1"
        break
    fi
    printf 'Rejected RDMA device %s:\n' "$DEVICE"
    cat "$RESULT_DIR/verbs-$DEVICE.log"
done <<< "$RAILS"
if [[ -z "$SINGLE_DEVICE" ]]; then
    echo "[FAIL] No discovered active fabric rail could be opened with ibv_devinfo" >&2
    exit 1
fi

export NEMO_RL_REQUIRE_NIXL=1
# Hold transports and logging fixed; same-host helpers remain allowed, never TCP.
export UCX_TLS=rc,cuda_copy,cuda_ipc,self,sm
export UCX_LOG_LEVEL=info
uv run --no-sync python - <<'PYINFO' | tee "$RESULT_DIR/runtime.log"
import importlib.metadata
import platform
from pathlib import Path

from nemo_rl.data_plane.adapters.transfer_queue_env import rail_link_layers, rdma_rails

print("platform:", platform.platform())
for name in ("nixl", "nixl-cu13", "torch", "ray"):
    print(f"{name}={importlib.metadata.version(name)}")
print("active link layers:", rail_link_layers())
print("fabric rails:", rdma_rails())
for path in sorted(Path("/sys/class/infiniband").glob("*/device/numa_node")):
    print(path, path.read_text().strip())
PYINFO
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a "$RESULT_DIR/runtime.log"
printf 'single_device=%s\nUCX_TLS=%s\nUCX_LOG_LEVEL=%s\n' \
    "$SINGLE_DEVICE" "$UCX_TLS" "$UCX_LOG_LEVEL" | tee -a "$RESULT_DIR/runtime.log"
printf 'trial\tmode\tucx_net_devices\texit_code\tstarted_utc\tended_utc\n' > "$RESULT_DIR/results.tsv"
FAILED=0
TRIAL=0
# Two repetitions per mode reduce order effects. SSD stays enabled in every trial.
for MODE in all single single all; do
    TRIAL=$((TRIAL + 1))
    if [[ "$MODE" == all ]]; then
        export UCX_NET_DEVICES=all
    else
        export UCX_NET_DEVICES="$SINGLE_DEVICE"
    fi
    STARTED=$(date -u +%FT%TZ)
    printf '\n=== NIXL transport trial %s: mode=%s UCX_NET_DEVICES=%s start=%s ===\n' \
        "$TRIAL" "$MODE" "$UCX_NET_DEVICES" "$STARTED"
    # Preserve the first failure and still measure the other mode.
    set +e
    uv run --no-sync bash "$SCRIPT_DIR/grpo_dp_nixl_tq_recovery.sh" nixl \
        2>&1 | tee "$RESULT_DIR/trial-$TRIAL-$MODE.log"
    STATUS=$?
    set -e
    ENDED=$(date -u +%FT%TZ)
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$TRIAL" "$MODE" "$UCX_NET_DEVICES" "$STATUS" "$STARTED" "$ENDED" | tee -a "$RESULT_DIR/results.tsv"
    if (( STATUS != 0 )); then
        FAILED=1
    fi
    if [[ -d "$SCRIPT_DIR/grpo_dp_nixl_tq_recovery" ]]; then
        mv "$SCRIPT_DIR/grpo_dp_nixl_tq_recovery" "$RESULT_DIR/trial-$TRIAL-$MODE"
    fi
done

printf '\n=== NIXL transport comparison results: %s ===\n' "$RESULT_DIR"
cat "$RESULT_DIR/results.tsv"
exit "$FAILED"
