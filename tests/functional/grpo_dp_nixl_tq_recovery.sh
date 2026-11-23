#!/bin/bash
# Run native TQ recovery with the nixl backend. The units' SSD tier is on by
# default; pass nixl_ssd for a put area so small that every step spills.

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# nixl is RDMA-only. Skip on hosts without a usable RDMA device, unless the
# runner says it has one (NEMO_RL_REQUIRE_NIXL=1): then a missing device fails.
source "$SCRIPT_DIR/../scripts/detect_rdma.sh"
if ! rdma_device_available; then
    if [[ "${NEMO_RL_REQUIRE_NIXL:-0}" == "1" ]]; then
        echo "[FAIL] NEMO_RL_REQUIRE_NIXL=1 but no usable RDMA device"
        exit 1
    fi
    echo "[SKIP] no usable RDMA device; the nixl data plane requires RDMA."
    exit 0
fi

# This two-GPU case stays on one node; clients and CPU units need the same RoCE rail.
# Leave explicit selections and InfiniBand jobs alone, including diagnostic "all".
if [[ -z "${UCX_NET_DEVICES:-}" ]]; then
    PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
    UCX_DEVICE=$(cd "$PROJECT_ROOT" && uv run --no-sync python - <<'PYDEVICE'
from nemo_rl.data_plane.adapters.transfer_queue_env import fabric_is_roce_only, rdma_rails

if fabric_is_roce_only():
    rails = rdma_rails()
    if not rails:
        raise RuntimeError("No active fabric rail for the NIXL recovery test")
    print(f"{rails[0]}:1")
PYDEVICE
    )
    if [[ -n "$UCX_DEVICE" ]]; then
        ibv_devinfo -d "${UCX_DEVICE%:*}" -i "${UCX_DEVICE##*:}" \
            || echo "[WARN] ibv_devinfo failed for $UCX_DEVICE" >&2
        export UCX_NET_DEVICES="$UCX_DEVICE"
        echo "NIXL recovery test: UCX_NET_DEVICES=$UCX_NET_DEVICES"
    fi
fi

exec bash "$SCRIPT_DIR/grpo_dp_single_controller_tq_recovery.sh" "${1:-nixl}"
