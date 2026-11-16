#!/bin/bash
# Run native TQ recovery with the nixl backend. Pass nixl_ssd to turn the
# units' SSD tier on.

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

# HPC-X's NCCL net plugin breaks NIXL's UCX (docs/design-docs/nixl-data-plane.md).
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}

exec bash "$SCRIPT_DIR/grpo_dp_single_controller_tq_recovery.sh" "${1:-nixl}"
