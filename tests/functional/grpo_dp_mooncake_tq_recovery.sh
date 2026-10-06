#!/bin/bash
# Run native TQ recovery with the Mooncake CPU backend.

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

source "$SCRIPT_DIR/../scripts/detect_rdma.sh"
if [[ -z "${MC_MOONCAKE_DEVICE:-}" ]] && ! rdma_device_available; then
    echo "[SKIP] no usable mlx5 RDMA device; mooncake_cpu requires RDMA." \
         "Set MC_MOONCAKE_DEVICE=<dev> to override."
    exit 0
fi

exec bash "$SCRIPT_DIR/grpo_dp_single_controller_tq_recovery.sh" mooncake_cpu
