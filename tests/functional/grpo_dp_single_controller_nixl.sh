#!/bin/bash
# grpo_dp_single_controller.sh on the nixl data plane, with the units' SSD tier
# on and a slab this run exceeds, saving a data-plane checkpoint at the last
# step. Fails unless a unit spilled and no unit memory was retired.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

set -eou pipefail

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

EXP_NAME=grpo_dp_single_controller_nixl
EXP_DIR=$SCRIPT_DIR/$EXP_NAME

# 256 KiB slab per unit: a 128 KiB put area (this run holds ~240 KB at once)
# and a 128 KiB SSD IO area. save_period=2 = one save, at the last step.
EXP_NAME=$EXP_NAME bash "$SCRIPT_DIR/grpo_dp_single_controller.sh" \
    data_plane.backend=nixl \
    data_plane.nixl.unit_slab_bytes=262144 \
    data_plane.nixl.ssd.enabled=true \
    data_plane.nixl.ssd.path=$EXP_DIR/ssd \
    data_plane.nixl.ssd.staging_bytes=131072 \
    checkpointing.enabled=true \
    checkpointing.checkpoint_dir=$EXP_DIR/ckpts \
    checkpointing.save_period=2 \
    checkpointing.metric_name=null \
    checkpointing.save_data_plane=true \
    "$@"

test -d "$EXP_DIR/ckpts/step_2/data_plane"
echo "✅ step_2 data-plane checkpoint saved"

# Per-step unit metrics the driver logs (tq_nixl.unit_metrics).
if [[ "${RUN_CONVERGENCE_CHECKS:-1}" == "1" ]]; then
    uv run "$SCRIPT_DIR/../check_metrics.py" "$EXP_DIR/metrics.json" \
        'max(data["data_plane/nixl/ssd_spills_sum"]) > 0' \
        'max(data["data_plane/nixl/retired_mb_sum"]) == 0'
fi
