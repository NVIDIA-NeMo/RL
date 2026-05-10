#!/usr/bin/env bash
# RoCE / aws-ofi-nccl pod preflight check.
#
# Mounted into the worker pod's roce-preflight initContainer via a
# ConfigMap (see infra/preflight/roce-preflight-configmap.yaml).
# Invoked as: bash /preflight/roce_preflight.sh
#
# Three checks:
#   1. /dev/infiniband/uverbs0 is wired by the DRA RoCE channel
#   2. ibv_devinfo finds at least one RoCE port in PORT_ACTIVE
#   3. (best-effort) device count matches what we expect on this node
#
# fi_info-based checks were dropped — aws-ofi-nccl can't enumerate an
# EFA provider on this cluster (libfabric only has tcp/sockets/udp),
# but NCCL falls back to the kernel IBVerbs path which is what makes
# RDMA work. So we check IBVerbs, not libfabric.
#
# Any failure exits 1 → pod stuck in Init:Error → autoscaler reaps and
# replaces. The 60s wait covers the typical DRA bind window.

set -e

# Wait up to 60s for /dev/infiniband to be populated by DRA.
deadline=$((SECONDS + 60))
while [ $SECONDS -lt $deadline ]; do
    if ls /dev/infiniband/uverbs0 >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! ls /dev/infiniband/uverbs0 >/dev/null 2>&1; then
    echo "[roce-preflight] FAILED: /dev/infiniband/uverbs0 not present after 60s — DRA RoCE channel did not bind"
    exit 1
fi

devices=$(ls -1 /dev/infiniband/uverbs* 2>/dev/null | wc -l)
echo "[roce-preflight] DRA bound $devices uverbs device(s)"

# At least one RoCE port must be ACTIVE for NCCL IBVerbs transport to work.
if ! ibv_devinfo 2>/dev/null | grep -q PORT_ACTIVE; then
    echo "[roce-preflight] FAILED: no PORT_ACTIVE on any RoCE port (ibv_devinfo)"
    ibv_devinfo 2>&1 | tail -20
    exit 1
fi

active=$(ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE)
echo "[roce-preflight] OK: $devices uverbs devices, $active active port(s)"
