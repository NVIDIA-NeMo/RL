#!/bin/bash
# True if the host exposes an RDMA device libibverbs can open (RoCE or
# InfiniBand) with port 1 ACTIVE, under any device name — the same rule as
# mooncake_cpu's rdma_devices() (nemo_rl/data_plane/adapters/transfer_queue.py). Gate on uverbs* specifically, not just the
# /dev/infiniband directory: a host can have the directory without a verbs
# node, which is what libibverbs actually opens.
rdma_device_available() {
  compgen -G "/dev/infiniband/uverbs*" >/dev/null &&
    grep -qs ACTIVE /sys/class/infiniband/*/ports/1/state
}
