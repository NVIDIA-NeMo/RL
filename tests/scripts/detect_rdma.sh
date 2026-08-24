#!/bin/bash
# True if the host exposes an mlx5 RDMA device libibverbs can open (RoCE or
# InfiniBand) — what mooncake_cpu's rdma_devices() (nemo_rl/data_plane/adapters/
# transfer_queue.py) requires. Gate on uverbs* specifically, not just the
# /dev/infiniband directory: a host can have the directory without a verbs
# node, which is what libibverbs actually opens.
rdma_device_available() {
  compgen -G "/dev/infiniband/uverbs*" >/dev/null &&
    compgen -G "/sys/class/infiniband/mlx5_*/ports/1/link_layer" >/dev/null
}
