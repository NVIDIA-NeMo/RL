#!/bin/bash
# True if the host exposes an RDMA device libibverbs can open whose port 1 is
# ACTIVE with an InfiniBand or Ethernet (RoCE) link layer, under any device
# name — exactly mooncake_cpu's rdma_devices() rule
# (nemo_rl/data_plane/adapters/transfer_queue.py). Other link layers (e.g.
# AWS EFA) must not count: CI would then require mooncake tests that
# rdma_devices() then rejects. Gate on uverbs* specifically, not just the
# /dev/infiniband directory: a host can have the directory without a verbs
# node, which is what libibverbs actually opens.
rdma_device_available() {
  compgen -G "/dev/infiniband/uverbs*" >/dev/null || return 1
  local port
  for port in /sys/class/infiniband/*/ports/1; do
    grep -qs ACTIVE "$port/state" &&
      grep -qsxE "InfiniBand|Ethernet" "$port/link_layer" && return 0
  done
  return 1
}
