#!/usr/bin/env bash
set -euo pipefail

V=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
B=/mnt/rl-workspace/kavink/ep_bench
LIBS=$V/lib/python3.13/site-packages/nixl_cu13.libs

: "${NODE_RANK:?set NODE_RANK=0 or 1}"
: "${MASTER_ADDR:?set MASTER_ADDR to node-rank-0 pod IP}"

cp "$B/mx_megatron_helpers.py" /opt/nemo-rl/nemo_rl/distributed/
cp "$B/mx_helpers.py" /opt/nemo-rl/nemo_rl/distributed/
(cd "$V/lib/python3.13/site-packages" && tar xzf "$B/mx_pkg.tgz")

export LD_LIBRARY_PATH="$LIBS:$LIBS/ucx:${LD_LIBRARY_PATH:-}"
export UCX_MODULE_DIR="$LIBS/ucx"
export NIXL_PLUGIN_DIR="$LIBS/nixl"
export UCX_TLS=rc,cuda_copy
export NIXL_UCX_TLS=rc,cuda_copy
export UCX_IB_GID_INDEX=3
export UCX_IB_GPU_DIRECT_RDMA=yes

export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_MNNVL_ENABLE=0

export HOLD_S="${HOLD_S:-3600}"
export MODEL_EXPRESS_URL="${MODEL_EXPRESS_URL:-modelexpress-server.kavin.svc.cluster.local:8001}"
export HF_HOME="${HF_HOME:-/mnt/rl-workspace/kavink/hf-cache}"

cd /tmp
exec "$V/bin/torchrun" \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="${MASTER_PORT:-29508}" \
  "$B/ep_publisher.py"
