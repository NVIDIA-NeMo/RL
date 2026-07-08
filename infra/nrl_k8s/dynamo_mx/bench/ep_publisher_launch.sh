#!/bin/bash
set -x
V=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
B=/mnt/rl-workspace/kavink/ep_bench
LIBS=$V/lib/python3.13/site-packages/nixl_cu12.libs
# overlay MX Megatron helpers into the image's nemo_rl + today's modelexpress into the venv
cp "$B/mx_megatron_helpers.py" /opt/nemo-rl/nemo_rl/distributed/ || exit 1
cp "$B/mx_helpers.py" /opt/nemo-rl/nemo_rl/distributed/ || exit 1
( cd "$V/lib/python3.13/site-packages" && tar xzf "$B/mx_pkg.tgz" ) || exit 1
# point UCX/NIXL at THIS venv's bundled libs (the fix: worker paths don't exist here)
export LD_LIBRARY_PATH="$LIBS:$LIBS/ucx:$LD_LIBRARY_PATH"
export UCX_MODULE_DIR="$LIBS/ucx"
export NIXL_PLUGIN_DIR="$LIBS/nixl"
export UCX_TLS=rc,cuda_copy
export NIXL_UCX_TLS=rc,cuda_copy
export UCX_IB_GID_INDEX=3
export UCX_IB_GPU_DIRECT_RDMA=yes
# single-node EP4 NCCL uses NVLink; disable NCCL IB/SHARP plugin so HPC-X UCX
# doesn't load and segfault-clash with NIXL's bundled UCX. (EP8 2-node will need
# a different reconciliation.)
export NCCL_IB_DISABLE=1
export NCCL_NET_PLUGIN=none
# NOTE: no MX_RDMA_NIC_PIN=stripe here — forcing UCX_NET_DEVICES=mlx5_0..3 made rc
# report "no usable transports"; let UCX auto-select the device (validated working).
export HOLD_S=3600
export MODEL_EXPRESS_URL=modelexpress-server.kavin.svc.cluster.local:8001
export HF_HOME=/mnt/rl-workspace/kavink/hf-cache
cd /tmp
exec "$V/bin/torchrun" --nproc_per_node=4 --master_port=29505 "$B/ep_publisher.py"
