#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Measure Mooncake's MNNVL (NVLink) transport across two nodes.
#
# The allocation is the hard-to-reproduce part -- two nodes, the right image,
# the right mounts -- so it lives here rather than in shell history. Re-run this
# after any mooncake bump: it decides the data plane's ceiling.
#
#   sbatch -A <account> -p <partition> tools/nvlink_crossnode.sbatch
#
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --job-name=nvlink-crossnode
set -uo pipefail

CONTAINER_IMAGE="${CONTAINER_IMAGE:-/scratch/fsw/portfolios/nemotron/users/zhiyul/enroot-images/nvcr.io+nvidian+nemo-rl+nightly-gym.2026-08-17.squashfs}"
MOUNTS="${MOUNTS:-/scratch:/scratch,/lustre:/lustre}"
WORKDIR="${WORKDIR:-$(pwd)}"
MB="${MB:-512}"
RV="${RV:-$WORKDIR/workspace/xnode-rv/$SLURM_JOB_ID}"
mkdir -p "$(dirname "$RV")"

srun_in_container() {
  srun --ntasks=2 --ntasks-per-node=1 "$@" \
    --container-image="$CONTAINER_IMAGE" \
    --no-container-mount-home \
    --container-mounts="$MOUNTS" \
    --container-workdir="$WORKDIR"
}

echo "===== NODES ====="
scontrol show hostnames "$SLURM_JOB_NODELIST"

# ClusterUUID *and* CliqueId: MNNVL reachability needs both, and two nodes can
# share a ClusterUUID while sitting in different cliques with no fabric path
# between them. ray.sub currently registers nvlink_domain_<ClusterUUID> only.
echo
echo "===== FABRIC DOMAIN PER NODE ====="
srun_in_container bash -c \
  'echo "--- $(hostname) ---"; nvidia-smi -q | grep -iE "ClusterUUID|CliqueId" | head -2'

echo
echo "===== CROSS-NODE NVLINK, ${MB}MB ====="
srun_in_container --export=ALL,MC_FORCE_MNNVL=1 \
  bash tools/nvlink_crossnode.sh "$RV.nv" "$MB" nvlink

# RDMA control at the same size. Without it a slow NVLink number is
# uninterpretable -- there is nothing to compare against on this fabric today.
echo
echo "===== CROSS-NODE RDMA CONTROL, ${MB}MB ====="
srun_in_container bash tools/nvlink_crossnode.sh "$RV.rd" "$MB" rdma
