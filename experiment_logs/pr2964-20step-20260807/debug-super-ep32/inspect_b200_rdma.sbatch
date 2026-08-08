#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --account=coreai_chef_posttrain
#SBATCH --partition=batch
#SBATCH --time=00:15:00
#SBATCH --job-name=coreai_chef_posttrain.hybridep-b200-rdma-inspect
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/pr2964-20step-20260807/debug-super-ep32/logs/rdma-inspect-%j.out

set -euo pipefail

container=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/containers/nemo-rl-nightly-cw-fallback-20260808/nemo_rl_nightly_20260805_15171871.sqsh
probe_gpus_per_node=${PROBE_GPUS_PER_NODE:-8}

srun \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node="${probe_gpus_per_node}" \
  --container-image="${container}" \
  --container-mounts=/lustre:/lustre \
  --no-container-mount-home \
  bash -lc '
    set -euo pipefail
    echo "host=$(hostname)"
    echo "memlock=$(ulimit -l)"
    echo "visible=${CUDA_VISIBLE_DEVICES-<unset>}"
    nvidia-smi -L
    nvidia-smi topo -m
    ls -l /dev/infiniband || true
    ls -l /sys/class/infiniband || true
    grep -E "^(nvidia_peermem|nv_peer_mem) " /proc/modules || true
    test -d /sys/module/nvidia_peermem && echo "nvidia_peermem=loaded" || echo "nvidia_peermem=missing"
    command -v modinfo && modinfo nvidia_peermem || true
    command -v ibv_devices && ibv_devices || true
    command -v ibv_devinfo && ibv_devinfo || true
    ldconfig -p | grep -E "lib(ibverbs|mlx5|nixl|ucp|ucs)" || true
    /opt/nemo_rl_venv/bin/python - <<"PY"
import importlib.util
import os
import site
import torch

print(f"torch={torch.__version__}")
print(f"cuda_devices={torch.cuda.device_count()}")
nixl_spec = importlib.util.find_spec("nixl")
print(f"nixl={nixl_spec}")
print(f"sitepackages={site.getsitepackages()}")
for root, _, files in os.walk("/opt/nemo_rl_venv"):
    for name in files:
        if name.startswith("libnixl") or name in {"nixl.h", "nixl.hpp"}:
            print(os.path.join(root, name))
PY
    /opt/nemo_rl_venv/bin/python -m pip show -f nixl nixl-cu13 || true
  '
