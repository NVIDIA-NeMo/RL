#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:10:00
#SBATCH --job-name=pr3436-ibgda-probe

set -euo pipefail

: "${PROBE_DIR:?Set PROBE_DIR to a Lustre path}"

mkdir -p "${PROBE_DIR}"

srun \
  --nodes="${SLURM_NNODES}" \
  --ntasks="${SLURM_NNODES}" \
  --ntasks-per-node=1 \
  --export=ALL \
  bash -lc '
    set -euo pipefail
    node_dir="${PROBE_DIR}/$(hostname)"
    mkdir -p "${node_dir}"
    {
      echo "host=$(hostname)"
      echo "kernel=$(uname -r)"
      echo "nvidia_driver=$(cat /proc/driver/nvidia/version | head -1)"
      echo "gpu_inventory"
      nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader
      echo "gpu_nic_topology"
      nvidia-smi topo -m
      echo "infiniband_devices"
      ls -l /dev/infiniband || true
      echo "ibdev2netdev"
      command -v ibdev2netdev >/dev/null && ibdev2netdev || true
      echo "ibstat"
      command -v ibstat >/dev/null && ibstat || true
      echo "rdma_gids"
      if command -v show_gids >/dev/null; then
        show_gids
      else
        for gid_file in /sys/class/infiniband/mlx5_*/ports/1/gids/*; do
          printf "%s=" "${gid_file}"
          cat "${gid_file}"
        done
      fi
      echo "gdrcopy"
      ls -l /dev/gdrdrv 2>/dev/null || true
      lsmod | grep -E "^gdrdrv" || true
      echo "nvidia_parameters"
      grep -E "EnableStreamMemOPs|PeerMappingOverride" /proc/driver/nvidia/params 2>/dev/null || true
      echo "rdma_libraries"
      ldconfig -p 2>/dev/null | grep -E "libmlx5|libibverbs|libgdrapi" || true
    } | tee "${node_dir}/transport-environment.txt"
  '

echo "IBGDA_ENV_PROBE_PASS"
