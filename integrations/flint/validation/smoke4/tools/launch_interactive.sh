#!/usr/bin/env bash
# Run manually on the OCI HSG login host after staging the private workspace.
# Math check:       bash tools/launch_interactive.sh
# Synthetic check:  FLINT_SYNTHETIC_SMOKE=1 bash tools/launch_interactive.sh
# Additional arguments are forwarded to run_grpo.sh as training overrides.
set -euo pipefail

flint_task_root=${FLINT_TASK_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/wenweng/Automodel/flint-grpo-4layer-20261019}
flint_image=${FLINT_CONTAINER_IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/wenweng/Automodel/flint-nemorl-20261019/images/flint-nemorl-terry-20261019-arm64-native-01.sqsh}
export FLINT_SYNTHETIC_SMOKE=${FLINT_SYNTHETIC_SMOKE:-0}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Run this launcher on the OCI HSG login host outside an existing allocation." >&2
  exit 2
fi
if [[ ! -f "$flint_image" || ! -f "$flint_task_root/tools/run_grpo.sh" || ! -f "$flint_task_root/checkpoint/config.json" ]]; then
  echo "The container image, staged launcher, and four-layer checkpoint must exist before launching." >&2
  exit 2
fi
if [[ -e "$flint_task_root/RL/tools/generate_fingerprint.py" ]]; then
  echo "Stage the private runtime source with the stock fingerprint helper omitted, matching the custom image contract." >&2
  exit 2
fi

# An unnamed Pyxis container gives this allocation a fresh writable instance.
# No existing named container or allocation is reused.
exec srun \
  --partition=batch \
  --qos=interactive \
  --account=coreai_dlalgo_nemorl \
  --job-name=flint-grpo-4layer \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=4 \
  --cpus-per-task=32 \
  --mem=256G \
  --time=02:00:00 \
  --export=ALL \
  --container-image="$flint_image" \
  --container-mounts="$flint_task_root:/flint-work" \
  --container-workdir=/flint-work/RL \
  --no-container-mount-home \
  --no-container-entrypoint \
  --container-writable \
  --pty \
  /bin/bash /flint-work/tools/run_grpo.sh "$@"
