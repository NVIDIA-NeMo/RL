#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
set -euo pipefail
[[ "$USER" == yuekaiz ]]
PR3_GEMMA_REPO=/lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr5090-streaming
cd "$PR3_GEMMA_REPO"
mkdir -p validation/pr3/gemma_e2b/slurm
exec 9>validation/pr3/gemma_e2b/slurm/acquire.lock
flock -n 9
if [[ -f validation/pr3/gemma_e2b/slurm/debug-job-id ]]; then
  echo 'An allocation has already been recorded; inspect it before acquiring another.'
  cat validation/pr3/gemma_e2b/slurm/debug-job-id
  exit 1
fi
PR3_GEMMA_JOB=$(CONTAINER=/lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.1030.sqsh \
  BASE_LOG_DIR="$PR3_GEMMA_REPO/validation/pr3/gemma_e2b/slurm" \
  MOUNTS=/lustre:/lustre COMMAND= CPUS_PER_WORKER=128 \
  sbatch --parsable --nodes=1 --account=coreai_dlalgo_nemorl \
    --job-name=yuekaiz-pr5090-gemma-debug --partition=interactive \
    --time=4:00:00 --gres=gpu:8 \
    --output=validation/pr3/gemma_e2b/slurm/slurm-%j.out \
    --error=validation/pr3/gemma_e2b/slurm/slurm-%j.err ray.sub)
printf '%s\n' "$PR3_GEMMA_JOB" | tee validation/pr3/gemma_e2b/slurm/debug-job-id
