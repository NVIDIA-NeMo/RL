#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=cpu
#SBATCH --time=00:12:00
#SBATCH --job-name=yuekaiz-pr5090-expanded-curves
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
PR3_REPO=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/single_controller/RL-pr5090-streaming
PR3_IMAGE=/lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.1030.sqsh
srun --cpu-bind=none --no-container-mount-home \
  --container-image="$PR3_IMAGE" --container-mounts=/lustre:/lustre \
  --container-workdir="$PR3_REPO" bash -c '
    set -euo pipefail
    export PYTHONPATH="$PWD:${PYTHONPATH:-}" OMP_NUM_THREADS=2
    export MPLCONFIGDIR="$PWD/validation/pr3/slurm/matplotlib-expanded"
    /opt/nemo_rl_venv/bin/python -m validation.pr3.plot_expanded_comparison
  '
