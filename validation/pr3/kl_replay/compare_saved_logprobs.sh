#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=cpu_interactive
#SBATCH --time=00:15:00
#SBATCH --job-name=yuekaiz-pr5090-saved-lp
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
PR3_REPO=/lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr5090-streaming
PR3_IMAGE=/lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.1030.sqsh
srun --cpu-bind=none --no-container-mount-home \
  --container-image="$PR3_IMAGE" --container-mounts=/lustre:/lustre \
  --container-workdir="$PR3_REPO" bash -c '
    set -euo pipefail
    export UV_NO_SYNC=1 UV_NO_PROGRESS=1 OMP_NUM_THREADS=1
    uv run --no-project --python /opt/nemo_rl_venv/bin/python \
      python validation/pr3/kl_replay/compare_saved_logprobs.py \
      validation/pr3/kl_replay/results/production-layout-21391011 \
      --reference validation/pr3/kl_replay/results/ep16-21373254
  '
