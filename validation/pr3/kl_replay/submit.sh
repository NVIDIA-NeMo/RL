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
set -euo pipefail
[[ "$USER" == yuekaiz ]]
cd /lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr5090-streaming
mkdir -p validation/pr3/kl_replay/slurm
export CONTAINER=/lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.1030.sqsh
export MOUNTS=/lustre:/lustre
export BASE_LOG_DIR="$PWD/validation/pr3/kl_replay/slurm"
export GPUS_PER_NODE=8 CPUS_PER_WORKER=128 OMP_NUM_THREADS=4
export COMMAND="bash '$PWD/validation/pr3/kl_replay/run.sh'"
sbatch --parsable --nodes=3 --ntasks-per-node=1 --cpus-per-task=128 \
  --gres=gpu:8 --exclusive --account=coreai_dlalgo_nemorl --partition=batch \
  --time=02:00:00 --job-name=yuekaiz-pr5090-kl-replay-ep16 \
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"Fixed-checkpoint streaming diagnostic: refit, rollout, profiling and numerical replay"}}' \
  --output="$BASE_LOG_DIR/slurm-%j.out" --error="$BASE_LOG_DIR/slurm-%j.err" ray.sub
