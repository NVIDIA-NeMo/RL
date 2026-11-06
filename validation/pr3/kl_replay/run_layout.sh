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
source /lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr5090-streaming/validation/pr3/environment.sh
[[ "$USER" == root && -n "${SLURM_JOB_ID:-}" ]]
export PR3_KL_ROOT="$PR3_REPO/validation/pr3/kl_replay/results/production-layout-$SLURM_JOB_ID"
export PR3_KL_CHECKPOINT="$PR3_REPO/validation/pr3/kl_replay/results/pinned_checkpoint/step_210"
[[ ! -e "$PR3_KL_ROOT" ]]
mkdir -p "$PR3_KL_ROOT"
# Reuse the exact captured responses; no rollout, checkpoint writes or W&B.
ln -s "$PR3_REPO/validation/pr3/kl_replay/results/ep16-21373254/stream-capture" \
  "$PR3_KL_ROOT/stream-capture"
sha256sum nemo_rl/models/policy/workers/automodel_policy_worker.py \
  tests/functional/automodel_sc_full_batch_parity.py validation/pr3/kl_replay/*.py \
  validation/pr3/kl_replay/*.sh validation/pr3/kl_replay/*.yaml >"$PR3_KL_ROOT/source-hashes.txt"
git rev-parse HEAD >"$PR3_KL_ROOT/source-head.txt"
uv run --no-project --python /opt/nemo_rl_venv/bin/python python \
  validation/pr3/kl_replay/distributed.py \
  --replay-script validation/pr3/kl_replay/production_layout.py
sha256sum -c "$PR3_KL_ROOT/source-hashes.txt"
