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
export PR3_KL_ROOT="$PR3_REPO/validation/pr3/kl_replay/results/ep16-$SLURM_JOB_ID"
export PR3_KL_CHECKPOINT="$PR3_REPO/validation/pr3/kl_replay/results/pinned_checkpoint/step_210"
mkdir -p "$PR3_KL_ROOT"
# Validation scripts alone may change; product worker and existing numerical
# harness remain identical to the source used by the ongoing convergence run.
sha256sum nemo_rl/models/policy/workers/automodel_policy_worker.py \
  tests/functional/automodel_sc_full_batch_parity.py validation/pr3/kl_replay/*.py \
  validation/pr3/kl_replay/*.sh validation/pr3/kl_replay/*.yaml >"$PR3_KL_ROOT/source-hashes.txt"
git rev-parse HEAD >"$PR3_KL_ROOT/source-head.txt"
export PR3_KL_PROFILE=0
export PR3_KL_DIR="$PR3_KL_ROOT/stream-capture"
uv run --no-project --python /opt/nemo_rl_venv/bin/python python \
  validation/pr3/kl_replay/capture.py --config validation/pr3/kl_replay/ep16.yaml \
  "logger.log_dir=$PR3_KL_ROOT/stream-logs" >"$PR3_KL_ROOT/stream.log" 2>&1
export PR3_KL_DIR="$PR3_KL_ROOT/full-capture"
uv run --no-project --python /opt/nemo_rl_venv/bin/python python \
  validation/pr3/kl_replay/capture.py --config validation/pr3/kl_replay/ep16.yaml \
  async_rl.min_groups_for_streaming_train=8 \
  "logger.log_dir=$PR3_KL_ROOT/full-logs" >"$PR3_KL_ROOT/full.log" 2>&1
export PR3_KL_PROFILE=1 PR3_KL_PROFILE_ITERATION=2 PR3_AUDIT_DIR="$PR3_KL_ROOT/trace"
export PR3_KL_DIR="$PR3_KL_ROOT/profile-capture"
uv run --no-project --python /opt/nemo_rl_venv/bin/python python \
  validation/pr3/kl_replay/capture.py --config validation/pr3/kl_replay/ep16.yaml \
  "logger.log_dir=$PR3_KL_ROOT/profile-logs" >"$PR3_KL_ROOT/profile.log" 2>&1
uv run --no-project --python /opt/nemo_rl_venv/bin/python python \
  validation/pr3/kl_replay/distributed.py
uv run --no-project --python /opt/nemo_rl_venv/bin/python python \
  validation/pr3/kl_replay/analyze.py "$PR3_KL_ROOT"
sha256sum -c "$PR3_KL_ROOT/source-hashes.txt"
