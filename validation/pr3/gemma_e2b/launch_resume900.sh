#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
set -euo pipefail
[[ "$USER" == yuekaiz ]]
cd /lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr5090-streaming
PR3_GEMMA_JOB=$1
[[ "$PR3_GEMMA_JOB" =~ ^[0-9]+$ ]]
PR3_GEMMA_BASE=validation/pr3/gemma_e2b/slurm
exec 9>"$PR3_GEMMA_BASE/resume900-launch-$PR3_GEMMA_JOB.lock"
flock -n 9
[[ ! -e "$PR3_GEMMA_BASE/resume900-exit-$PR3_GEMMA_JOB" ]]
[[ -f "$PR3_GEMMA_BASE/$PR3_GEMMA_JOB-logs/STARTED_RAY_HEAD" ]]
[[ ! -e "$PR3_GEMMA_BASE/resume650/HOLD" ]]
set +e
COMMAND='bash /lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr5090-streaming/validation/pr3/gemma_e2b/run_resume900.sh' \
  bash "$PR3_GEMMA_JOB-attach.sh" >"$PR3_GEMMA_BASE/resume900-driver-$PR3_GEMMA_JOB.log" 2>&1
PR3_GEMMA_EXIT=$?
set -e
printf '%s\n' "$PR3_GEMMA_EXIT" >"$PR3_GEMMA_BASE/resume900-exit-$PR3_GEMMA_JOB"
exit "$PR3_GEMMA_EXIT"
