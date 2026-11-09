#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
set -euo pipefail
[[ "$USER" == yuekaiz ]]
PR3_GEMMA_REPO=/lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr5090-streaming
cd "$PR3_GEMMA_REPO"
PR3_GEMMA_JOB=$(< validation/pr3/gemma_e2b/slurm/debug-job-id)
[[ "$PR3_GEMMA_JOB" =~ ^[0-9]+$ ]]
[[ -f "validation/pr3/gemma_e2b/slurm/$PR3_GEMMA_JOB-logs/STARTED_RAY_HEAD" ]]
[[ -f "$PR3_GEMMA_JOB-attach.sh" ]]
set +e
COMMAND="bash $PR3_GEMMA_REPO/validation/pr3/gemma_e2b/run.sh" \
  bash "$PR3_GEMMA_JOB-attach.sh" >validation/pr3/gemma_e2b/slurm/driver.log 2>&1
PR3_GEMMA_EXIT=$?
set -e
printf '%s\n' "$PR3_GEMMA_EXIT" >validation/pr3/gemma_e2b/slurm/driver-exit-code
if [[ "$PR3_GEMMA_EXIT" == 0 ]]; then
  /usr/bin/python3 - <<'PY'
import json
from pathlib import Path
p=Path('validation/pr3/gemma_e2b/results/streaming-la0/checkpoints/latest_checkpoint_status.json')
assert json.loads(p.read_text())['last_checkpoint_step'] == 150
assert Path('validation/pr3/gemma_e2b/results/curve-comparison/comparison.png').is_file()
PY
  # Only release this experiment's debug allocation after training and plotting.
  scancel "$PR3_GEMMA_JOB"
fi
exit "$PR3_GEMMA_EXIT"
