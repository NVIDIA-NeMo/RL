#!/bin/bash
set -euo pipefail
[[ "$USER" == yuekaiz ]]
cd /lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr5090-streaming
[[ -f /lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.1030.sqsh ]]
mkdir -p validation/pr3/slurm
mode=${1:?Choose parity or train}
args=()
if [[ "$mode" == parity ]]; then
  nodes=2
elif [[ "$mode" == train ]]; then
  nodes=3
  if [[ -n "${PR3_DEPENDENCY:-}" ]]; then
    args+=(--dependency="afterok:$PR3_DEPENDENCY")
  fi
else
  echo 'Choose parity or train' >&2
  exit 2
fi
PR3_MODE="$mode" sbatch --parsable "${args[@]}" --nodes="$nodes" \
  --job-name="yuekaiz-pr5090-${mode}-la0" \
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"Streaming GRPO qualification: rollout, refit and checkpoint intervals"}}' \
  --output="$PWD/validation/pr3/slurm/slurm-%j.out" \
  --error="$PWD/validation/pr3/slurm/slurm-%j.err" validation/pr3/batch.sh
