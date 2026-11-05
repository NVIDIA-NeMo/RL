#!/bin/bash
# Deterministic status collection; no retries, credential access, or job mutation.
set -euo pipefail
cd /lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr5090-streaming
[[ "$USER" == yuekaiz ]]
mkdir -p validation/pr3/slurm
exec 9>validation/pr3/slurm/watch.lock
flock -n 9 || exit 0
for (( poll=0; poll<288; poll++ )); do
  ids=$(cut -f2 validation/pr3/slurm/submitted.tsv | paste -sd, -)
  {
    date --iso-8601=seconds
    squeue -j "$ids" -o '%.16i %.28j %.12T %.12M %.35R'
    states=$(sacct -X -n -j "$ids" --format=JobIDRaw,State,ExitCode,Elapsed -P)
    printf '%s\n' "$states"
    if [[ "$states" =~ FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL ]]; then
      printf '%s\n' "$states" >validation/pr3/slurm/FAILURE_DETECTED
    fi
    for file in validation/pr3/slurm/slurm-*.out; do
      [[ -f "$file" ]] || continue
      printf '%s\n' "$file"
      tail -n 1000 "$file" | rg 'train step [0-9]+/|EP16_FIXED_INPUT_PASSED|PARITY_PASSED|already completed' | tail -n 3 || true
    done
  } >>validation/pr3/slurm/watch.log 2>&1
  # All monitoring is read-only; failed qualifications never release training.
  for (( minute=0; minute<10; minute++ )); do sleep 60; done
done
