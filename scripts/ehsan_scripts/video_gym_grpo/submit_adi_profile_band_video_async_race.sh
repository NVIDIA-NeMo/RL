#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_id="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
base_name="${BASE_NAME:-vg16a_adi_rpb_${run_id}}"
code_dir="$(cd "${script_dir}/../../.." && pwd)"
results_dir="${RESULTS_DIR:-${code_dir}/results/${base_name}}"
race_root="${RACE_ROOT:-${results_dir}/scheduler_race}"

export RUN_ID="${run_id}" BASE_NAME="${base_name}" RESULTS_DIR="${results_dir}" RACE_ROOT="${race_root}"

candidates=(
  "nemotron_edge_omni batch"
  "nemotron_edge_omni backfill"
  "nemotron_omni_vision batch"
  "nemotron_omni_vision backfill"
)

for candidate in "${candidates[@]}"; do
  read -r candidate_account candidate_partition <<<"${candidate}"
  echo "SUBMIT_CANDIDATE account=${candidate_account} partition=${candidate_partition}"
  SLURM_ACCOUNT="${candidate_account}" \
  SLURM_PARTITION="${candidate_partition}" \
  bash "${script_dir}/launch_adi_profile_band_video_async.sh"
done
