#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_id="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
base_name="${BASE_NAME:-s35v_sync2_rpb_${run_id}}"
code_dir="$(cd "${script_dir}/../../.." && pwd)"
results_dir="${RESULTS_DIR:-${code_dir}/results/${base_name}}"
race_root="${RACE_ROOT:-${results_dir}/scheduler_race}"

export RUN_ID="${run_id}" BASE_NAME="${base_name}" RESULTS_DIR="${results_dir}" RACE_ROOT="${race_root}"
export CONFIG_IN_CONTAINER=examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-h100-2n8g-megatron-tp4ep4-sync-gym-video-adi-profile-band.v1.yaml
export JOB_NODES=2 JOB_TIME=02:00:00 JOB_CYCLES=1 MAX_NUM_STEPS=2
export WANDB_RUN_NAME=s35v-sync2-rpb-mpo125

candidates=(
  "nemotron_edge_omni interactive"
  "nemotron_omni_vision interactive"
)

for candidate in "${candidates[@]}"; do
  read -r candidate_account candidate_partition <<<"${candidate}"
  echo "SUBMIT_CANDIDATE account=${candidate_account} partition=${candidate_partition}"
  SLURM_ACCOUNT="${candidate_account}" \
  SLURM_PARTITION="${candidate_partition}" \
  bash "${script_dir}/launch_adi_profile_band_video_async.sh"
done
