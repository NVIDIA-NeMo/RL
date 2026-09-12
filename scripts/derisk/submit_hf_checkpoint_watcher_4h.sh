#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#
# Submit a serialized 20-cycle watcher for a malformed-v3 training result.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_path="$(cd "${script_dir}/../.." && pwd)"

run_root="${RUN_ROOT:?Set RUN_ROOT to the training result directory}"
run_name="$(basename "${run_root}")"
log_dir="${HF_WATCHER_LOG_DIR:-${run_root}/hf_watcher_logs}"
mkdir -p "${log_dir}"

submit_args=(
  --parsable
  --array=0-19%1
  --time=04:00:00
  --account="${SLURM_ACCOUNT:-nemotron_omni_vision}"
  --partition="${SLURM_PARTITION:-batch}"
  --job-name="hf-${run_name:0:90}"
  --output="${log_dir}/%A_%a.out"
  --error="${log_dir}/%A_%a.out"
  --export="ALL,RUN_ROOT=${run_root},MAX_STEPS=${MAX_STEPS:-60},SAVE_PERIOD=${SAVE_PERIOD:-5}"
  "${repo_path}/scripts/derisk/watch_and_convert_megatron_checkpoints_4h.sh"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'sbatch '
  printf '%q ' "${submit_args[@]}"
  printf '\n'
  exit 0
fi

sbatch "${submit_args[@]}"
