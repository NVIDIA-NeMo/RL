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
[[ -f /lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.1030.sqsh ]]
mkdir -p validation/pr3/slurm
exec 9>validation/pr3/slurm/resume600-submit.lock
flock -n 9
record=validation/pr3/slurm/resume600-submitted.tsv
[[ ! -e "$record" ]] || {
  echo "Existing resume600 job record: $record; inspect it before another submission." >&2
  exit 1
}
export PR3_MODE=exploratory-train
export PR3_CONFIG=validation/pr3/streaming_qwen35_exploratory_600.yaml
export PR3_TARGET_STEPS=600 PR3_MIN_RESUME_STEP=300
dependency=()
for segment in 1 2 3 4; do
  job=$(sbatch --parsable --export=ALL "${dependency[@]}" \
    --job-name="yuekaiz-pr5090-stream-la0-600-seg$segment" \
    --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"Resume existing exploratory streaming LA0 from300 to600; same checkpoint and W&B run"}}' \
    --output="$PWD/validation/pr3/slurm/slurm-%j.out" \
    --error="$PWD/validation/pr3/slurm/slurm-%j.err" validation/pr3/batch.sh)
  job=${job%%;*}
  [[ "$job" =~ ^[0-9]+$ ]]
  printf '%s\t%s\n' "resume600-segment-$segment" "$job" | tee -a "$record"
  printf '%s\t%s\n' "resume600-segment-$segment" "$job" >>validation/pr3/slurm/submitted.tsv
  dependency=(--dependency="afterok:$job")
done
