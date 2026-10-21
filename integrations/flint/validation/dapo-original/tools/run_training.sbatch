#!/usr/bin/env bash
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
#SBATCH --job-name=flint-dapo50-aime24
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --export=ALL
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/wenweng/Automodel/flint-dapo50-49layer-20261019/evidence/allocation-%j.log

set -euo pipefail
set +x
umask 077
flint_base=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/wenweng/Automodel/flint-grpo-49layer-20261019
flint_run=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/wenweng/Automodel/flint-dapo50-49layer-20261019
if [[ -e "$flint_run/STOP_REQUESTED.json" ]]; then
  printf "%s\n" "Run stopped by user; new authorization is required to resume." >&2
  exit 0
fi
flint_model=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/wenweng/Automodel/cedargrove-model_vv1
flint_image=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/wenweng/Automodel/flint-nemorl-20261019/images/flint-nemorl-terry-20261019-arm64-native-01.sqsh
flint_secrets=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/wenweng/nemo-rl-internal/secrets.sh
test -f "$flint_run/config/grpo-flint-dapo50.yaml"
test -f "$flint_run/tools/preflight_dapo.py"
test -f "$flint_run/tools/final_audit.py"
test -f "$flint_run/tools/launch_grpo_diagnostic.py"
test -x "$flint_run/tools/python-sglang-diagnostic.sh"
test -f "$flint_run/tools/watch_sglang_faults.py"
test -f "$flint_base/tools/launch_grpo.py"
test ! -e "$flint_base/RL/tools/generate_fingerprint.py"
test -f "$flint_model/config.json"
test -f "$flint_image"
[[ "${SLURM_JOB_NUM_NODES:?}" == 8 ]]
source "$flint_secrets" >/dev/null 2>&1
set +x
export FLINT_ATTEMPT_ID="${SLURM_JOB_ID}-restart${SLURM_RESTART_COUNT:-0}"
flint_attempt="$flint_run/evidence/jobs/$FLINT_ATTEMPT_ID"
mkdir -p "$flint_run/evidence/jobs"
# A restart gets its own marker directory; existing markers never open the gate.
mkdir "$flint_attempt"
flint_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export FLINT_HEAD_HOST=${flint_nodes%%$'\n'*}
export FLINT_EXPECTED_NODES=8 FLINT_GPUS_PER_NODE=4
export FLINT_NODE_CPUS=64 FLINT_RAY_PORT=1200
printf '%s\n' "$flint_nodes" >"$flint_attempt/slurm-nodes.txt"
flint_srun_pid=
flint_cleanup() {
  local flint_status=$?
  trap - EXIT TERM INT HUP
  if [[ -n "$flint_srun_pid" ]] && kill -0 "$flint_srun_pid" 2>/dev/null; then
    kill -TERM "$flint_srun_pid" 2>/dev/null || true
    wait "$flint_srun_pid" 2>/dev/null || true
  fi
  printf '%s\n' "$flint_status" >"$flint_attempt/allocation.exit-code"
  exit "$flint_status"
}
trap flint_cleanup EXIT
trap 'exit 143' TERM HUP
trap 'exit 130' INT
srun --ntasks=8 --ntasks-per-node=1 --cpus-per-task=64 --gpus-per-node=4 \
  --kill-on-bad-exit=1 --wait=120 \
  --output="$flint_attempt/node-%t.log" \
  --container-image="$flint_image" \
  --container-mounts="$flint_base:/flint-work:ro,$flint_run:/flint-run,$flint_model:/flint-model:ro" \
  --container-workdir=/flint-work/RL \
  --no-container-mount-home --no-container-entrypoint --container-writable \
  /bin/bash /flint-run/tools/run_node.sh &
flint_srun_pid=$!
set +e
wait "$flint_srun_pid"
flint_status=$?
set -e
flint_srun_pid=
if [[ "$flint_status" == 0 && -f "$flint_attempt/window-result.json" ]]; then
  flint_next_step=$(python3 - "$flint_attempt/window-result.json" <<'PY'
import json, sys
record = json.load(open(sys.argv[1]))
if record.get("status") != "pass":
    raise RuntimeError("Window checkpoint gate did not pass")
step = record["checkpoint_step"]
if record["disposition"] == "continue" and 0 < step < 50:
    print(step)
elif record["disposition"] != "complete" or step != 50:
    raise RuntimeError("Invalid continuation disposition")
PY
  )
  if [[ -n "$flint_next_step" && ! -e "$flint_run/STOP_REQUESTED.json" ]]; then
    # The complete srun step has exited before submitting. afterok keeps the
    # successor pending until this allocation has finished successfully.
    printf '%s\n' "$SLURM_JOB_ID" >"$flint_attempt/continuation-parent-job.txt"
    if sbatch --parsable --partition=batch --qos=normal --time=04:00:00 \
      --nodes=8 --gpus-per-node=4 --dependency="afterok:$SLURM_JOB_ID" \
      --export="ALL,FLINT_PREVIOUS_JOB_ID=$SLURM_JOB_ID,FLINT_EXPECTED_RESUME_STEP=$flint_next_step" \
      "$flint_run/tools/run_training.sbatch" >"$flint_attempt/continuation-submission.txt" \
      2>"$flint_attempt/continuation-submission.stderr"; then
      printf '0\n' >"$flint_attempt/continuation-submission.exit-code"
    else
      flint_submit_status=$?
      printf '%s\n' "$flint_submit_status" >"$flint_attempt/continuation-submission.exit-code"
      exit "$flint_submit_status"
    fi
  fi
fi
exit "$flint_status"
