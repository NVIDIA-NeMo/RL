#!/bin/bash
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
set -euo pipefail
set +x
umask 077
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash tools/super_rl/cmh_repro/submit.sh USER_ENV fresh|resume [--submit]" >&2
    exit 2
fi
user_env=$(readlink -f "$1")
mode=$2
[[ "$mode" == fresh || "$mode" == resume ]]
[[ $# -eq 2 || "$3" == --submit ]]
code=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
source "$user_env"
required=(SUPER_RL_ROOT SUPER_RL_MODEL SUPER_RL_DATA SUPER_RL_DATA_SHA256
    SUPER_RL_SCICODE_HDF5 SUPER_RL_PARSER SUPER_RL_TRAIN_IMAGE SUPER_RL_WHEEL
    SUPER_RL_SECRET_FILE SUPER_RL_ACCOUNT SUPER_RL_RUN_NAME WANDB_ENTITY WANDB_PROJECT
    WANDB_RUN_ID SUPER_RL_PARTITION SUPER_RL_QOS SUPER_RL_WALLTIME SUPER_RL_SAFE_DEADLINE
    SUPER_RL_CPU_PARTITION SUPER_RL_CPU_QOS)
for name in "${required[@]}"; do
    value=${!name:-}
    if [[ -z "$value" || "$value" == *REPLACE_ME* || "$value" =~ [[:space:],] ]]; then
        echo "Missing/unfilled/unsupported input: $name" >&2
        exit 2
    fi
done
[[ "$SUPER_RL_ROOT" == /scratch/* && "$SUPER_RL_ROOT" != /scratch/ ]]
for name in SUPER_RL_MODEL SUPER_RL_DATA SUPER_RL_SCICODE_HDF5 SUPER_RL_PARSER SUPER_RL_TRAIN_IMAGE SUPER_RL_WHEEL SUPER_RL_SECRET_FILE; do
    value=${!name}
    [[ "$value" == /scratch/* && -r "$value" ]]
done
[[ "$SUPER_RL_SECRET_FILE" != "$SUPER_RL_ROOT/"* ]]
[[ "$(stat -c %a "$SUPER_RL_SECRET_FILE")" == 600 ]]
[[ "$code" == /scratch/* && ! "$code" =~ [[:space:],] ]]
[[ "$SUPER_RL_DATA_SHA256" =~ ^[a-f0-9]{64}$ ]]
[[ "$SUPER_RL_PARTITION" == batch_long && "$SUPER_RL_QOS" == normal ]]
[[ "$SUPER_RL_WALLTIME" == 2-00:00:00 && "$SUPER_RL_SAFE_DEADLINE" == 01:23:00:00 ]]
revision=$(git -C "$code" rev-parse HEAD)
git -C "$code" merge-base --is-ancestor b41e457589ff9c9040aec739121fa0ba199de4c8 "$revision"
test -z "$(git -C "$code" status --porcelain --untracked-files=normal)"
submodules=$(git -C "$code" submodule status --recursive)
if printf '%s\n' "$submodules" | grep -qE '^[-+U]'; then
    echo "Initialize the exact pinned submodules; see the dependency-access guide." >&2
    exit 2
fi
root=$SUPER_RL_ROOT
if [[ "$mode" == fresh ]]; then
    test ! -e "$root"
else
    cmp "$user_env" "$root/user.env"
    test "$(cat "$root/source.commit")" = "$revision"
    test -f "$root/checkpoints/latest_checkpoint_status.json"
    previous=$(cat "$root/training.jobid")
    [[ "$previous" =~ ^[0-9]+$ ]]
    test -z "$(squeue -h -j "$previous" -o '%i' 2>/dev/null)"
fi
# Reuse the repository's allow-listed submission environment, not an inherited salloc.
clean_env=(env -i "PATH=$PATH" "HOME=$HOME" "USER=$USER" "LOGNAME=$LOGNAME" "LANG=${LANG:-C}")
train_args=(--parsable --account="$SUPER_RL_ACCOUNT" --partition="$SUPER_RL_PARTITION"
    --qos="$SUPER_RL_QOS" --nodes=64 --segment=16 --gpus-per-node=4
    --ntasks-per-node=1 --cpus-per-task=140 --exclusive --time="$SUPER_RL_WALLTIME"
    --no-requeue --export=NONE --job-name="$SUPER_RL_RUN_NAME"
    --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"75","reason":"other","description":"nemo-rl-run-bootstrap"}}')
if [[ $# -eq 2 ]]; then
    "${clean_env[@]}" sbatch --test-only "${train_args[@]}" "$code/tools/super_rl/cmh_repro/job.sbatch" "$root" "$mode"
    echo "Scheduler check only. No job or run directory created."
    exit 0
fi
if [[ "$mode" == fresh ]]; then
    mkdir -m 700 "$root"
    mkdir -p "$root/slurm" "$root/manifests" "$root/runtime" "$root/attempts" "$root/chain"
    install -m 600 "$user_env" "$root/user.env"
    printf '%s\n' "$code" > "$root/source.path"
    printf '%s\n' "$revision" > "$root/source.commit"
    sha256sum "$root/user.env" > "$root/user-env.sha256"
    preflight=$("${clean_env[@]}" sbatch --parsable --account="$SUPER_RL_ACCOUNT" \
        --partition="$SUPER_RL_CPU_PARTITION" --qos="$SUPER_RL_CPU_QOS" \
        --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128G --time=01:00:00 \
        --no-requeue --export=NONE --job-name="$SUPER_RL_RUN_NAME-preflight" \
        --output="$root/slurm/preflight-%j.out" "$code/tools/super_rl/cmh_repro/job.sbatch" "$root" preflight)
    [[ "$preflight" =~ ^[0-9]+$ ]]
    printf '%s\n' "$preflight" > "$root/preflight.jobid"
    train_args+=(--dependency="afterok:$preflight" --kill-on-invalid-dep=yes)
fi
# No automatic retry here: a transport error may conceal a successful submission.
job=$("${clean_env[@]}" sbatch "${train_args[@]}" --output="$root/slurm/train-%j.out" \
    "$code/tools/super_rl/cmh_repro/job.sbatch" "$root" "$mode")
[[ "$job" =~ ^[0-9]+$ ]]
printf '%s\n' "$job" > "$root/training.jobid"
printf 'Training job %s; logs %s/slurm. Native preflight must pass before fresh training starts.\n' "$job" "$root"
