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
root=$(readlink -f "$1")
mode=$2
source "$root/user.env"
test "$(readlink -f "$SUPER_RL_ROOT")" = "$root"
sha256sum --check --status "$root/user-env.sha256"
code=$(cat "$root/source.path")
test "$(git -C "$code" rev-parse HEAD)" = "$(cat "$root/source.commit")"
test -z "$(git -C "$code" status --porcelain --untracked-files=normal)"
export SUPER_RL_ROOT="$root" SUPER_RL_RUNTIME_ENV="$root/user.env"
export SLURM_EXPORT_ENV=ALL NRL_SLURM_JOB_ID="$SLURM_JOB_ID"
source "$code/tools/super_rl/cmh_repro/profile.env"
mounts="/scratch:/scratch,$code:/opt/nemo-rl:ro"
datasets=/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/datasets
if [[ "$mode" == preflight ]]; then
    test "$(sha256sum "$SUPER_RL_TRAIN_IMAGE" | cut -d' ' -f1)" = c39942d83407b08e0b0ad73cdc2334ddee744ab420cd9c5945a03fe3a0abef41
    srun --export=ALL --input=/dev/null --no-container-mount-home \
        --container-image="$SUPER_RL_TRAIN_IMAGE" --container-mounts="$mounts" --container-workdir=/opt/nemo-rl \
        bash -c 'set -euo pipefail
          uv run --no-project /opt/nemo_rl_venv/bin/python tools/super_rl/stage_gym.py \
            --source /opt/nemo-rl/3rdparty/Gym-workspace/Gym --output "$SUPER_RL_ROOT/runtime/gym"
          /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python \
            tools/super_rl/build_mcore_helpers.py build \
            --source /opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/datasets \
            --output "$SUPER_RL_ROOT/runtime/mcore-datasets"'
    mounts+=",$root/runtime/gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:ro,$root/runtime/mcore-datasets:$datasets:ro"
    srun --export=ALL --input=/dev/null --no-container-mount-home \
        --container-image="$SUPER_RL_TRAIN_IMAGE" --container-mounts="$mounts" --container-workdir=/opt/nemo-rl \
        bash -c 'set -euo pipefail; set +x
          export SUPER_RL_SETUP_ATTEMPT=native
          bash tools/super_rl/cmh_repro/node.sh
          uv run --no-project /opt/nemo_rl_venv/bin/python tools/super_rl/cmh_repro/preflight.py
          unset NVIDIA_API_KEY WANDB_API_KEY
          set -a; source "$SUPER_RL_SECRET_FILE"; set +a
          : "${NVIDIA_API_KEY:?}" "${WANDB_API_KEY:?}"
          /opt/gym_venvs/responses_api_models/inference_provider/.venv/bin/python tools/super_rl/cmh_repro/judge_canary.py'
    echo CMH_REPRO_NATIVE_PREFLIGHT_COMPLETE
    exit 0
fi
[[ "$mode" == fresh || "$mode" == resume ]]
exec 9>"$root/writer.lock"
flock -n 9 || { echo "Another writer owns this run root" >&2; exit 1; }
jq -e '.complete and .total_gpus == 256 and .num_steps == 100' "$root/manifests/preflight.json" >/dev/null
jq -e '.complete and .all_valid and .reasoning_present' "$root/manifests/judge-canary.json" >/dev/null
test "$(sha256sum "$SUPER_RL_DATA" | cut -d' ' -f1)" = "$SUPER_RL_DATA_SHA256"
test "${SLURM_JOB_NUM_NODES:?}" -eq 64
test "$SLURM_JOB_ACCOUNT" = "$SUPER_RL_ACCOUNT"
test "$SLURM_JOB_PARTITION" = "$SUPER_RL_PARTITION"
export SUPER_RL_RUN_MODE="$mode" SUPER_RL_RUN_OUTPUT="$root/attempts/$SLURM_JOB_ID"
mkdir -p "$SUPER_RL_RUN_OUTPUT" "$root/ray-logs"
export GPUS_PER_NODE=4 CPUS_PER_WORKER=140 NRL_CONTAINER_WORKDIR=/opt/nemo-rl
export BASE_LOG_DIR="$root/ray-logs" CONTAINER="$SUPER_RL_TRAIN_IMAGE"
unset SANDBOX_CONTAINER SANDBOX_COMMAND SANDBOX_EXTRA_MOUNTS SANDBOX_ENV_VARS
export MOUNTS="$mounts,$root/runtime/gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:ro,$root/runtime/mcore-datasets:$datasets:ro"
export SETUP_COMMAND="SUPER_RL_SETUP_ATTEMPT=\$(date +%s%N) bash /opt/nemo-rl/tools/super_rl/cmh_repro/node.sh"
export COMMAND="bash /opt/nemo-rl/tools/super_rl/cmh_repro/driver.sh"
cd "$root/chain"
# The same immutable ray.sub used by the proven production source.
source "$code/ray.sub"
