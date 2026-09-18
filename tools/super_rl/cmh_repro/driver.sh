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
source "${SUPER_RL_RUNTIME_ENV:?}"
source /opt/nemo-rl/tools/super_rl/cmh_repro/profile.env
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset NVIDIA_API_KEY WANDB_API_KEY HF_TOKEN NEMO_GYM_VLLM_TRANSPORT_LOG
set -a
source "$SUPER_RL_SECRET_FILE"
set +a
: "${NVIDIA_API_KEY:?}" "${WANDB_API_KEY:?}" "${WANDB_RUN_ID:?}"
export WANDB_MODE=online WANDB_NAME="$SUPER_RL_RUN_NAME" WANDB_FORK_ON_RESUME=0
export WANDB_CACHE_DIR="$SUPER_RL_RUN_OUTPUT/wandb-cache"
export WANDB_CONFIG_DIR="/tmp/super-rl-$NRL_SLURM_JOB_ID-wandb-config"
mkdir -m 700 -p "$WANDB_CONFIG_DIR"
if [[ "$SUPER_RL_RUN_MODE" == resume ]]; then
    export WANDB_RESUME=must
    /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python \
        /opt/nemo-rl/tools/super_rl/cmh_repro/validate_resume.py \
        "$SUPER_RL_ROOT/checkpoints" "$SUPER_RL_RUN_OUTPUT/resume-receipt.json"
    if [[ "$(jq -r .current_step "$SUPER_RL_RUN_OUTPUT/resume-receipt.json")" -ge 100 ]]; then
        echo ABSOLUTE_TARGET_ALREADY_COMPLETE
        exit 0
    fi
else
    export WANDB_RESUME=never
    test ! -e "$SUPER_RL_ROOT/checkpoints/latest_checkpoint_status.json"
fi
cd /opt/nemo-rl
exec /opt/nemo_rl_venv/bin/python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config "$SUPER_RL_CONFIG" checkpointing.load_replay_buffer=false \
    "logger.log_dir=$SUPER_RL_RUN_OUTPUT/logs" \
    "env.nemo_gym.nemo_gym_log_dir=$SUPER_RL_RUN_OUTPUT/logs/nemo_gym" \
    "env.nemo_gym.results_dir=$SUPER_RL_RUN_OUTPUT/gym-results"
