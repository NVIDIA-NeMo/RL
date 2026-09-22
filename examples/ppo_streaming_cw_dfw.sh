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

# Submit from the cw-dfw login host, inside the checkout:
#   bash examples/ppo_streaming_cw_dfw.sh
# Defaults: two exclusive 8-GPU nodes, three PPO steps, a 32-prompt-group readiness
# threshold, one policy epoch, two full-batch value epochs, async lag one.
# Policy and value share the training node; vLLM uses the other node. The
# recipe's "noncolocated" name refers to generation, not policy/value placement.
#
# Optional environment overrides:
#   PPO_MAX_STEPS=10 PPO_STREAMING_GROUPS=64 bash examples/ppo_streaming_cw_dfw.sh
#   PPO_CONTAINER=/shared/nemo_rl.sqsh PPO_SLURM_ACCOUNT=... bash ...
# Additional arguments are Hydra overrides, e.g. policy.generation.max_new_tokens=128.
# The nightly image supplies the driver/actor environments. Its dependency
# fingerprint must match this checkout; no mismatch bypass is enabled here.
# Logs: logs/ppo_streaming_cw_dfw/<job-id>-logs/ray-driver.log and slurm-<job-id>.out.

set -euo pipefail

export PPO_MAX_STEPS=${PPO_MAX_STEPS:-3}
export PPO_STREAMING_GROUPS=${PPO_STREAMING_GROUPS:-32}
export PPO_PROMPTS_PER_STEP=${PPO_PROMPTS_PER_STEP:-256}

for key in PPO_MAX_STEPS PPO_STREAMING_GROUPS PPO_PROMPTS_PER_STEP; do
    if ! [[ ${!key} =~ ^[1-9][0-9]*$ ]]; then
        echo "$key must be a positive integer (got '${!key}')." >&2
        exit 1
    fi
done
if (( PPO_STREAMING_GROUPS >= PPO_PROMPTS_PER_STEP ||
      PPO_PROMPTS_PER_STEP % PPO_STREAMING_GROUPS != 0 )); then
    echo 'PPO_STREAMING_GROUPS must divide PPO_PROMPTS_PER_STEP and be smaller.' >&2
    exit 1
fi

if [[ ${1:-} == --train ]]; then
    shift
    cd /repo
    export PYTHONPATH="/repo${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONUNBUFFERED=1
    export HF_HOME=${HF_HOME:-/home/lustre_home/hf_home}
    export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
    export RAY_DEDUP_LOGS=0

    # Advantage normalization remains a boolean: true normalizes each streaming
    # chunk independently; false disables it. Keep the recipe's default (true).
    exec uv run --no-sync python examples/run_grpo_single_controller.py \
        --config examples/configs/recipes/llm/ppo-qwen2.5-1.5b-gsm8k-2n8g-megatron-valuetp2sp-dynbatch-noncolocated-async-single-controller.yaml \
        ppo.num_prompts_per_step="$PPO_PROMPTS_PER_STEP" \
        ppo.num_generations_per_prompt=1 \
        ppo.max_num_steps="$PPO_MAX_STEPS" \
        ppo.max_num_epochs=1000 \
        ppo.ppo_epochs=1 \
        ppo.critic_ppo_epochs=2 \
        ppo.policy_training_start_step=0 \
        async_rl.min_groups_for_streaming_train="$PPO_STREAMING_GROUPS" \
        async_rl.sampler.name=in_order \
        async_rl.sampler.max_lookahead_versions=1 \
        async_rl.sampler.warmup_lookahead_versions=null \
        async_rl.max_inflight_prompts="$((2 * PPO_PROMPTS_PER_STEP))" \
        async_rl.max_buffered_rollouts="$((2 * PPO_PROMPTS_PER_STEP))" \
        policy.train_global_batch_size="$PPO_PROMPTS_PER_STEP" \
        value.train_global_batch_size="$PPO_PROMPTS_PER_STEP" \
        logger.log_dir="${BASE_LOG_DIR:?}/training" \
        logger.wandb_enabled=false \
        logger.tensorboard_enabled=true \
        checkpointing.enabled=false \
        "$@"
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
PPO_SHARED_ROOT=${PPO_SHARED_ROOT:-$HOME/lustre_home}
PPO_SHARED_ROOT=$(cd -- "$PPO_SHARED_ROOT" && pwd -P)
export CONTAINER=${PPO_CONTAINER:-$PPO_SHARED_ROOT/sqsh/nemo_rl.0922.sqsh}
export BASE_LOG_DIR=${PPO_LOG_DIR:-$REPO_ROOT/logs/ppo_streaming_cw_dfw}
export GPUS_PER_NODE=8
export HF_HOME=/home/lustre_home/hf_home
export HF_DATASETS_CACHE=$HF_HOME/datasets
export RAY_DEDUP_LOGS=0

if [[ ! -f "$CONTAINER" || ! -f "$REPO_ROOT/ray.sub" ]]; then
    echo "Expected container $CONTAINER and launcher $REPO_ROOT/ray.sub." >&2
    exit 1
fi

# ray.sub uses the original absolute checkout/log paths for shared-file signals;
# the driver and all actor processes import the edited source through /repo.
export MOUNTS="$PPO_SHARED_ROOT:$PPO_SHARED_ROOT,$PPO_SHARED_ROOT:/home/lustre_home,$REPO_ROOT:/repo"
if [[ -n ${PPO_EXTRA_MOUNTS:-} ]]; then
    MOUNTS+=",$PPO_EXTRA_MOUNTS"
fi
printf -v COMMAND '%q ' bash /repo/examples/ppo_streaming_cw_dfw.sh --train "$@"
export COMMAND
mkdir -p "$BASE_LOG_DIR"
cd "$REPO_ROOT"

echo "Streaming PPO: $PPO_MAX_STEPS steps; $PPO_STREAMING_GROUPS ready groups / $PPO_PROMPTS_PER_STEP groups per step; value epochs run on the full batch."
echo "Container: $CONTAINER"
echo "Logs: $BASE_LOG_DIR"
exec sbatch \
    --nodes=2 \
    --gpus-per-node=8 \
    --exclusive \
    --account="${PPO_SLURM_ACCOUNT:-coreai_dlalgo_nemorl}" \
    --partition="${PPO_SLURM_PARTITION:-batch}" \
    --job-name=ppo.streaming-dev \
    --time="${PPO_TIME_LIMIT:-01:00:00}" \
    --output="$BASE_LOG_DIR/slurm-%j.out" \
    --export=ALL \
    ray.sub
