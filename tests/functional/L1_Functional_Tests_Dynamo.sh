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

#!/bin/bash
set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
cd "${PROJECT_ROOT}"

# run_test [fast] <command...>
# - "run_test fast <cmd>" = always runs (both fast and full modes)
# - "run_test <cmd>"      = only runs in full mode; skipped when FAST=1
run_test() {
    if [[ "$1" == "fast" ]]; then
        shift
        time "$@"
    elif [[ "${FAST:-0}" == "1" ]]; then
        echo "FAST: Skipping: $*"
    else
        time "$@"
    fi
}

# Dynamo overlays: native SingleController tests rerun with policy.generation.dynamo=true.
# Each keeps the native test's checks and adds check_dynamo_overlay.sh's.
# Fleet health on: the lane's only run of its probe and serving-shard push behind real frontends.
run_test fast uv run --no-sync bash ./tests/functional/grpo_dp_single_controller.sh ++policy.generation.dynamo=true ++async_rl.generation_fleet_health.enabled=true
run_test fast uv run --no-sync bash ./tests/functional/grpo_async_gym_single_controller.sh ++token_capture.enabled=true ++policy.generation.dynamo=true ++policy.generation.dynamo_cfg.frontend_args.num_frontends=2
# Gym's default mode: token arrays come back in the reply's nvext.nemo_rl.
run_test fast uv run --no-sync bash ./tests/functional/grpo_async_gym_single_controller.sh ++policy.generation.dynamo=true
run_test fast uv run --no-sync bash ./tests/functional/grpo_dp_single_controller_reload_refit.sh ++policy.generation.dynamo=true
# Forest training: token-capture calls that may continue an earlier call.
run_test fast uv run --no-sync bash ./tests/functional/grpo_async_gym_single_controller_context_compaction.sh ++policy.generation.dynamo=true
# Router replay with deferred routes: the engine stages each captured call's routed experts.
run_test fast uv run --no-sync bash ./tests/functional/grpo_async_gym_single_controller.sh ++token_capture.enabled=true policy.model_name=silence09/DeepSeek-R1-Small-2layers ++policy.router_replay.enabled=true ++token_capture.defer_routed_experts_to_policy=true ++policy.generation.dynamo=true
# Turn recovery: Gym checkpoint + TQ snapshot, kill, restore, re-sent calls.
run_test fast uv run --no-sync bash ./tests/functional/grpo_async_gym_single_controller_turn_recovery.sh ++policy.generation.dynamo=true
# Prefix recovery: a checkpoint cuts running calls; restore resumes them from the saved tokens.
run_test fast uv run --no-sync bash ./tests/functional/grpo_async_gym_single_controller_prefix_recovery.sh ++policy.generation.dynamo=true
# Router replay across a prefix cut, behind Dynamo: each restored call must stage its saved routes.
# The tiny DeepSeek model adds a second routing layout to the native run's Qwen3-MoE build.
run_test fast uv run --no-sync bash ./tests/functional/grpo_router_replay_prefix_recovery_single_controller_1n2g.sh ++policy.generation.dynamo=true policy.model_name=silence09/DeepSeek-R1-Small-2layers policy.tokenizer.name=silence09/DeepSeek-R1-Small-2layers
# Fleet health: a lost shard leaves the frontends' routing, and a restarted one joins it after a refit.
# Both need 3 GPUs and skip below that, as the native variants do. A skip and a pass
# look the same in the lane's verdict, so say which it was in the run summary.
_DYNAMO_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l) || _DYNAMO_GPUS=0
if (( _DYNAMO_GPUS < 3 )); then
    echo "::warning title=Dynamo fleet health tests skipped::A shard loss behind Dynamo needs >= 3 GPUs; this runner has ${_DYNAMO_GPUS}. Those tests self-skipped and this lane proves nothing about them."
fi
run_test uv run --no-sync bash ./tests/functional/grpo_sc_generation_shard_recovery.sh ++policy.generation.dynamo=true
run_test env RESTART_DEAD_SHARDS=true uv run --no-sync bash ./tests/functional/grpo_sc_generation_shard_recovery.sh ++policy.generation.dynamo=true

cd "${PROJECT_ROOT}/tests"
if compgen -G ".coverage*" > /dev/null; then
    coverage combine .coverage*
fi
