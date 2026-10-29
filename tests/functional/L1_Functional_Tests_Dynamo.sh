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
run_test fast uv run --no-sync bash ./tests/functional/grpo_dp_single_controller.sh ++policy.generation.dynamo=true
run_test fast uv run --no-sync bash ./tests/functional/grpo_async_gym_single_controller.sh ++token_capture.enabled=true ++policy.generation.dynamo=true
# Gym's default mode: token arrays come back through the frontend in engine_data.
run_test uv run --no-sync bash ./tests/functional/grpo_async_gym_single_controller.sh ++policy.generation.dynamo=true
run_test fast uv run --no-sync bash ./tests/functional/grpo_dp_single_controller_reload_refit.sh ++policy.generation.dynamo=true
# Router replay with deferred routes: the engine stages each captured call's routed experts.
run_test uv run --no-sync bash ./tests/functional/grpo_async_gym_single_controller.sh ++token_capture.enabled=true policy.model_name=silence09/DeepSeek-R1-Small-2layers ++policy.router_replay.enabled=true ++token_capture.defer_routed_experts_to_policy=true ++policy.generation.dynamo=true

cd "${PROJECT_ROOT}/tests"
if compgen -G ".coverage*" > /dev/null; then
    coverage combine .coverage*
fi
