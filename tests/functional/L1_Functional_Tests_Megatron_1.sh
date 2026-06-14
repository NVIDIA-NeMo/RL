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
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../..)

cd ${PROJECT_ROOT}

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

run_test fast uv run --no-sync bash ./tests/functional/audio_grpo_megatron.sh
run_test      uv run --no-sync bash ./tests/functional/grpo_megatron.sh
run_test      uv run --no-sync bash ./tests/functional/grpo_megatron_mbridge_restore.sh
run_test fast uv run --no-sync bash ./tests/functional/grpo_megatron_eagle3_online.sh
run_test      uv run --no-sync bash ./tests/functional/grpo_megatron_generation.sh
run_test      uv run --no-sync bash ./tests/functional/grpo_megatron_generation_non_colocated.sh
run_test      uv run --no-sync bash ./tests/functional/grpo_megatron_generation_async.sh
run_test      uv run --no-sync bash ./tests/functional/grpo_megatron_generation_colocated_async.sh
run_test      uv run --no-sync bash ./tests/functional/grpo_megatron_generation_async_gym.sh
run_test      uv run --no-sync bash ./tests/functional/grpo_megatron_generation_topp_topk.sh
run_test      uv run --no-sync bash ./tests/functional/grpo_megatron_generation_multiturn.sh

cd ${PROJECT_ROOT}/tests
if compgen -G ".coverage*" > /dev/null; then
    coverage combine .coverage*
fi
