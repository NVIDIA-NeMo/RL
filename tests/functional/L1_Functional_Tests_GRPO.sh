# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

# This test is intentionally not run with uv run --no-sync to verify that the frozen environment is working correctly.
run_test      bash ./tests/functional/grpo_frozen_env.sh

run_test fast uv run --no-sync bash ./tests/functional/gdpo.sh
run_test fast uv run --no-sync bash ./tests/functional/gdpo_async_grpo.sh
run_test fast uv run --no-sync bash ./tests/functional/grpo.sh
run_test fast uv run --no-sync bash ./tests/functional/grpo_fsdp2.sh
run_test      uv run --no-sync bash ./tests/functional/grpo_multiple_dataloaders.sh
run_test      uv run --no-sync bash ./tests/functional/grpo_multiturn.sh
run_test      uv run --no-sync bash ./tests/functional/grpo_non_colocated.sh
run_test      uv run --no-sync bash ./tests/functional/grpo_rm_env.sh
run_test fast uv run --no-sync bash ./tests/functional/grpo_topp_topk.sh
run_test      uv run --no-sync bash ./tests/functional/prorlv2.sh
run_test      uv run --no-sync bash ./tests/functional/vlm_grpo.sh

cd ${PROJECT_ROOT}/tests
coverage combine .coverage*
