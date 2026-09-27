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

# Native TQ + metadata-only completed replay recovery (#3480).
run_test fast uv run --no-sync bash ./tests/functional/grpo_dp_single_controller_tq_recovery.sh
# Same recovery flow with Mooncake CPU storage; skips without an RDMA device.
run_test fast uv run --no-sync bash ./tests/functional/grpo_dp_mooncake_tq_recovery.sh
# Deterministic process restart with an admitted group held before canonical TQ
# commit, followed by exact-once redispatch at its stable group ID.
run_test fast uv run --no-sync bash ./tests/functional/grpo_dp_single_controller_unfinished_recovery.sh

# Token-capture (gate-authoritative) path: same SC+Gym smoke with the gate
# custodying token lineage and the finalizer publishing training rows.
run_test uv run --no-sync bash ./tests/functional/grpo_async_gym_single_controller.sh ++token_capture.enabled=true
# Two-process token-capture recovery: preserve one sealed sibling in TQ and
# redispatch only its unfinished peer after restoring the step checkpoint.
run_test fast uv run --no-sync bash ./tests/functional/grpo_async_gym_single_controller_sibling_recovery.sh
# Periodic native-TQ snapshot while a streamed step owns only part of its
# rollout batch, followed by SIGKILL and rollback to the durable trainer anchor.
run_test fast uv run --no-sync bash ./tests/functional/grpo_async_gym_single_controller_streaming_recovery.sh

cd ${PROJECT_ROOT}/tests
if compgen -G ".coverage*" > /dev/null; then
    coverage combine .coverage*
fi
