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

run_test fast uv run --no-sync bash ./tests/functional/grpo_dp_single_controller.sh
run_test fast uv run --no-sync bash ./tests/functional/grpo_async_gym_single_controller.sh
# Full mode only (~10 min): SIGKILLs a generation worker and asserts the job fails fast
# and attributably instead of wedging. This is the ONLY end-to-end check of the
# containment behaviour -- without it, a regression that restores the silent wedge is
# caught by nothing, because a wedged job produces no exception and no failing assertion
# anywhere else.
run_test uv run --no-sync bash ./tests/functional/grpo_dp_single_controller_chaos.sh

# grpo_dp_single_controller_chaos.sh again, this time killing a worker that is mid-rollout
# rather than between calls. Registered because pinning the victim state -- which is what
# makes that test reproducible at all -- would otherwise silently drop a scenario the old,
# non-deterministic selection used to hit by chance. The two fail by different routes:
# killing an idle worker leaves the loss to be *detected*, killing a serving one destroys
# an in-flight RPC that surfaces at once (222s vs 12s when measured). A regression in
# either is invisible to the other.
#
# Cheap to add: the serving path fails in seconds, so this is dominated by startup.
run_test env VICTIM_STATE=serving uv run --no-sync bash ./tests/functional/grpo_dp_single_controller_chaos.sh

# Checkpoint save/restore (upstream #3429).
run_test uv run --no-sync bash ./tests/functional/grpo_checkpoint_single_controller.sh

cd ${PROJECT_ROOT}/tests
if compgen -G ".coverage*" > /dev/null; then
    coverage combine .coverage*
fi
