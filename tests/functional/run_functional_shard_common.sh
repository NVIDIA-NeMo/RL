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

# Common boilerplate for functional test shard scripts.
# Source this file at the top of each L1_Functional_Tests_*.sh shard script.

set -xeuo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../..)

cd ${PROJECT_ROOT}

tests_run=0

# run_test [fast] <command...>
# - "run_test fast <cmd>" = always runs (both fast and full modes)
# - "run_test <cmd>"      = only runs in full mode; skipped when FAST=1
run_test() {
    if [[ "$1" == "fast" ]]; then
        shift
        tests_run=$((tests_run + 1))
        time "$@"
    elif [[ "${FAST:-0}" == "1" ]]; then
        echo "FAST: Skipping: $*"
    else
        tests_run=$((tests_run + 1))
        time "$@"
    fi
}

combine_functional_coverage() {
    cd ${PROJECT_ROOT}/tests
    if compgen -G ".coverage*" > /dev/null; then
        coverage combine .coverage*
    elif [[ "${FAST:-0}" == "1" && "${tests_run}" == "0" ]]; then
        echo "FAST: No tests selected for this shard; skipping coverage combine."
    else
        echo "[ERROR]: No coverage data files were produced."
        exit 1
    fi
}
