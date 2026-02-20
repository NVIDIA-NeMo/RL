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
uv run tests/unit/prepare_unit_test_assets.py
uv run --no-sync bash -x ./tests/run_unit.sh unit/models/policy/ --cov=nemo_rl --cov-report=term-missing --cov-report=json --hf-gated

# Check and run mcore tests
exit_code=$(uv run --extra mcore pytest tests/unit/models/policy/ --collect-only --hf-gated --mcore-only -q >/dev/null 2>&1; echo $?)
if [[ $exit_code -eq 5 ]]; then
    echo "No mcore tests to run"
else
    uv run --extra mcore bash -x ./tests/run_unit.sh unit/models/policy/ --cov=nemo_rl --cov-append --cov-report=term-missing --cov-report=json --hf-gated --mcore-only
fi

# Check and run automodel tests
exit_code=$(uv run --extra automodel pytest tests/unit/models/policy/ --collect-only --hf-gated --automodel-only -q >/dev/null 2>&1; echo $?)
if [[ $exit_code -eq 5 ]]; then
    echo "No automodel tests to run"
else
    uv run --extra automodel bash -x ./tests/run_unit.sh unit/models/policy/ --cov=nemo_rl --cov-append --cov-report=term-missing --cov-report=json --hf-gated --automodel-only
fi

# Check and run vllm tests
exit_code=$(uv run --extra vllm pytest tests/unit/models/policy/ --collect-only --hf-gated --vllm-only -q >/dev/null 2>&1; echo $?)
if [[ $exit_code -eq 5 ]]; then
    echo "No vllm tests to run"
else
    uv run --extra vllm bash -x ./tests/run_unit.sh unit/models/policy/ --cov=nemo_rl --cov-append --cov-report=term-missing --cov-report=json --hf-gated --vllm-only
fi

# Check and run sglang tests
exit_code=$(uv run --extra sglang pytest tests/unit/models/policy/ --collect-only --hf-gated --sglang-only -q >/dev/null 2>&1; echo $?)
if [[ $exit_code -eq 5 ]]; then
    echo "No sglang tests to run"
else
    uv run --extra sglang bash -x ./tests/run_unit.sh unit/models/policy/ --cov=nemo_rl --cov-append --cov-report=term-missing --cov-report=json --hf-gated --sglang-only
fi
